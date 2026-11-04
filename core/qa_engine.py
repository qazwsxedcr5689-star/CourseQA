import re
import math
import time
from datetime import datetime
import jieba
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import MD_DIR, CHUNK_MIN_LENGTH, BM25_K1, BM25_B, logger

# 注入计算机与信息检索领域专有词汇，避免分词碎片化
CUSTOM_TERMS = [
    "Dijkstra", "Bellman-Ford", "Floyd", "A-Star", "Min-Heap", "Top-K",
    "TF-IDF", "BM25", "CountVectorizer", "RRF", "RAG", "MVCC", "ACID",
    "SOLID", "RESTful", "FastAPI", "LRU", "MRR", "NDCG",
    "最小堆", "优先队列", "最短路径", "哈夫曼树", "哈夫曼编码", "红黑树",
    "二叉搜索树", "平衡二叉树", "动态规划", "贪心算法", "倒排索引",
    "余弦相似度", "语义分块", "混合检索", "倒数排名融合", "零命中率",
    "多版本并发控制", "页面置换", "缺页中断", "银行家算法", "死锁"
]
for term in CUSTOM_TERMS:
    jieba.add_word(term)

SYNONYM_MAP: Dict[str, List[str]] = {
    "最短路": ["最短路径", "Dijkstra"],
    "dijkstra": ["最短路径", "单源最短路径"],
    "堆": ["最小堆", "优先队列"],
    "优先队列": ["最小堆", "Top-K"],
    "topk": ["Top-K", "最小堆"],
    "huffman": ["哈夫曼", "哈夫曼树", "哈夫曼编码", "无损压缩"],
    "哈夫曼": ["哈夫曼树", "哈夫曼编码", "前缀码"],
    "压缩": ["哈夫曼编码", "无损压缩"],
    "tfidf": ["TF-IDF", "余弦相似度", "逆文档频率"],
    "bm25": ["BM25", "概率检索", "词频饱和度"],
    "索引": ["倒排索引", "B+树"],
    "事务": ["ACID", "MVCC", "并发控制"],
    "死锁": ["银行家算法", "循环等待"],
    "设计模式": ["策略模式", "工厂模式", "观察者模式", "SOLID"]
}


def chinese_tokenizer(text: str) -> List[str]:
    text = re.sub(r"[^\w\s\u4e00-\u9fa5\+\-\*]", " ", text)
    tokens = []
    for tok in jieba.lcut(text):
        cleaned = tok.strip().lower()
        if len(cleaned) > 1 or cleaned in {"b", "k", "v", "e"}:
            tokens.append(cleaned)
    return tokens


def expand_query_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for syn in SYNONYM_MAP.get(tok.lower(), []):
            syn_low = syn.lower()
            if syn_low not in seen:
                expanded.append(syn_low)
                seen.add(syn_low)
    return expanded


class CourseQAEngine:
    """支持 Hybrid (RRF)、BM25、TF-IDF 与 CountVectorizer 的多引擎课程知识问答与检索核心"""

    def __init__(self, corpus_dir: Path = MD_DIR):
        self.corpus_dir = corpus_dir
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.count_vectorizer: Optional[CountVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.char_matrix = None
        self.doc_freqs: Dict[str, int] = {}
        self.tokenized_chunks: List[List[str]] = []
        self.avg_dl: float = 0.0
        self.last_reload_time: str = ""
        self.reload_corpus()

    def _parse_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        doc_title = file_path.stem
        current_title = file_path.stem
        buffer: List[str] = []
        extracted: List[Dict[str, Any]] = []

        def flush_buffer(title: str, buf: List[str]):
            text = "\n".join(buf).strip()
            if len(text) >= CHUNK_MIN_LENGTH:
                extracted.append({
                    "chunk_id": f"{file_path.stem}#{len(extracted) + 1}",
                    "source": file_path.name,
                    "doc_title": doc_title,
                    "section": title,
                    "content": text,
                    "char_count": len(text)
                })

        for line in lines:
            stripped = line.strip()
            h1_match = re.match(r"^#\s+(.*)", stripped)
            if h1_match:
                if buffer:
                    flush_buffer(current_title, buffer)
                    buffer = []
                doc_title = h1_match.group(1).strip()
                current_title = doc_title
                continue

            header_match = re.match(r"^(#{2,3})\s+(.*)", stripped)
            if header_match:
                if buffer:
                    flush_buffer(current_title, buffer)
                    buffer = []
                current_title = header_match.group(2).strip()
            else:
                if stripped == "" and len("\n".join(buffer)) >= 220:
                    flush_buffer(current_title, buffer)
                    buffer = []
                elif stripped:
                    buffer.append(stripped)

        if buffer:
            flush_buffer(current_title, buffer)
        return extracted

    def reload_corpus(self) -> Dict[str, Any]:
        self.chunks = []
        md_files = sorted(self.corpus_dir.glob("*.md"))
        for md_file in md_files:
            self.chunks.extend(self._parse_markdown(md_file))

        self.last_reload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.chunks:
            logger.warning("未检测到有效 Markdown 语料，请检查 data/md 目录。")
            return {"files": 0, "chunks": 0, "vocab_size": 0, "reloaded_at": self.last_reload_time}

        texts = [f"{c['doc_title']} {c['section']} {c['content']}" for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
        self.count_vectorizer = CountVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.count_matrix = self.count_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

        self.tokenized_chunks = [chinese_tokenizer(t) for t in texts]
        total_len = sum(len(tokens) for tokens in self.tokenized_chunks)
        self.avg_dl = total_len / len(self.tokenized_chunks) if self.tokenized_chunks else 1.0
        self.doc_freqs = {}
        for tokens in self.tokenized_chunks:
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        vocab_size = len(self.doc_freqs)
        logger.info(f"语料加载完成：共 {len(md_files)} 个文件，切分为 {len(self.chunks)} 个知识块，词表规模 {vocab_size}。")
        return {
            "files": len(md_files),
            "chunks": len(self.chunks),
            "vocab_size": vocab_size,
            "reloaded_at": self.last_reload_time
        }

    def _compute_bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        n_docs = len(self.tokenized_chunks)
        scores = np.zeros(n_docs, dtype=float)
        expanded_tokens = expand_query_tokens(query_tokens)
        raw_set = set(query_tokens)

        for idx, doc_tokens in enumerate(self.tokenized_chunks):
            dl = len(doc_tokens)
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for q in expanded_tokens:
                if q not in tf_map:
                    continue
                weight = 1.0 if q in raw_set else 0.55
                df = self.doc_freqs.get(q, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_map[q]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avg_dl)
                score += weight * idf * (tf * (BM25_K1 + 1.0)) / denom
            scores[idx] = score
        return scores

    def _compute_hybrid_scores(self, query: str, q_tokens: List[str]) -> np.ndarray:
        """基于倒数排名融合 (RRF) + 归一化得分加权的混合检索引擎"""
        n_docs = len(self.chunks)
        bm25_raw = self._compute_bm25_scores(q_tokens)
        q_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_raw = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
        q_char = self.char_vectorizer.transform([query])
        char_raw = cosine_similarity(q_char, self.char_matrix).flatten()

        def normalize(arr: np.ndarray) -> np.ndarray:
            max_v = float(np.max(arr)) if len(arr) else 0.0
            return arr / max_v if max_v > 1e-9 else arr

        bm25_norm = normalize(bm25_raw)
        tfidf_norm = normalize(tfidf_raw)
        char_norm = normalize(char_raw)

        rrf_scores = np.zeros(n_docs, dtype=float)
        rrf_k = 60.0
        for arr in (bm25_raw, tfidf_raw, char_raw):
            if np.max(arr) <= 0:
                continue
            ranks = np.argsort(arr)[::-1]
            for rank_pos, doc_idx in enumerate(ranks, start=1):
                if arr[doc_idx] > 0:
                    rrf_scores[doc_idx] += 1.0 / (rrf_k + rank_pos)

        rrf_norm = normalize(rrf_scores)
        hybrid = 0.45 * bm25_norm + 0.30 * tfidf_norm + 0.15 * char_norm + 0.10 * rrf_norm
        # 若全路均为0则保持为0
        zero_mask = (bm25_raw <= 0) & (tfidf_raw <= 0) & (char_raw <= 0.05)
        hybrid[zero_mask] = 0.0
        return hybrid

    def _extract_matched_terms(self, q_tokens: List[str], chunk: Dict[str, Any]) -> List[str]:
        haystack = f"{chunk['section']} {chunk['content']}".lower()
        matched = []
        for tok in expand_query_tokens(q_tokens):
            if len(tok) >= 2 and tok in haystack and tok not in matched:
                matched.append(tok)
        return matched[:8]

    def search(self, query: str, engine: str = "tfidf", top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or not query.strip():
            return []

        engine = engine.lower()
        q_tokens = chinese_tokenizer(query)
        expanded_query = " ".join(expand_query_tokens(q_tokens)) or query

        if engine == "count":
            q_vec = self.count_vectorizer.transform([expanded_query])
            scores = cosine_similarity(q_vec, self.count_matrix).flatten()
        elif engine == "bm25":
            scores = self._compute_bm25_scores(q_tokens)
        elif engine == "hybrid":
            scores = self._compute_hybrid_scores(query, q_tokens)
        else:
            q_vec = self.tfidf_vectorizer.transform([expanded_query])
            scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score_val = float(scores[idx])
            if score_val <= 0:
                continue
            chunk = self.chunks[idx]
            matched_terms = self._extract_matched_terms(q_tokens, chunk)
            results.append({
                "rank": rank,
                "score": round(score_val, 4),
                "chunk_id": chunk.get("chunk_id", f"chunk-{idx}"),
                "source": chunk["source"],
                "doc_title": chunk.get("doc_title", chunk["source"]),
                "section": chunk["section"],
                "content": chunk["content"],
                "char_count": chunk.get("char_count", len(chunk["content"])),
                "matched_terms": matched_terms
            })
        return results

    def synthesize_answer(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于检索出的 Top-K 知识块生成抽取式课程问答综合解答、引用溯源与追问建议"""
        if not results:
            return {
                "has_answer": False,
                "confidence": "low",
                "confidence_score": 0,
                "summary": "当前课程知识库中未检索到与该问题直接匹配的章节片段。建议尝试切换为「Hybrid 混合检索」或「BM25 模型」，或使用更具体的课程核心术语（如：Dijkstra、最小堆、BM25、B+树、MVCC）。",
                "key_points": [],
                "citations": [],
                "related_questions": [
                    "Dijkstra最短路径算法的时间复杂度与适用条件是什么？",
                    "为什么海量数据Top-K推荐使用最小堆而不是全局排序？",
                    "BM25检索模型相比传统TF-IDF做了哪些核心改进？"
                ]
            }

        top1 = results[0]
        q_tokens = set(expand_query_tokens(chinese_tokenizer(query)))

        # 评估置信度
        matched_cnt = len(top1.get("matched_terms", []))
        raw_score = top1["score"]
        conf_pct = min(98, max(45, int(55 + min(raw_score, 4.0) * 8 + matched_cnt * 6)))
        confidence = "high" if conf_pct >= 80 else ("medium" if conf_pct >= 62 else "low")

        # 从前 3 个高分片段中抽取最契合的句子
        key_points = []
        citations = []
        for idx, item in enumerate(results[:3], start=1):
            citations.append({
                "index": idx,
                "source": item["source"],
                "doc_title": item.get("doc_title", item["source"]),
                "section": item["section"],
                "score": item["score"]
            })
            sentences = [s.strip() for s in re.split(r"[。！？\n]+", item["content"]) if len(s.strip()) >= 12]
            if not sentences:
                continue
            scored_sents = []
            for s in sentences:
                s_low = s.lower()
                overlap = sum(1 for tok in q_tokens if tok in s_low)
                scored_sents.append((overlap, len(s), s))
            scored_sents.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_sent = scored_sents[0][2]
            key_points.append({
                "citation_index": idx,
                "section": item["section"],
                "source": item["source"],
                "text": best_sent + ("。" if not best_sent.endswith("。") else "")
            })

        summary_lead = f"根据《{top1.get('doc_title', top1['source'])}》中「{top1['section']}」章节及相关课程讲义整理："
        summary_body = " ".join(
            f"{kp['text']} [{kp['citation_index']}]" for kp in key_points[:2]
        )

        # 生成同文档或相邻章节的关联追问
        hit_sections = {r["section"] for r in results}
        related_sections = [
            c["section"] for c in self.chunks
            if c["section"] not in hit_sections
        ]
        unique_related = []
        for sec in related_sections:
            q_text = f"请解析「{sec}」的核心原理与应用场景？"
            if q_text not in unique_related:
                unique_related.append(q_text)
            if len(unique_related) >= 3:
                break

        return {
            "has_answer": True,
            "confidence": confidence,
            "confidence_score": conf_pct,
            "summary": f"{summary_lead}\n{summary_body}",
            "key_points": key_points,
            "citations": citations,
            "related_questions": unique_related
        }

    def compare_engines(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """在同一查询下并行对比 Hybrid、BM25、TF-IDF、Count 四种检索引擎的表现"""
        engines = [
            ("hybrid", "Hybrid (RRF 混合检索)"),
            ("bm25", "BM25 概率检索模型"),
            ("tfidf", "TF-IDF + 余弦相似度"),
            ("count", "CountVectorizer 词频向量")
        ]
        comparison = []
        base_set = set()

        for eng_key, eng_label in engines:
            t0 = time.perf_counter()
            res = self.search(query, engine=eng_key, top_k=top_k)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            chunk_ids = {r["chunk_id"] for r in res}
            if eng_key == "hybrid":
                base_set = chunk_ids
            overlap = (
                round(len(chunk_ids & base_set) / max(len(chunk_ids | base_set), 1) * 100, 1)
                if chunk_ids or base_set else 100.0
            )
            comparison.append({
                "engine": eng_key,
                "label": eng_label,
                "elapsed_ms": elapsed,
                "hit_count": len(res),
                "top_score": res[0]["score"] if res else 0.0,
                "top_section": res[0]["section"] if res else "未命中",
                "overlap_with_hybrid_pct": overlap,
                "results": res
            })
        return {
            "query": query,
            "top_k": top_k,
            "total_chunks": len(self.chunks),
            "engines": comparison
        }

    def batch_search(self, queries: List[str], engine: str = "tfidf", top_k: int = 3) -> List[Dict[str, Any]]:
        items = []
        for q in queries:
            if not q.strip():
                continue
            t0 = time.perf_counter()
            res = self.search(q, engine=engine, top_k=top_k)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            items.append({
                "query": q.strip(),
                "elapsed_ms": elapsed,
                "hit_count": len(res),
                "top_score": res[0]["score"] if res else 0.0,
                "top_section": res[0]["section"] if res else "无命中",
                "results": res
            })
        return items

    def get_corpus_documents(self) -> List[Dict[str, Any]]:
        docs = []
        for md_file in sorted(self.corpus_dir.glob("*.md")):
            file_chunks = [c for c in self.chunks if c["source"] == md_file.name]
            stat = md_file.stat()
            docs.append({
                "filename": md_file.name,
                "doc_title": file_chunks[0]["doc_title"] if file_chunks else md_file.stem,
                "size_bytes": stat.st_size,
                "chunk_count": len(file_chunks),
                "sections": [c["section"] for c in file_chunks],
                "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        return docs

    def get_document_detail(self, filename: str) -> Dict[str, Any]:
        safe_name = Path(filename).name
        target = self.corpus_dir / safe_name
        if not target.exists() or target.suffix.lower() != ".md":
            raise FileNotFoundError(f"文档不存在: {safe_name}")
        file_chunks = [c for c in self.chunks if c["source"] == safe_name]
        return {
            "filename": safe_name,
            "content": target.read_text(encoding="utf-8"),
            "chunk_count": len(file_chunks),
            "chunks": file_chunks
        }

    def save_document(self, filename: str, content: str) -> Dict[str, Any]:
        safe_name = Path(filename).name
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        target = self.corpus_dir / safe_name
        target.write_text(content.strip() + "\n", encoding="utf-8")
        reload_info = self.reload_corpus()
        return {"filename": safe_name, **reload_info}

    def delete_document(self, filename: str) -> Dict[str, Any]:
        safe_name = Path(filename).name
        target = self.corpus_dir / safe_name
        if not target.exists():
            raise FileNotFoundError(f"文档不存在: {safe_name}")
        target.unlink()
        reload_info = self.reload_corpus()
        return {"deleted": safe_name, **reload_info}
