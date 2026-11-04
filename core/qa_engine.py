import re
import math
import jieba
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import MD_DIR, CHUNK_MIN_LENGTH, BM25_K1, BM25_B, logger


def chinese_tokenizer(text: str) -> List[str]:
    text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)
    return [tok.strip() for tok in jieba.lcut(text) if len(tok.strip()) > 1]


class CourseQAEngine:
    """支持 TF-IDF、CountVectorizer 与 BM25 的多引擎课程知识检索核心"""

    def __init__(self, corpus_dir: Path = MD_DIR):
        self.corpus_dir = corpus_dir
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.doc_freqs: Dict[str, int] = {}
        self.tokenized_chunks: List[List[str]] = []
        self.avg_dl: float = 0.0
        self.reload_corpus()

    def _parse_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        current_title = file_path.stem
        buffer = []
        extracted = []

        def flush_buffer(title: str, buf: List[str]):
            text = "\n".join(buf).strip()
            if len(text) >= CHUNK_MIN_LENGTH:
                extracted.append({
                    "source": file_path.name,
                    "section": title,
                    "content": text
                })

        for line in lines:
            header_match = re.match(r"^(#{1,3})\s+(.*)", line.strip())
            if header_match:
                if buffer:
                    flush_buffer(current_title, buffer)
                    buffer = []
                current_title = header_match.group(2).strip()
            else:
                if line.strip() == "" and len("\n".join(buffer)) >= 180:
                    flush_buffer(current_title, buffer)
                    buffer = []
                elif line.strip():
                    buffer.append(line.strip())

        if buffer:
            flush_buffer(current_title, buffer)
        return extracted

    def reload_corpus(self) -> Dict[str, int]:
        self.chunks = []
        md_files = sorted(self.corpus_dir.glob("*.md"))
        for md_file in md_files:
            self.chunks.extend(self._parse_markdown(md_file))

        if not self.chunks:
            logger.warning("未检测到有效 Markdown 语料，请检查 data/md 目录。")
            return {"files": 0, "chunks": 0}

        texts = [f"{c['section']} {c['content']}" for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)
        self.count_vectorizer = CountVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.count_matrix = self.count_vectorizer.fit_transform(texts)

        self.tokenized_chunks = [chinese_tokenizer(t) for t in texts]
        total_len = sum(len(tokens) for tokens in self.tokenized_chunks)
        self.avg_dl = total_len / len(self.tokenized_chunks) if self.tokenized_chunks else 1.0
        self.doc_freqs = {}
        for tokens in self.tokenized_chunks:
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        logger.info(f"语料加载完成：共 {len(md_files)} 个文件，切分为 {len(self.chunks)} 个知识块。")
        return {"files": len(md_files), "chunks": len(self.chunks)}

    def _compute_bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        n_docs = len(self.tokenized_chunks)
        scores = np.zeros(n_docs, dtype=float)
        for idx, doc_tokens in enumerate(self.tokenized_chunks):
            dl = len(doc_tokens)
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for q in query_tokens:
                if q not in tf_map:
                    continue
                df = self.doc_freqs.get(q, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_map[q]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avg_dl)
                score += idf * (tf * (BM25_K1 + 1.0)) / denom
            scores[idx] = score
        return scores

    def search(self, query: str, engine: str = "tfidf", top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or not query.strip():
            return []

        engine = engine.lower()
        if engine == "count":
            q_vec = self.count_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.count_matrix).flatten()
        elif engine == "bm25":
            q_tokens = chinese_tokenizer(query)
            scores = self._compute_bm25_scores(q_tokens)
        else:
            q_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score_val = float(scores[idx])
            if score_val <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": round(score_val, 4),
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"]
            })
        return results

    def batch_search(self, queries: List[str], engine: str = "tfidf", top_k: int = 3) -> List[Dict[str, Any]]:
        return [
            {"query": q, "results": self.search(q, engine=engine, top_k=top_k)}
            for q in queries if q.strip()
        ]
