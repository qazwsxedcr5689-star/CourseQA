import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.qa_engine import CourseQAEngine


class TestCourseQAEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CourseQAEngine()

    def test_corpus_loaded_and_chunked(self):
        docs = self.engine.get_corpus_documents()
        self.assertGreaterEqual(len(docs), 4)
        self.assertGreaterEqual(len(self.engine.chunks), 20)
        self.assertGreater(len(self.engine.doc_freqs), 50)

    def test_all_four_engines_search(self):
        query = "Dijkstra最短路径算法的时间复杂度是多少？"
        for eng in ("hybrid", "bm25", "tfidf", "count"):
            results = self.engine.search(query, engine=eng, top_k=3)
            self.assertGreater(len(results), 0, f"Engine {eng} returned empty results")
            self.assertIn("Dijkstra", results[0]["content"] + results[0]["section"])
            self.assertGreater(results[0]["score"], 0)

    def test_extractive_answer_synthesis(self):
        query = "为什么海量数据Top-K推荐使用最小堆而不是全局排序？"
        results = self.engine.search(query, engine="hybrid", top_k=3)
        qa = self.engine.synthesize_answer(query, results)
        self.assertTrue(qa["has_answer"])
        self.assertGreaterEqual(qa["confidence_score"], 60)
        self.assertGreater(len(qa["key_points"]), 0)
        self.assertGreater(len(qa["citations"]), 0)

    def test_compare_engines_arena(self):
        comp = self.engine.compare_engines("BM25检索模型相比TF-IDF有什么改进？", top_k=3)
        self.assertEqual(len(comp["engines"]), 4)
        engine_names = [e["engine"] for e in comp["engines"]]
        self.assertEqual(engine_names, ["hybrid", "bm25", "tfidf", "count"])

    def test_batch_search(self):
        queries = [
            "哈夫曼树如何实现无损压缩编码？",
            "数据库事务ACID四大特性与MVCC原理",
            "操作系统死锁的四个必要条件"
        ]
        batch = self.engine.batch_search(queries, engine="hybrid", top_k=2)
        self.assertEqual(len(batch), 3)
        for item in batch:
            self.assertGreater(item["hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
