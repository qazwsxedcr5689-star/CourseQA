from collections import Counter
from fastapi import APIRouter
from api.history_api import load_history

router = APIRouter(prefix="/api/stats", tags=["Statistics"])


@router.get("")
def get_system_stats():
    records = load_history()
    total_queries = len(records)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "avg_elapsed_ms": 0.0,
            "avg_top_score": 0.0,
            "zero_hit_rate": 0.0,
            "engine_distribution": {},
            "top_queries": []
        }

    avg_elapsed = round(sum(r.get("elapsed_ms", 0.0) for r in records) / total_queries, 2)
    avg_score = round(sum(r.get("top_score", 0.0) for r in records) / total_queries, 4)
    zero_hits = sum(1 for r in records if r.get("hit_count", 0) == 0)
    engine_counts = dict(Counter(r.get("engine", "tfidf") for r in records))
    query_counts = Counter(r.get("query", "") for r in records).most_common(5)

    return {
        "total_queries": total_queries,
        "avg_elapsed_ms": avg_elapsed,
        "avg_top_score": avg_score,
        "zero_hit_rate": round(zero_hits / total_queries, 4),
        "engine_distribution": engine_counts,
        "top_queries": [{"query": q, "count": c} for q, c in query_counts]
    }
