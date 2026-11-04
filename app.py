from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import STATIC_DIR, DEFAULT_TOP_K
from core.qa_engine import CourseQAEngine
from utils.time_utils import Timer, get_current_timestamp
from api.history_api import router as history_router, append_history_record
from api.stats_api import router as stats_router

app = FastAPI(title="CourseQA 智能课程问答与检索引擎", version="2.0.0")
qa_engine = CourseQAEngine()

app.include_router(history_router)
app.include_router(stats_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    query: str
    engine: str = "tfidf"
    top_k: int = DEFAULT_TOP_K


class BatchQueryRequest(BaseModel):
    queries: List[str]
    engine: str = "tfidf"
    top_k: int = 3


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.post("/api/search")
def search_endpoint(req: QueryRequest):
    with Timer() as t:
        results = qa_engine.search(req.query, engine=req.engine, top_k=req.top_k)
    record = append_history_record(
        query=req.query,
        engine=req.engine,
        top_k=req.top_k,
        elapsed_ms=t.elapsed_ms,
        results=results,
        timestamp=get_current_timestamp()
    )
    return {
        "query_id": record["id"],
        "elapsed_ms": t.elapsed_ms,
        "total_chunks": len(qa_engine.chunks),
        "results": results
    }


@app.post("/api/batch_search")
def batch_search_endpoint(req: BatchQueryRequest):
    with Timer() as t:
        batch_results = qa_engine.batch_search(req.queries, engine=req.engine, top_k=req.top_k)
    return {
        "elapsed_ms": t.elapsed_ms,
        "count": len(batch_results),
        "items": batch_results
    }


@app.post("/api/reload")
def reload_corpus_endpoint():
    info = qa_engine.reload_corpus()
    return {"message": "知识库热重载成功", **info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
