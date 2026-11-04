from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import STATIC_DIR, DEFAULT_TOP_K
from core.qa_engine import CourseQAEngine
from utils.time_utils import Timer, get_current_timestamp
from api.history_api import router as history_router, append_history_record
from api.stats_api import router as stats_router

app = FastAPI(
    title="CourseQA 智能课程问答与多引擎信息检索系统",
    description="支持 Hybrid(RRF) / BM25 / TF-IDF / CountVectorizer 四引擎检索、抽取式问答溯源、多引擎竞技场与在线知识库热重载",
    version="3.0.0"
)
qa_engine = CourseQAEngine()

app.include_router(history_router)
app.include_router(stats_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    query: str
    engine: str = "hybrid"
    top_k: int = DEFAULT_TOP_K


class CompareRequest(BaseModel):
    query: str
    top_k: int = 5


class BatchQueryRequest(BaseModel):
    queries: List[str]
    engine: str = "hybrid"
    top_k: int = 3


class DocumentSaveRequest(BaseModel):
    filename: str
    content: str


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
        qa_answer = qa_engine.synthesize_answer(req.query, results)
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
        "engine": req.engine,
        "elapsed_ms": t.elapsed_ms,
        "total_chunks": len(qa_engine.chunks),
        "total_files": len(qa_engine.get_corpus_documents()),
        "vocab_size": len(qa_engine.doc_freqs),
        "results": results,
        "qa_answer": qa_answer
    }


@app.post("/api/compare")
def compare_engines_endpoint(req: CompareRequest):
    with Timer() as t:
        comparison = qa_engine.compare_engines(req.query, top_k=req.top_k)
    return {
        "total_elapsed_ms": t.elapsed_ms,
        **comparison
    }


@app.post("/api/batch_search")
def batch_search_endpoint(req: BatchQueryRequest):
    with Timer() as t:
        batch_results = qa_engine.batch_search(req.queries, engine=req.engine, top_k=req.top_k)
    hit_queries = sum(1 for item in batch_results if len(item.get("results", [])) > 0)
    return {
        "elapsed_ms": t.elapsed_ms,
        "count": len(batch_results),
        "hit_rate": round(hit_queries / max(len(batch_results), 1), 4),
        "items": batch_results
    }


@app.post("/api/reload")
def reload_corpus_endpoint():
    info = qa_engine.reload_corpus()
    return {"message": "知识库热重载成功", **info}


@app.get("/api/corpus")
def list_corpus_documents():
    docs = qa_engine.get_corpus_documents()
    return {
        "total_files": len(docs),
        "total_chunks": len(qa_engine.chunks),
        "vocab_size": len(qa_engine.doc_freqs),
        "reloaded_at": qa_engine.last_reload_time,
        "documents": docs
    }


@app.get("/api/corpus/{filename}")
def get_corpus_document(filename: str):
    try:
        return qa_engine.get_document_detail(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/corpus")
def save_corpus_document(req: DocumentSaveRequest):
    if not req.filename.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="文件名与内容不能为空")
    info = qa_engine.save_document(req.filename, req.content)
    return {"message": "文档保存并热重载成功", **info}


@app.delete("/api/corpus/{filename}")
def delete_corpus_document(filename: str):
    try:
        info = qa_engine.delete_document(filename)
        return {"message": "文档已删除并完成索引更新", **info}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
