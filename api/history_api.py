import csv
import io
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from config.settings import HISTORY_FILE

router = APIRouter(prefix="/api/history", tags=["History"])


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(records: List[Dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history_record(
    query: str,
    engine: str,
    top_k: int,
    elapsed_ms: float,
    results: List[Dict[str, Any]],
    timestamp: str
) -> Dict[str, Any]:
    records = load_history()
    top_score = results[0]["score"] if results else 0.0
    top_source = results[0]["source"] if results else "N/A"
    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": timestamp,
        "query": query,
        "engine": engine,
        "top_k": top_k,
        "hit_count": len(results),
        "top_score": top_score,
        "top_source": top_source,
        "elapsed_ms": elapsed_ms
    }
    records.insert(0, record)
    save_history(records)
    return record


@router.get("")
def get_history_list(keyword: Optional[str] = None, engine: Optional[str] = None):
    records = load_history()
    if keyword:
        records = [r for r in records if keyword.lower() in r["query"].lower()]
    if engine:
        records = [r for r in records if r["engine"].lower() == engine.lower()]
    return {"total": len(records), "items": records}


@router.delete("/{record_id}")
def delete_history_item(record_id: str):
    records = load_history()
    filtered = [r for r in records if r["id"] != record_id]
    if len(filtered) == len(records):
        raise HTTPException(status_code=404, detail="记录不存在")
    save_history(filtered)
    return {"message": "删除成功", "remaining": len(filtered)}


@router.delete("")
def clear_all_history():
    save_history([])
    return {"message": "历史记录已清空"}


@router.get("/export")
def export_history_csv():
    records = load_history()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Query", "Engine", "TopK", "HitCount", "TopScore", "TopSource", "Elapsed(ms)"])
    for r in records:
        writer.writerow([
            r.get("id"), r.get("timestamp"), r.get("query"),
            r.get("engine"), r.get("top_k"), r.get("hit_count"),
            r.get("top_score"), r.get("top_source"), r.get("elapsed_ms")
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue().encode("utf-8-sig")]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=courseqa_history.csv"}
    )
