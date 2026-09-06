"""
Analysis Service

Responsibilities:
- GET /analyze/{trace_id}: loads the shared baseline FAISS index from disk,
  runs anomaly detection, RAG retrieval, and the LangGraph agentic diagnostic
  pipeline for anomalous entries.
- GET /logs: lists stored trace IDs.
- GET /stats: Pandas-based aggregate analytics.

Log ingestion into Redis is handled entirely by the separate
consumer-service (see services/consumer-service) -- this service only ever
reads from Redis, never writes to it. This keeps analysis-service's FastAPI
process purely request/response, with no background threads competing with
the event loop.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from .anomaly import is_anomalous
from .rag import load_knowledge_base, build_kb_index, retrieve_context
from .agentic_pipeline import run_diagnostic_pipeline
from .analytics import compute_log_stats

from ..libs.logsage_common.logsage_common.vector_store import VectorStore
from ..libs.logsage_common.logsage_common.log_store import LogStore


baseline_store = VectorStore()
kb_store = VectorStore()
kb_lookup = {}
log_store = LogStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb_lookup
    kb_entries = load_knowledge_base()
    kb_lookup = build_kb_index(kb_store, kb_entries)
    yield


app = FastAPI(title="LogSage", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "analysis"}


@app.get("/analyze/{trace_id}")
def analyze(trace_id: str):
    entry = log_store.get(trace_id)
    if not entry:
        raise HTTPException(status_code=404, detail="trace_id not found")

    # Reload the baseline index fresh on each request so we always reflect
    # the latest /upload/baseline call from ingestion-service, without
    # needing any direct coupling between the two services.
    loaded = baseline_store.load()
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail="No baseline index available yet -- call ingestion-service's /upload/baseline first",
        )

    anomalous, nearest = is_anomalous(entry["message"], baseline_store, threshold=0.6)

    if not anomalous:
        return {"trace_id": trace_id, "anomalous": False, "nearest_match": nearest}

    result = run_diagnostic_pipeline(entry["message"], kb_store, kb_lookup)
    return {"trace_id": trace_id, "anomalous": True, **result}


@app.get("/logs")
def list_logs():
    return log_store.list_trace_ids()


@app.get("/stats")
def stats():
    entries = [log_store.get(tid) for tid in log_store.list_trace_ids()]
    return compute_log_stats(entries)