"""
Analysis Service (REST API)

Versioned API under /api/v1 (typed request/response models, pagination,
optional API-key auth). The original unversioned routes (/analyze/{id},
/logs, /stats, /health) are kept as deprecated aliases so existing smoke
tests and clients keep working.

Log ingestion into Redis is handled entirely by consumer_service -- this
service only reads from Redis, never writes to it.
"""
import os, secrets
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from .anomaly import is_anomalous
from .rag import load_knowledge_base, build_kb_index, retrieve_context
from .agentic_pipeline import run_diagnostic_pipeline
from .analytics import compute_log_stats

from logsage_common.vector_store_qdrant import QdrantVectorStore
from logsage_common.log_store import LogStore


baseline_store = QdrantVectorStore(collection_name="baseline")
kb_store = QdrantVectorStore(collection_name="knowledge_base")
kb_lookup = {}
log_store = LogStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb_lookup
    kb_entries = load_knowledge_base()
    kb_lookup = build_kb_index(kb_store, kb_entries)
    yield


# ---------- Schemas ----------

class HealthResponse(BaseModel):
    status: str
    service: str


class AnalyzeResponse(BaseModel):
    """Diagnostic pipeline fields (root cause, fix, ...) pass through as extras."""
    model_config = ConfigDict(extra="allow")

    trace_id: str
    anomalous: bool
    nearest_match: Optional[dict[str, Any]] = None


class LogListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[str]


# ---------- Auth ----------

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Enforced only when LOGSAGE_API_KEY is set, so local/CI runs stay open."""
    expected = os.getenv("LOGSAGE_API_KEY")

    if not expected:
        return

    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ---------- Shared logic ----------

def _decode(tid) -> str:
    return tid.decode("utf-8") if isinstance(tid, bytes) else tid


def _analyze(trace_id: str) -> dict:
    entry = log_store.get(trace_id)

    if not entry:
        raise HTTPException(status_code=404, detail="trace_id not found")

    try:
        baseline_store.load()

    except (FileNotFoundError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="No baseline index available yet -- call ingestion_service's /upload/baseline first",
        )

    anomalous, nearest = is_anomalous(entry["message"], baseline_store)

    if not anomalous:
        return {"trace_id": trace_id, "anomalous": False, "nearest_match": nearest}

    result = run_diagnostic_pipeline(entry["message"], kb_store, kb_lookup)

    return {"trace_id": trace_id, "anomalous": True, **result}


def _stats() -> dict:
    entries = [
        entry for raw_tid in log_store.list_trace_ids()
        if (tid := _decode(raw_tid)) and (entry := log_store.get(tid)) is not None
    ]

    return compute_log_stats(entries)


# ---------- App + routes ----------

app = FastAPI(title="LogSage", version="1.0.0", lifespan=lifespan)
v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["v1"])


@v1.get("/health", response_model=HealthResponse)
def health_v1():
    return {"status": "ok", "service": "analysis"}


@v1.get("/analyze/{trace_id}", response_model=AnalyzeResponse)
def analyze_v1(trace_id: str):
    return _analyze(trace_id)


@v1.get("/logs", response_model=LogListResponse)
def list_logs_v1(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    ids = [_decode(t) for t in log_store.list_trace_ids()]

    return {"total": len(ids), "limit": limit, "offset": offset, "items": ids[offset:offset + limit]}


@v1.get("/stats")
def stats_v1():
    return _stats()


app.include_router(v1)


# Deprecated unversioned aliases (kept for the CI smoke test and existing clients).
@app.get("/health", response_model=HealthResponse, deprecated=True, tags=["legacy"])
def health():
    return {"status": "ok", "service": "analysis"}


@app.get("/analyze/{trace_id}", deprecated=True, tags=["legacy"], dependencies=[Depends(require_api_key)])
def analyze(trace_id: str):
    return _analyze(trace_id)


@app.get("/logs", deprecated=True, tags=["legacy"], dependencies=[Depends(require_api_key)])
def list_logs():
    return log_store.list_trace_ids()


@app.get("/stats", deprecated=True, tags=["legacy"], dependencies=[Depends(require_api_key)])
def stats():
    return _stats()