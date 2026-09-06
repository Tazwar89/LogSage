"""
Ingestion Service

Responsibilities:
- POST /upload/baseline: parses a clean log file, mines templates via Drain3,
  rebuilds the shared FAISS baseline index, and persists it to the shared
  volume so analysis-service can load it.
- POST /upload/logs: parses a log file and publishes each line to Kafka for
  asynchronous processing by analysis-service's consumer.

This service does NOT perform anomaly detection, RAG, or LLM calls -- that's
analysis-service's job. Keeping this boundary strict is what makes the split
meaningful rather than cosmetic.
"""
from fastapi import FastAPI, UploadFile

from .parsing import parse_line
from .embedding import build_template_miner, deduplicate_logs, get_unique_templates
from .kafka_producer import get_producer, publish_batch

from libs.logsage_common.logsage_common.vector_store import VectorStore

app = FastAPI(title="LogSage Ingestion Service")

template_miner = build_template_miner()
baseline_store = VectorStore()
producer = get_producer()


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}


@app.post("/upload/baseline")
async def upload_baseline(file: UploadFile):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]

    deduplicate_logs(parsed_logs, template_miner)
    templates = get_unique_templates(template_miner)

    baseline_store.build_index(templates)
    baseline_store.save()  # persist to shared volume for analysis-service to load

    return {"baseline_templates": len(templates)}


@app.post("/upload/logs")
async def upload_logs(file: UploadFile):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]

    entries_with_ids = [(f"{file.filename}-{i}", entry) for i, entry in enumerate(parsed_logs)]
    published = publish_batch(producer, entries_with_ids)

    return {"published_to_kafka": published, "total_parsed": len(parsed_logs)}