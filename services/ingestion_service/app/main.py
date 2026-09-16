"""
Ingestion Service

Responsibilities:
- POST /upload/baseline: parses a clean log file, mines templates via Drain3,
  rebuilds the shared Qdrant baseline collection.
- POST /upload/logs: parses a log file, scores each entry's mined template
  against the baseline via embedding-distance anomaly detection, and
  publishes only anomalous entries to Kafka for GenAI analysis -- cutting
  the volume of (costly) LLM diagnostic calls to the entries that actually
  look like something went wrong.
"""
from fastapi import FastAPI, UploadFile

from .parsing import parse_line
from .embedding import build_template_miner, deduplicate_logs, get_unique_templates
from .kafka_producer import get_producer, publish_batch
from .anomaly import BaselineAnomalyDetector, fetch_baseline_vectors, filter_anomalous_logs

from logsage_common.vector_store_qdrant import QdrantVectorStore

app = FastAPI(title="LogSage Ingestion Service")

template_miner = build_template_miner()
baseline_store = QdrantVectorStore(collection_name="baseline")
producer = get_producer()  # now retries with backoff instead of crashing immediately


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
    baseline_store.save()  # no-op for Qdrant; kept for interface parity

    return {"baseline_templates": len(templates)}


@app.post("/upload/logs")
async def upload_logs(file: UploadFile):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]

    deduplicate_logs(parsed_logs, template_miner)

    baseline_vectors = fetch_baseline_vectors(baseline_store)
    detector = BaselineAnomalyDetector().fit(baseline_vectors)
    anomalous_logs, scored_logs = filter_anomalous_logs(parsed_logs, baseline_store, detector)

    entries_with_ids = [(f"{file.filename}-{i}", entry) for i, entry in enumerate(anomalous_logs)]
    published = publish_batch(producer, entries_with_ids)

    return {
        "total_parsed": len(parsed_logs),
        "anomalous": len(anomalous_logs),
        "published_to_kafka": published,
        "suppressed_as_normal": len(parsed_logs) - len(anomalous_logs),
    }