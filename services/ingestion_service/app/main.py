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
- POST /upload/logs/sequence: groups a log file's lines into sessions (HDFS
  BlockIds), scores each session's event sequence with a DeepLog-style LSTM,
  and publishes one entry per anomalous session. Catches anomalies the
  per-line detector cannot see (missing/reordered/truncated event sequences
  built entirely from individually normal lines).
- GET /sequence/status: whether a trained sequence model is loaded/available.
"""
import os
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .parsing import parse_line
from .embedding import build_template_miner, deduplicate_logs, get_unique_templates
from .kafka_producer import get_producer, publish_batch
from .anomaly import BaselineAnomalyDetector, fetch_baseline_vectors, filter_anomalous_logs
from .sequence import SequenceScorer, build_block_entry

from logsage_common.vector_store_qdrant import QdrantVectorStore

app = FastAPI(title="LogSage Ingestion Service")

template_miner = build_template_miner()
baseline_store = QdrantVectorStore(collection_name="baseline")
producer = get_producer()  # now retries with backoff instead of crashing immediately
sequence_scorer = SequenceScorer()  # loads the trained model lazily on first use


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}


@app.get("/sequence/status")
def sequence_status():
    return sequence_scorer.status()


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


@app.post("/upload/logs/sequence")
async def upload_logs_sequence(file: UploadFile):
    if not sequence_scorer.available:
        raise HTTPException(
            status_code=503,
            detail=f"No sequence model found in {sequence_scorer.model_dir} -- "
                   "train one with scripts/train_sequence_model.py and bake it into the image",
        )

    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    parsed_logs = [p for p in (parse_line(l) for l in lines if l.strip()) if p]

    results = await run_in_threadpool(sequence_scorer.score_logs, parsed_logs)
    anomalous = [r for r in results if r.is_anomalous]

    entries_with_ids = [(f"{file.filename}-{r.block_id}", build_block_entry(parsed_logs, r)) for r in anomalous]
    published = publish_batch(producer, entries_with_ids)

    ids = [r.block_id for r in anomalous]
    cap = int(os.getenv("MAX_RETURNED_BLOCK_IDS", "500"))

    return {
        "total_parsed": len(parsed_logs),
        "blocks_scored": len(results),
        "anomalous_blocks": len(anomalous),
        "published_to_kafka": published,
        "suppressed_as_normal": len(results) - len(anomalous),
        "anomalous_block_ids": ids[:cap],
        "anomalous_block_ids_returned": min(len(ids), cap),
        "anomalous_block_ids_truncated": len(ids) > cap,
    }