"""
Consumer Service

A standalone, non-HTTP microservice. Its only job is consuming from the
Kafka log-ingestion topic (published by ingestion-service) and writing
each entry into Redis (read by analysis-service).

Previously this ran as a background thread inside analysis-service's
FastAPI process. Splitting it into its own container means:
- It can be scaled independently of analysis-service's HTTP traffic
  (e.g. run 3 consumer replicas during a large batch ingest, 1 API replica).
- A crash or restart here never takes down the /analyze or /stats endpoints.
- analysis-service's FastAPI process is no longer running a background
  thread outside the request/response lifecycle, which was already an
  acknowledged compromise.

Run with: python -m app.main
"""
import json
import logging

from kafka import KafkaConsumer

from libs.logsage_common.logsage_common.log_store import LogStore
from libs.logsage_common.logsage_common.kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    LOG_INGESTION_TOPIC,
    CONSUMER_GROUP_ID,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer-service")


def run_consumer():
    logger.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}, topic={LOG_INGESTION_TOPIC}")

    consumer = KafkaConsumer(
        LOG_INGESTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v is not None else None,
        auto_offset_reset="earliest",
        group_id=CONSUMER_GROUP_ID,
    )
    log_store = LogStore()

    logger.info("Consumer started, waiting for messages...")

    for message in consumer:
        payload = message.value
        trace_id = payload["trace_id"]
        entry = payload["entry"]
        log_store.save(trace_id, entry)
        logger.info(f"Stored {trace_id}")


if __name__ == "__main__":
    run_consumer()