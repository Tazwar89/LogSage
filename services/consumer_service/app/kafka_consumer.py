"""
Kafka consumer worker.

Runs as a separate process (see docker-compose.yml `worker` service),
consuming from LOG_INGESTION_TOPIC and writing each entry into Redis via
LogStore -- decoupled from the FastAPI request/response cycle.

Run standalone with: python -m app.kafka_consumer
"""
import json
from typing import Any, cast
from venv import logger
from kafka import KafkaConsumer

from logsage_common.log_store import LogStore
from logsage_common.kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    LOG_INGESTION_TOPIC,
    CONSUMER_GROUP_ID,
    KAFKA_SECURITY_KWARGS,
)


def _safe_json_deserializer(v):
    if v is None:
        return None

    try:
        return json.loads(v.decode("utf-8"))

    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"__malformed__": True, "raw": v}


def run_consumer():
    consumer = KafkaConsumer(
        LOG_INGESTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=_safe_json_deserializer,
        auto_offset_reset="earliest",
        group_id=CONSUMER_GROUP_ID,
        **cast(dict[str, Any], KAFKA_SECURITY_KWARGS),
    )
    log_store = LogStore()

    logger.info(f"Consuming from topic '{LOG_INGESTION_TOPIC}'...")

    for message in consumer:
        payload = message.value

        # Guard against malformed payloads inside message loop
        if not isinstance(payload, dict) or "__malformed__" in payload or "trace_id" not in payload:
            logger.warning(f"Skipping malformed message at offset {message.offset}")
            continue

        trace_id = payload["trace_id"]
        entry = payload["entry"]
        log_store.save(trace_id, entry)
        logger.info(f"Stored {trace_id}")


if __name__ == "__main__":
    run_consumer()