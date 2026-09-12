"""
Consumer Service

A standalone, non-HTTP microservice. Its only job is consuming from the
Kafka log-ingestion topic (published by ingestion_service) and writing
each entry into Redis (read by analysis_service).
"""
import json, logging, time
from typing import Any, cast
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from logsage_common.log_store import LogStore
from logsage_common.kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    LOG_INGESTION_TOPIC,
    CONSUMER_GROUP_ID,
    KAFKA_SECURITY_KWARGS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer_service")


def _safe_json_deserializer(v):
    if v is None:
        return None

    try:
        return json.loads(v.decode("utf-8"))

    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"__malformed__": True, "raw": v}


def connect_with_retry(max_retries: int = 10, initial_delay: float = 2.0) -> KafkaConsumer:
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                LOG_INGESTION_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=_safe_json_deserializer,
                auto_offset_reset="earliest",
                group_id=CONSUMER_GROUP_ID,
                **cast(dict[str, Any], KAFKA_SECURITY_KWARGS),
            )
            logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS} on attempt {attempt}")

            return consumer

        except KafkaError as e:
            last_error = e
            logger.warning(
                f"Kafka not available yet (attempt {attempt}/{max_retries}): {e}, "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 30)

    raise RuntimeError(
        f"Could not connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS} after {max_retries} attempts"
    ) from last_error


def run_consumer():
    logger.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}, topic={LOG_INGESTION_TOPIC}")

    consumer = connect_with_retry()
    log_store = LogStore()

    logger.info("Consumer started, waiting for messages...")

    for message in consumer:
        payload = message.value

        if not isinstance(payload, dict) or "__malformed__" in payload or "trace_id" not in payload:
            logger.warning(f"Skipping malformed message at offset {message.offset}")
            continue

        trace_id = payload["trace_id"]
        entry = payload["entry"]
        log_store.save(trace_id, entry)
        logger.info(f"Stored {trace_id}")


if __name__ == "__main__":
    run_consumer()