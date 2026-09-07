"""
Consumer Service

A standalone, non-HTTP microservice. Its only job is consuming from the
Kafka log-ingestion topic (published by ingestion_service) and writing
each entry into Redis (read by analysis_service).
"""
import json
import logging
import time

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from logsage_common.log_store import LogStore
from logsage_common.kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    LOG_INGESTION_TOPIC,
    CONSUMER_GROUP_ID,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consumer-service")


def connect_with_retry(max_retries: int = 10, initial_delay: float = 2.0) -> KafkaConsumer:
    """
    Same rationale as ingestion_service's get_producer(): Compose's
    `depends_on: condition: service_started` doesn't guarantee Kafka is
    actually ready to accept connections yet, so this retries with
    exponential backoff instead of crashing on the first attempt.
    """
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                LOG_INGESTION_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v is not None else None,
                auto_offset_reset="earliest",
                group_id=CONSUMER_GROUP_ID,
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
        trace_id = payload["trace_id"]
        entry = payload["entry"]
        log_store.save(trace_id, entry)
        logger.info(f"Stored {trace_id}")


if __name__ == "__main__":
    run_consumer()