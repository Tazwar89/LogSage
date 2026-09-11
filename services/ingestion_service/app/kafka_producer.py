"""
Kafka producer/consumer for asynchronous log ingestion.

Instead of /upload/logs synchronously parsing and storing every line in the
request/response cycle, this publishes each parsed log line to a Kafka topic.
A separate consumer worker (kafka_consumer.py) processes the topic
asynchronously and writes results into Redis via LogStore.

This decouples ingestion throughput from analysis/storage latency -- large
log files no longer block the HTTP request while every line is written to
Redis one by one.
"""
import json
import logging
import time
from typing import Any, cast

from kafka import KafkaProducer
from kafka.errors import KafkaError

from logsage_common.kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    LOG_INGESTION_TOPIC,
    KAFKA_SECURITY_KWARGS,
)

logger = logging.getLogger("ingestion_service")


def get_producer(max_retries: int = 10, initial_delay: float = 2.0) -> KafkaProducer:
    """
    Connects to Kafka with exponential backoff instead of failing immediately.

    Compose's `depends_on: condition: service_started` only waits for the
    Kafka *container* to start, not for the broker to finish loading and
    accept connections -- which can take several seconds after container
    start. Constructing KafkaProducer eagerly with no retry meant any
    timing mismatch (or a transient Kafka restart) permanently crashed this
    service instead of waiting it out.

    Catches the broad KafkaError base class rather than a specific subclass
    like NoBrokersAvailable, since that subclass's exact name/availability
    has varied across kafka-python versions/forks -- KafkaError is the
    stable base every connection-related exception inherits from.
    """
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=3,
                **cast(dict[str, Any], KAFKA_SECURITY_KWARGS),
            )
            logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS} on attempt {attempt}")

            return producer

        except KafkaError as e:
            last_error = e
            logger.warning(
                f"Kafka not available yet (attempt {attempt}/{max_retries}): {e}, "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, 30)  # cap backoff at 30s

    raise RuntimeError(
        f"Could not connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS} after {max_retries} attempts"
    ) from last_error


def publish_log_entry(producer: KafkaProducer, trace_id: str, entry: dict) -> bool:
    payload = {"trace_id": trace_id, "entry": entry}

    try:
        future = producer.send(LOG_INGESTION_TOPIC, value=payload)
        future.get(timeout=10)

        return True

    except KafkaError:
        return False


def publish_batch(producer: KafkaProducer, entries_with_ids: list) -> int:
    published = 0

    for trace_id, entry in entries_with_ids:
        if publish_log_entry(producer, trace_id, entry):
            published += 1

    producer.flush()

    return published