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
import os
from kafka import KafkaProducer
from kafka.errors import KafkaError

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
LOG_INGESTION_TOPIC = "logsage.logs.raw"


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
    )


def publish_log_entry(producer: KafkaProducer, trace_id: str, entry: dict) -> bool:
    """Publishes a single parsed log entry to the ingestion topic. Returns True on success."""
    payload = {"trace_id": trace_id, "entry": entry}

    try:
        future = producer.send(LOG_INGESTION_TOPIC, value=payload)
        future.get(timeout=10)  # block until ack, so caller knows if publish failed

        return True

    except KafkaError:
        return False


def publish_batch(producer: KafkaProducer, entries_with_ids: list[tuple[str, dict]]) -> int:
    """Publishes a batch of (trace_id, entry) pairs. Returns count of successful publishes."""
    published = 0

    for trace_id, entry in entries_with_ids:
        if publish_log_entry(producer, trace_id, entry):
            published += 1

    producer.flush()

    return published