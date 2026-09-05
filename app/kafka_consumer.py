"""
Kafka consumer worker.

Runs as a separate process (see docker-compose.yml `worker` service),
consuming from LOG_INGESTION_TOPIC and writing each entry into Redis via
LogStore -- decoupled from the FastAPI request/response cycle.

Run standalone with: python -m app.kafka_consumer
"""
import json
import os
from kafka import KafkaConsumer

from .log_store import LogStore
from .kafka_producer import KAFKA_BOOTSTRAP_SERVERS, LOG_INGESTION_TOPIC


def run_consumer():
    consumer = KafkaConsumer(
        LOG_INGESTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v is not None else None,
        auto_offset_reset="earliest",
        group_id="logsage-ingestion-workers",
    )
    log_store = LogStore()

    print(f"Consuming from topic '{LOG_INGESTION_TOPIC}'...")

    for message in consumer:
        payload = message.value

        # Guard against empty payloads inside message loop
        if payload is None:
            continue

        trace_id = payload["trace_id"]
        entry = payload["entry"]
        log_store.save(trace_id, entry)
        print(f"Stored {trace_id}")


if __name__ == "__main__":
    run_consumer()