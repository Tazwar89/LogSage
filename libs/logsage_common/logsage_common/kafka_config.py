"""
Single source of truth for Kafka topic names and broker config.

Both ingestion_service (producer) and analysis_service (consumer) import
from here, so a topic rename only ever happens in one place.
"""
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
LOG_INGESTION_TOPIC = "logsage.logs.raw"
CONSUMER_GROUP_ID = "logsage-analysis-service"