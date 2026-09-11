"""
Single source of truth for Kafka topic names and broker config.

Both ingestion_service (producer) and analysis_service (consumer) import
from here, so a topic rename only ever happens in one place.
"""
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
LOG_INGESTION_TOPIC = "logsage.logs.raw"
CONSUMER_GROUP_ID = "logsage-analysis-service"

KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD")

# Confluent Cloud requires SASL_SSL; local/Compose Kafka has no auth.
# Presence of credentials is what decides which mode to use.
KAFKA_SECURITY_KWARGS = (
    {
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
        "sasl_plain_username": KAFKA_SASL_USERNAME,
        "sasl_plain_password": KAFKA_SASL_PASSWORD,
    }
    if KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD
    else {}
)