"""
Tests for services/ingestion-service/app/kafka_producer.py

KafkaProducer is mocked throughout -- these verify message construction,
serialization, and error handling, not connectivity to a real broker.
"""
from unittest.mock import MagicMock

from kafka.errors import KafkaError

from services.ingestion_service.app.kafka_producer import publish_log_entry, publish_batch, LOG_INGESTION_TOPIC


class TestPublishLogEntry:
    def test_publishes_correct_payload_shape(self):
        producer = MagicMock()
        future = MagicMock()
        future.get.return_value = None
        producer.send.return_value = future

        result = publish_log_entry(producer, "file.log-0", {"message": "hello"})

        assert result is True
        producer.send.assert_called_once_with(
            LOG_INGESTION_TOPIC,
            value={"trace_id": "file.log-0", "entry": {"message": "hello"}},
        )


    def test_returns_false_on_kafka_error(self):
        producer = MagicMock()
        future = MagicMock()
        future.get.side_effect = KafkaError("broker unreachable")
        producer.send.return_value = future

        result = publish_log_entry(producer, "file.log-0", {"message": "hello"})

        assert result is False


    def test_waits_for_ack_before_returning(self):
        producer = MagicMock()
        future = MagicMock()
        producer.send.return_value = future

        publish_log_entry(producer, "trace-1", {"message": "x"})

        future.get.assert_called_once()


class TestPublishBatch:
    def test_publishes_all_entries_and_returns_success_count(self):
        producer = MagicMock()
        future = MagicMock()
        future.get.return_value = None
        producer.send.return_value = future

        batch = [("a.log-0", {"message": "one"}), ("a.log-1", {"message": "two"})]
        count = publish_batch(producer, batch)

        assert count == 2
        assert producer.send.call_count == 2
        producer.flush.assert_called_once()


    def test_partial_failure_still_reports_correct_success_count(self):
        producer = MagicMock()
        good_future = MagicMock()
        good_future.get.return_value = None
        bad_future = MagicMock()
        bad_future.get.side_effect = KafkaError("failed")

        producer.send.side_effect = [good_future, bad_future]

        batch = [("a.log-0", {"message": "ok"}), ("a.log-1", {"message": "fails"})]
        count = publish_batch(producer, batch)

        assert count == 1


    def test_empty_batch_returns_zero(self):
        producer = MagicMock()
        count = publish_batch(producer, [])

        assert count == 0
        producer.flush.assert_called_once()