"""
Tests for services/consumer-service/app/main.py (run_consumer)

KafkaConsumer and LogStore are mocked -- verifies per-message handling
logic, not connectivity to a real broker or Redis instance.
"""
from unittest.mock import MagicMock, patch


class TestKafkaConsumer:
    def test_consumer_stores_each_message_via_log_store(self):
        from services.consumer_service.app import main as consumer_main

        fake_message_1 = MagicMock()
        fake_message_1.value = {"trace_id": "a.log-0", "entry": {"message": "one"}}
        fake_message_2 = MagicMock()
        fake_message_2.value = {"trace_id": "a.log-1", "entry": {"message": "two"}}

        with patch("services.consumer-service.app.main.KafkaConsumer") as mock_consumer_cls, \
             patch("services.consumer-service.app.main.LogStore") as mock_log_store_cls:

            mock_consumer_cls.return_value = iter([fake_message_1, fake_message_2])
            mock_log_store = MagicMock()
            mock_log_store_cls.return_value = mock_log_store

            consumer_main.run_consumer()

            assert mock_log_store.save.call_count == 2
            mock_log_store.save.assert_any_call("a.log-0", {"message": "one"})
            mock_log_store.save.assert_any_call("a.log-1", {"message": "two"})