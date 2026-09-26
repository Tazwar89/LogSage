"""
Tests for services/consumer_service/app/main.py (run_consumer)

KafkaConsumer and LogStore are mocked -- verifies per-message handling
logic, not connectivity to a real broker or Redis instance.
"""
from unittest.mock import MagicMock, patch

import pytest


class TestKafkaConsumer:
    def test_consumer_stores_each_message_via_log_store(self):
        from services.consumer_service.app import main as consumer_main

        fake_message_1 = MagicMock()
        fake_message_1.value = {"trace_id": "a.log-0", "entry": {"message": "one"}}
        fake_message_2 = MagicMock()
        fake_message_2.value = {"trace_id": "a.log-1", "entry": {"message": "two"}}

        with patch("services.consumer_service.app.main.KafkaConsumer") as mock_consumer_cls, \
             patch("services.consumer_service.app.main.LogStore") as mock_log_store_cls:

            mock_consumer = MagicMock()
            mock_consumer.__iter__.return_value = iter([fake_message_1, fake_message_2])
            mock_consumer_cls.return_value = mock_consumer
            mock_log_store = MagicMock()
            mock_log_store_cls.return_value = mock_log_store

            consumer_main.run_consumer()

            assert mock_log_store.save.call_count == 2
            assert mock_consumer.commit.call_count == 2
            mock_log_store.save.assert_any_call("a.log-0", {"message": "one"})
            mock_log_store.save.assert_any_call("a.log-1", {"message": "two"})


    def test_consumer_does_not_commit_on_redis_write_failure(self):
        from services.consumer_service.app import main as consumer_main

        fake_message = MagicMock()
        fake_message.offset = 5
        fake_message.value = {"trace_id": "a.log-0", "entry": {"message": "one"}}

        with patch("services.consumer_service.app.main.KafkaConsumer") as mock_consumer_cls, \
             patch("services.consumer_service.app.main.LogStore") as mock_log_store_cls:

            mock_consumer = MagicMock()
            mock_consumer.__iter__.return_value = iter([fake_message])
            mock_consumer_cls.return_value = mock_consumer
            mock_log_store = MagicMock()
            mock_log_store.save.side_effect = RuntimeError("redis down")
            mock_log_store_cls.return_value = mock_log_store

            with pytest.raises(RuntimeError):
                consumer_main.run_consumer()

            mock_consumer.commit.assert_not_called()