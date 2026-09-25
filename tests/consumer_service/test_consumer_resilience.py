"""
Resilience tests for services/consumer_service/app/main.py

Covers the failure paths: malformed Kafka payloads, Redis write failures,
and Kafka connection retry/backoff. KafkaConsumer, LogStore and time.sleep
are mocked -- no broker, Redis, or real waiting.
"""
from unittest.mock import MagicMock, call, patch

import pytest
from kafka.errors import KafkaError

MODULE = "services.consumer_service.app.main"


def _message(value, offset=0):
    msg = MagicMock()
    msg.value = value
    msg.offset = offset
    return msg


class TestSafeJsonDeserializer:
    def test_valid_json_is_parsed(self):
        from services.consumer_service.app import main as consumer_main

        result = consumer_main._safe_json_deserializer(b'{"trace_id": "a-0", "entry": {}}')
        assert result == {"trace_id": "a-0", "entry": {}}


    def test_invalid_json_is_flagged_malformed(self):
        from services.consumer_service.app import main as consumer_main

        result = consumer_main._safe_json_deserializer(b"{not json")
        assert result is not None
        assert result["__malformed__"] is True
        assert result["raw"] == b"{not json"


    def test_invalid_utf8_is_flagged_malformed(self):
        from services.consumer_service.app import main as consumer_main

        result = consumer_main._safe_json_deserializer(b"\xff\xfe\x00")
        assert result is not None
        assert result["__malformed__"] is True


    def test_none_passes_through(self):
        from services.consumer_service.app import main as consumer_main

        assert consumer_main._safe_json_deserializer(None) is None


class TestMalformedMessages:
    @pytest.mark.parametrize(
        "bad_value",
        [
            {"__malformed__": True, "raw": b"garbage"},
            ["not", "a", "dict"],
            None,
            {"entry": {"message": "no trace id"}},
            {"trace_id": "a.log-9"},  # missing "entry" would previously raise KeyError
        ],
    )

    def test_malformed_message_is_skipped_and_consumer_continues(self, bad_value):
        from services.consumer_service.app import main as consumer_main

        good = _message({"trace_id": "a.log-1", "entry": {"message": "ok"}}, offset=2)

        with patch(f"{MODULE}.KafkaConsumer") as mock_consumer_cls, \
             patch(f"{MODULE}.LogStore") as mock_log_store_cls:

            mock_consumer = MagicMock()
            mock_consumer.__iter__.return_value = iter([_message(bad_value, offset=1), good])
            mock_consumer_cls.return_value = mock_consumer
            mock_log_store = MagicMock()
            mock_log_store_cls.return_value = mock_log_store

            consumer_main.run_consumer()

            mock_log_store.save.assert_called_once_with("a.log-1", {"message": "ok"})


class TestRedisFailure:
    def test_redis_write_failure_propagates_and_stops_processing(self):
        """Fail-fast: the worker crashes so the platform restarts it, rather
        than logging and silently dropping entries."""
        from services.consumer_service.app import main as consumer_main

        messages = [
            _message({"trace_id": "a.log-0", "entry": {"message": "one"}}, offset=0),
            _message({"trace_id": "a.log-1", "entry": {"message": "two"}}, offset=1),
        ]

        with patch(f"{MODULE}.KafkaConsumer") as mock_consumer_cls, \
             patch(f"{MODULE}.LogStore") as mock_log_store_cls:

            mock_consumer_cls.return_value = iter(messages)
            mock_log_store = MagicMock()
            mock_log_store.save.side_effect = ConnectionError("redis down")
            mock_log_store_cls.return_value = mock_log_store

            with pytest.raises(ConnectionError, match="redis down"):
                consumer_main.run_consumer()

            assert mock_log_store.save.call_count == 1


class TestConnectWithRetry:
    def test_retries_with_exponential_backoff_then_succeeds(self):
        from services.consumer_service.app import main as consumer_main

        fake_consumer = MagicMock()

        with patch(f"{MODULE}.KafkaConsumer") as mock_consumer_cls, \
             patch(f"{MODULE}.time.sleep") as mock_sleep:

            mock_consumer_cls.side_effect = [
                KafkaError("no brokers"),
                KafkaError("no brokers"),
                fake_consumer,
            ]

            result = consumer_main.connect_with_retry(max_retries=5, initial_delay=2.0)

            assert result is fake_consumer
            assert mock_consumer_cls.call_count == 3
            assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


    def test_backoff_delay_is_capped_at_30_seconds(self):
        from services.consumer_service.app import main as consumer_main

        with patch(f"{MODULE}.KafkaConsumer") as mock_consumer_cls, \
             patch(f"{MODULE}.time.sleep") as mock_sleep:

            mock_consumer_cls.side_effect = KafkaError("no brokers")

            with pytest.raises(RuntimeError):
                consumer_main.connect_with_retry(max_retries=7, initial_delay=2.0)

            delays = [c.args[0] for c in mock_sleep.call_args_list]
            assert delays == [2.0, 4.0, 8.0, 16.0, 30, 30, 30]


    def test_raises_runtime_error_after_max_retries(self):
        from services.consumer_service.app import main as consumer_main

        with patch(f"{MODULE}.KafkaConsumer") as mock_consumer_cls, \
             patch(f"{MODULE}.time.sleep"):

            mock_consumer_cls.side_effect = KafkaError("no brokers")

            with pytest.raises(RuntimeError, match="after 3 attempts") as exc_info:
                consumer_main.connect_with_retry(max_retries=3, initial_delay=1.0)

            assert mock_consumer_cls.call_count == 3
            assert isinstance(exc_info.value.__cause__, KafkaError)