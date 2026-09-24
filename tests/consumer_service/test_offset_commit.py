"""
Regression test for manual offset commit: a Redis write failure must not
advance the Kafka offset, so the message is redelivered on restart.
"""
from unittest.mock import MagicMock, patch

import pytest

from services.consumer_service.app.main import run_consumer


def _make_message(trace_id="trace-1", entry="log line", offset=0):
    msg = MagicMock()
    msg.value = {"trace_id": trace_id, "entry": entry}
    msg.offset = offset

    return msg


@patch("services.consumer_service.app.main.LogStore")
@patch("services.consumer_service.app.main.connect_with_retry")
def test_commit_called_after_successful_save(mock_connect, mock_log_store_cls):
    message = _make_message()
    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = iter([message])
    mock_connect.return_value = mock_consumer

    mock_log_store = MagicMock()
    mock_log_store_cls.return_value = mock_log_store

    run_consumer()

    mock_log_store.save.assert_called_once_with("trace-1", "log line")
    mock_consumer.commit.assert_called_once()


@patch("services.consumer_service.app.main.LogStore")
@patch("services.consumer_service.app.main.connect_with_retry")
def test_commit_not_called_when_redis_save_fails(mock_connect, mock_log_store_cls):
    message = _make_message()
    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = iter([message])
    mock_connect.return_value = mock_consumer

    mock_log_store = MagicMock()
    mock_log_store.save.side_effect = ConnectionError("redis unavailable")
    mock_log_store_cls.return_value = mock_log_store

    with pytest.raises(ConnectionError):
        run_consumer()

    mock_consumer.commit.assert_not_called()


@patch("services.consumer_service.app.main.LogStore")
@patch("services.consumer_service.app.main.connect_with_retry")
def test_commit_called_after_skipping_malformed_message(mock_connect, mock_log_store_cls):
    bad_message = MagicMock()
    bad_message.value = {"__malformed__": True, "raw": b"not json"}
    bad_message.offset = 0

    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = iter([bad_message])
    mock_connect.return_value = mock_consumer

    mock_log_store = MagicMock()
    mock_log_store_cls.return_value = mock_log_store

    run_consumer()

    mock_log_store.save.assert_not_called()
    mock_consumer.commit.assert_called_once()