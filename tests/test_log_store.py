"""
Tests for app/log_store.py

Uses fakeredis so these tests run offline with no real Redis server.
"""
import pytest
import fakeredis

from app.log_store import LogStore


@pytest.fixture
def store():
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return LogStore(client=fake_client)


class TestLogStore:
    def test_save_and_get_round_trip(self, store):
        entry = {"message": "PacketResponder terminating", "level": "INFO"}
        store.save("file.log-0", entry)

        retrieved = store.get("file.log-0")

        assert retrieved == entry

    def test_get_missing_trace_id_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_list_trace_ids_returns_all_saved_ids(self, store):
        store.save("a.log-0", {"message": "one"})
        store.save("a.log-1", {"message": "two"})
        store.save("b.log-0", {"message": "three"})

        trace_ids = store.list_trace_ids()

        assert trace_ids == ["a.log-0", "a.log-1", "b.log-0"]

    def test_clear_removes_all_entries(self, store):
        store.save("a.log-0", {"message": "one"})
        store.save("a.log-1", {"message": "two"})

        store.clear()

        assert store.list_trace_ids() == []
        assert store.get("a.log-0") is None

    def test_save_overwrites_existing_entry(self, store):
        store.save("a.log-0", {"message": "original"})
        store.save("a.log-0", {"message": "updated"})

        assert store.get("a.log-0") == {"message": "updated"}
        assert store.list_trace_ids() == ["a.log-0"]  # not duplicated in the index