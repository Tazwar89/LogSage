import types

import pytest

pytest.importorskip("qdrant_client")

from typing import Any, cast

from logsage_common.vector_store_qdrant import QdrantVectorStore


class FakeClient:
    def __init__(self, exists):
        self._exists = exists
        self.calls = []


    def get_collections(self):
        self.calls.append(("exists", None))
        names = ["baseline"] if self._exists else ["other"]
        return types.SimpleNamespace(collections=[types.SimpleNamespace(name=n) for n in names])

 
    def get_collection(self, name):
        raise AssertionError("must not call get_collection (parses optimizer_config)")


    def delete_collection(self, name):
        self.calls.append(("delete", name))


    def create_collection(self, **kw):
        self.calls.append(("create", kw["collection_name"]))


    def upsert(self, **kw):
        self.calls.append(("upsert", len(kw["points"])))


def _store(exists):
    s = object.__new__(QdrantVectorStore)
    fake_client = cast(Any, FakeClient(exists))
    s.collection_name, s.dim, s.client = "baseline", 4, fake_client
    s.embed = lambda texts: __import__("numpy").ones((len(texts), 4), dtype="float32")

    return s


def test_load_succeeds_when_collection_exists():
    _store(True).load()


def test_load_raises_file_not_found_when_missing():
    with pytest.raises(FileNotFoundError):
        _store(False).load()


def test_build_index_recreates_existing_collection():
    s = _store(True)
    s.build_index({1: "a", 2: "b"})
    client = cast(FakeClient, s.client)
    assert [c[0] for c in client.calls] == ["exists", "delete", "create", "upsert"]


def test_build_index_creates_when_absent():
    s = _store(False)
    s.build_index({1: "a"})
    client = cast(FakeClient, s.client)
    assert [c[0] for c in client.calls] == ["exists", "create", "upsert"]


def test_existence_check_does_not_need_collection_exists_attribute():
    # qdrant-client < 1.8 has no collection_exists(); FakeClient doesn't define it either.
    assert not hasattr(FakeClient(True), "collection_exists")
    _store(True).load()