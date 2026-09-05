"""
Tests for app/vector_store.py

The real SentenceTransformer model is mocked out so these tests run fast,
offline, and deterministically -- they exercise FAISS indexing/query/rebuild
logic, not the embedding model itself.
"""
import numpy as np
import pytest

from app import vector_store as vector_store_module
from app.vector_store import VectorStore


class FakeSentenceTransformer:
    """
    Deterministic fake embedder: maps each unique string to a fixed-size
    vector derived from its hash, so identical text always embeds identically
    and different text embeds differently (good enough for distance checks).
    """

    def __init__(self, *args, **kwargs):
        pass


    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []

        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.random(384).astype("float32")

            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)

            vectors.append(vec)

        return np.array(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def mock_embedding_model(monkeypatch):
    monkeypatch.setattr(vector_store_module, "SentenceTransformer", FakeSentenceTransformer)


class TestVectorStore:
    def test_build_index_and_query_self_match(self):
        store = VectorStore()
        templates = {1: "PacketResponder terminating", 2: "OutOfMemoryError in FSNamesystem"}

        store.build_index(templates)
        results = store.query("PacketResponder terminating", k=1)

        assert len(results) == 1
        assert results[0]["template_id"] == 1
        assert results[0]["distance"] < 1e-6  # querying an indexed string should match itself near-exactly


    def test_query_returns_k_results(self):
        store = VectorStore()
        templates = {i: f"template number {i}" for i in range(5)}

        store.build_index(templates)
        results = store.query("template number 0", k=3)

        assert len(results) == 3


    def test_rebuilding_index_does_not_accumulate_stale_vectors(self):
        """
        Regression test for the bug where build_index() appended to the
        existing FAISS index instead of replacing it, causing KeyErrors
        on id_map lookups after a second upload.
        """
        store = VectorStore()

        first_batch = {1: "first batch template"}
        store.build_index(first_batch)
        assert store.index.ntotal == 1

        second_batch = {1: "second batch template A", 2: "second batch template B"}
        store.build_index(second_batch)

        # Index size must reflect only the second batch, not first + second
        assert store.index.ntotal == 2
        # id_map must be fully replaced, not merged
        assert set(store.id_map.keys()) == {0, 1}


    def test_query_result_ids_always_resolve_in_id_map(self):
        """
        Regression test: every FAISS-returned index must have a corresponding
        id_map entry. This would fail before the index-reset fix.
        """
        store = VectorStore()
        store.build_index({1: "alpha"})
        store.build_index({1: "beta", 2: "gamma"})

        results = store.query("beta", k=2)

        for r in results:
            assert "template_id" in r
            assert "text" in r


    def test_query_on_empty_index_returns_empty_list(self):
        store = VectorStore()
        store.build_index({})

        results = store.query("anything", k=1)

        assert results == []