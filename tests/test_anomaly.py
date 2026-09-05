"""
Tests for app/anomaly.py

Uses a lightweight fake VectorStore so these tests run instantly with
no embedding model or FAISS index required.
"""
from app.anomaly import is_anomalous


class FakeVectorStore:
    """Returns a pre-set list of query results regardless of input text."""

    def __init__(self, results):
        self._results = results

    def query(self, text, k=1):
        return self._results[:k]


class TestIsAnomalous:
    def test_below_threshold_is_not_anomalous(self):
        store = FakeVectorStore([{"template_id": 1, "text": "known pattern", "distance": 0.1}])

        anomalous, nearest = is_anomalous("some log line", store, threshold=0.6)

        assert anomalous is False
        assert nearest is not None, "No nearest object was found"
        assert nearest["distance"] == 0.1


    def test_above_threshold_is_anomalous(self):
        store = FakeVectorStore([{"template_id": 1, "text": "known pattern", "distance": 0.9}])

        anomalous, nearest = is_anomalous("a very different log line", store, threshold=0.6)

        assert anomalous is True
        assert nearest is not None, "No nearest object was found"
        assert nearest["distance"] == 0.9


    def test_exactly_at_threshold_is_not_anomalous(self):
        # Strictly greater-than semantics: distance == threshold should NOT flag as anomalous
        store = FakeVectorStore([{"template_id": 1, "text": "boundary case", "distance": 0.6}])

        anomalous, _ = is_anomalous("boundary log line", store, threshold=0.6)

        assert anomalous is False


    def test_empty_index_is_treated_as_anomalous(self):
        # No results at all (e.g. index not built yet) should be conservatively flagged
        store = FakeVectorStore([])

        anomalous, nearest = is_anomalous("anything", store, threshold=0.6)

        assert anomalous is True
        assert nearest is None


    def test_custom_threshold_is_respected(self):
        store = FakeVectorStore([{"template_id": 1, "text": "pattern", "distance": 0.3}])

        # With a stricter threshold, the same distance now counts as anomalous
        anomalous, _ = is_anomalous("log line", store, threshold=0.2)

        assert anomalous is True