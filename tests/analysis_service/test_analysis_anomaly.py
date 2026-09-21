"""
Tests for app/anomaly.py.

The old tests targeted a previous `threshold=` signature and a store that
only exposed query(); is_anomalous now fits a detector on the baseline
vectors, so these tests patch fetch_baseline_vectors and use a fake store
with embed()/query().
"""
import numpy as np
import pytest
from analysis_service.app import anomaly


class FakeStore:
    def __init__(self, query_vec, nearest=None):
        self._vec = np.asarray([query_vec], dtype="float32")
        self._nearest = nearest


    def embed(self, texts):
        return self._vec


    def query(self, text, k=1):
        return [self._nearest] if self._nearest else []


def _baseline():
    rng = np.random.default_rng(0)

    return (rng.normal(size=(100, 8)) * 0.05).astype("float32")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    anomaly._autoencoder_cache.clear()
    monkeypatch.setattr(anomaly, "fetch_baseline_vectors", lambda store: _baseline())
    monkeypatch.delenv("ANOMALY_DETECTOR", raising=False)
    monkeypatch.delenv("AUTOENCODER_DIR", raising=False)


def test_in_distribution_point_is_not_anomalous():
    store = FakeStore(_baseline()[0], nearest={"template_id": 1, "text": "known", "distance": 0.0})
    anomalous, nearest = anomaly.is_anomalous("line", store)
    assert anomalous is False
    assert nearest is not None
    assert nearest["template_id"] == 1


def test_far_point_is_anomalous():
    store = FakeStore(np.full(8, 5.0), nearest={"template_id": 1, "text": "known", "distance": 9.0})
    anomalous, _ = anomaly.is_anomalous("line", store)
    assert anomalous is True


def test_empty_query_results_returns_none_nearest():
    store = FakeStore(np.full(8, 5.0), nearest=None)
    _, nearest = anomaly.is_anomalous("line", store)
    assert nearest is None


def test_empty_baseline_is_conservatively_anomalous(monkeypatch):
    monkeypatch.setattr(anomaly, "fetch_baseline_vectors", lambda s: np.empty((0, 8), dtype="float32"))
    anomalous, _ = anomaly.is_anomalous("line", FakeStore(np.zeros(8)))
    assert anomalous is True


def test_build_detector_defaults_to_knn():
    from logsage_common.anomaly import BaselineAnomalyDetector

    assert isinstance(anomaly.build_detector(_baseline()), BaselineAnomalyDetector)


def test_autoencoder_selected_by_env_and_cached(monkeypatch):
    pytest.importorskip("tensorflow")
    monkeypatch.setenv("ANOMALY_DETECTOR", "autoencoder")
    base = _baseline()

    first = anomaly.build_detector(base)
    second = anomaly.build_detector(base)

    assert type(first).__name__ == "AutoencoderAnomalyDetector"
    assert first is second


def test_autoencoder_falls_back_to_knn_with_tiny_baseline(monkeypatch):
    from logsage_common.anomaly import BaselineAnomalyDetector

    monkeypatch.setenv("ANOMALY_DETECTOR", "autoencoder")
    det = anomaly.build_detector(_baseline()[:1])
    assert isinstance(det, BaselineAnomalyDetector)