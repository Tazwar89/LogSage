import numpy as np
from ingestion_service.app.anomaly import BaselineAnomalyDetector, filter_anomalous_logs


def _make_clustered_baseline(n=20, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.05, size=(n, dim)).astype("float32")


def test_fit_sets_threshold_from_baseline_spread():
    baseline = _make_clustered_baseline()
    detector = BaselineAnomalyDetector(percentile=95.0).fit(baseline)

    assert detector.threshold >= 0.0
    assert detector._fitted_on == len(baseline)


def test_near_baseline_point_is_not_anomalous():
    baseline = _make_clustered_baseline()
    detector = BaselineAnomalyDetector().fit(baseline)

    near_point = (baseline[0] + baseline[1]) / 2  # interpolated, stays within the cluster
    assert not detector.is_anomalous(near_point.reshape(1, -1))[0]


def test_far_point_is_anomalous():
    baseline = _make_clustered_baseline()
    detector = BaselineAnomalyDetector().fit(baseline)

    far_point = np.full((1, baseline.shape[1]), 50.0, dtype="float32")
    assert detector.is_anomalous(far_point)[0]


def test_empty_baseline_flags_everything():
    detector = BaselineAnomalyDetector().fit(np.empty((0, 8), dtype="float32"))

    some_point = np.zeros((1, 8), dtype="float32")
    assert detector.is_anomalous(some_point)[0]


def test_single_point_baseline_flags_everything():
    # k-NN needs >=2 points to establish a spread; a single-template
    # baseline should degrade to "treat everything as anomalous" rather
    # than raise.
    detector = BaselineAnomalyDetector().fit(np.zeros((1, 8), dtype="float32"))

    assert detector.threshold == 0.0
    assert detector.is_anomalous(np.ones((1, 8), dtype="float32"))[0]


class _FakeVectorStore:
    """Minimal stand-in so filter_anomalous_logs can be tested without Qdrant."""

    dim = 8

    def embed(self, texts):
        # Deterministic per-text vector so 'baseline-like' vs 'far' text is stable across runs.
        return np.array(
            [np.full(self.dim, 0.01 if "normal" in t else 50.0, dtype="float32") for t in texts]
        )


def test_filter_anomalous_logs_splits_correctly():
    baseline = _make_clustered_baseline()
    detector = BaselineAnomalyDetector().fit(baseline)
    store = _FakeVectorStore()

    logs = [
        {"template_string": "normal startup sequence"},
        {"template_string": "CRITICAL disk failure at sector 0"},
    ]

    anomalous, scored_all = filter_anomalous_logs(logs, store, detector)

    assert len(scored_all) == 2
    assert all("anomaly_score" in log for log in scored_all)
    assert len(anomalous) == 1
    assert "disk failure" in anomalous[0]["template_string"]