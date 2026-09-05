"""Tests for app/analytics.py"""
import numpy as np
import pytest

from app.analytics import compute_log_stats, IsolationForestDetector


class TestComputeLogStats:
    def test_empty_entries_returns_zeroed_stats(self):
        stats = compute_log_stats([])
        assert stats == {"total_logs": 0, "level_distribution": {}, "top_templates": []}

    def test_counts_total_logs(self):
        entries = [{"level": "INFO"}, {"level": "ERROR"}, {"level": "INFO"}]
        stats = compute_log_stats(entries)
        assert stats["total_logs"] == 3

    def test_level_distribution_is_correct(self):
        entries = [{"level": "INFO"}, {"level": "ERROR"}, {"level": "INFO"}]
        stats = compute_log_stats(entries)
        assert stats["level_distribution"] == {"INFO": 2, "ERROR": 1}

    def test_top_templates_ranked_by_frequency(self):
        entries = [
            {"template_string": "common pattern"},
            {"template_string": "common pattern"},
            {"template_string": "rare pattern"},
        ]
        stats = compute_log_stats(entries)

        assert stats["top_templates"][0] == {"template": "common pattern", "count": 2}
        assert stats["top_templates"][1] == {"template": "rare pattern", "count": 1}

    def test_missing_level_column_does_not_crash(self):
        entries = [{"template_string": "x"}]
        stats = compute_log_stats(entries)
        assert stats["level_distribution"] == {}


class TestIsolationForestDetector:
    def test_unfitted_detector_fails_open(self):
        detector = IsolationForestDetector()
        point = np.zeros(10, dtype="float32")

        assert detector.predict(point) is False

    def test_fitting_with_insufficient_data_stays_unfitted(self):
        detector = IsolationForestDetector()
        detector.fit(np.zeros((1, 10), dtype="float32"))

        assert detector._fitted is False

    def test_flags_clear_outlier_after_fitting(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, size=(100, 10)).astype("float32")

        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(baseline)

        clear_outlier = np.full(10, 50.0, dtype="float32")
        assert detector.predict(clear_outlier) is True
