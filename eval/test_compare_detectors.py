import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "logsage_common"))

from eval.compare_detectors import _metrics, evaluate


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = _metrics(y, y)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0 and m["false_positive_rate"] == 0.0


def test_metrics_handles_no_positive_predictions():
    m = _metrics(np.array([1, 1, 0]), np.array([0, 0, 0]))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_evaluate_separates_synthetic_clusters():
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(3, 16)).astype("float32")
    make = lambda n: (rng.normal(size=(n, 3)) @ basis).astype("float32")
    train, holdout = make(300), make(60)
    anomalous = (4 * rng.normal(size=(40, 16))).astype("float32")

    res = evaluate(train, holdout, anomalous)

    assert res["autoencoder"]["recall"] >= 0.9
    assert res["knn"]["recall"] >= 0.9
    assert res["n_train"] == 300 and res["n_anomalous"] == 40