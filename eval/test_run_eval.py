"""
Sanity-checks the eval harness's own wiring (dataset loads, pipeline runs,
judge scores, results file is written) under MOCK_LLM=true (set in conftest.py
before imports). This does NOT validate real diagnostic accuracy -- run
`python -m eval.run_eval` with a real API key for that.
"""
import json

from . import run_eval as run_eval_module
from .run_eval import DATASET_PATH, run_eval


def test_golden_dataset_is_well_formed():
    cases = json.loads(DATASET_PATH.read_text())
    assert len(cases) >= 1

    positives = [c for c in cases if c["expected_root_cause"] is not None]
    negatives = [c for c in cases if c["expected_root_cause"] is None]
    assert positives, "dataset should contain at least one positive (anomalous) case"
    assert negatives, "dataset should contain at least one negative (known-normal) case"

    for case in cases:
        assert case.get("id")
        assert case.get("raw_log")
        assert "source" in case and case["source"] in ("synthetic", "hdfs_2k")

        if case["expected_root_cause"] is not None:
            assert case.get("expected_fix"), f"positive case {case['id']} missing expected_fix"

        else:
            assert case.get("expected_fix") is None


def test_run_eval_produces_scored_summary(tmp_path, monkeypatch):
    # Guard against a silently-real run, and never overwrite the committed eval_results.json.
    assert run_eval_module.MOCK_LLM is True
    results_path = tmp_path / "eval_results.json"
    monkeypatch.setattr(run_eval_module, "RESULTS_PATH", results_path)

    summary = run_eval()

    cases = json.loads(DATASET_PATH.read_text())
    expected_positives = sum(1 for c in cases if c["expected_root_cause"] is not None)
    expected_negatives = sum(1 for c in cases if c["expected_root_cause"] is None)

    assert summary["num_positive_cases"] == expected_positives
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["mean_score"] <= 1.0

    for case_result in summary["positive_cases"]:
        assert case_result["judge_verdict"] in ("pass", "fail")
        assert 0.0 <= case_result["judge_score"] <= 1.0

    # No live Qdrant baseline offline, so negatives are skipped, not silently passed.
    assert summary["num_negative_cases"] + summary["skipped_negatives_no_baseline"] == expected_negatives

    if summary["num_negative_cases"] == 0:
        assert summary["false_positive_rate"] is None

    assert results_path.exists()