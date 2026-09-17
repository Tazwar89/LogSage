"""
Sanity-checks the eval harness's own wiring (dataset loads, pipeline runs,
judge scores, results file is written) under MOCK_LLM=true. This does NOT
validate real diagnostic accuracy -- it only proves the harness won't be
silently broken in CI. Run `python -m eval.run_eval` with a real API key for
an actual accuracy signal.
"""
import os, json

from .run_eval import run_eval, DATASET_PATH, RESULTS_PATH

os.environ["MOCK_LLM"] = "true"


def test_golden_dataset_is_well_formed():
    cases = json.loads(DATASET_PATH.read_text())
    assert len(cases) >= 1

    positives = [c for c in cases if c["expected_root_cause"] is not None]
    negatives = [c for c in cases if c["expected_root_cause"] is None]
    assert positives, "dataset should contain at least one positive (anomalous) case"
    assert negatives, "dataset should contain at least one negative (known-normal) case"

    for case in cases:
        assert "id" in case and case["id"]
        assert "raw_log" in case and case["raw_log"]
        assert "source" in case and case["source"] in ("synthetic", "hdfs_2k")

        if case["expected_root_cause"] is not None:
            assert case.get("expected_fix"), f"positive case {case['id']} missing expected_fix"
        else:
            # Negative cases are deliberately unlabeled -- nothing to diagnose.
            assert case.get("expected_fix") is None


def test_run_eval_produces_scored_summary():
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

    # No live Qdrant baseline in this offline/MOCK_LLM test environment, so
    # negative cases are skipped (not silently scored as passing) and
    # false_positive_rate is left unset rather than computed on nothing.
    assert summary["num_negative_cases"] + summary["skipped_negatives_no_baseline"] == expected_negatives

    if summary["num_negative_cases"] == 0:
        assert summary["false_positive_rate"] is None

    assert RESULTS_PATH.exists()