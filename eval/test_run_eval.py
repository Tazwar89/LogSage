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
    for case in cases:
        for key in ("id", "raw_log", "expected_root_cause", "expected_fix"):
            assert key in case and case[key], f"case {case.get('id')} missing {key}"


def test_run_eval_produces_scored_summary():
    summary = run_eval()

    assert summary["num_cases"] == len(json.loads(DATASET_PATH.read_text()))
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["mean_score"] <= 1.0
    for case_result in summary["cases"]:
        assert case_result["judge_verdict"] in ("pass", "fail")
        assert 0.0 <= case_result["judge_score"] <= 1.0

    assert RESULTS_PATH.exists()
