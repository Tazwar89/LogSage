"""
LLM-as-judge eval harness for analysis_service's agentic diagnostic pipeline.

Usage:
    # Fully offline (mocked diagnostic LLM + mocked judge -- sanity check /
    # CI wiring only, no real accuracy signal):
    MOCK_LLM=true python -m eval.run_eval

    # Real eval run (needs GROQ_API_KEY and a reachable Qdrant -- QDRANT_URL
    # defaults to http://localhost:6333, point it at the deployed
    # logsage-qdrant instance for a production-representative run):
    GROQ_API_KEY=... QDRANT_URL=https://logsage-qdrant.fly.dev python -m eval.run_eval

    # To measure the false-positive rate too, a baseline must already be
    # built (ingestion_service's /upload/baseline against HDFS_2k.log) in
    # the same Qdrant instance QDRANT_URL points at -- otherwise negative
    # cases are skipped with a warning instead of silently passing.

Writes eval_results.json (per-case scores + aggregate) next to this script
and prints a summary, e.g.:
    Achieved a 92% diagnostic accuracy score using LLM-as-a-Judge evaluation

golden_dataset.json now mixes two kinds of cases, split on
`expected_root_cause`:
  - positive cases (expected_root_cause set) -- run through
    is_anomalous() then, if flagged anomalous, the full diagnostic
    pipeline + LLM judge, same as before.
  - negative cases (expected_root_cause: null) -- known-normal HDFS_2k
    lines. These only run through is_anomalous(); the diagnostic
    pipeline is never called on them. A negative case that gets flagged
    anomalous is a false positive and is never scored by the judge.

This means "accuracy"/"mean_score" below are computed only over positive
cases (unchanged meaning from before), while false_positive_rate is
computed only over negative cases -- report both, not one folded into
the other.
"""
import os, sys, json
from pathlib import Path

# Make services/ and libs/logsage_common importable, same pattern as
# tests/analysis_service/conftest.py.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "services"))
sys.path.insert(0, str(_ROOT / "libs" / "logsage_common"))

from analysis_service.app.agentic_pipeline import run_diagnostic_pipeline
from analysis_service.app.rag import load_knowledge_base, build_kb_index
from analysis_service.app.anomaly import is_anomalous
from .judge import judge_diagnosis

MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"
PASS_THRESHOLD = 0.7
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


class _FakeKbStore:
    """
    Deterministic in-process substitute for QdrantVectorStore, used only
    when MOCK_LLM=true so `python -m eval.run_eval` works with zero external
    dependencies (no live Qdrant, no API key). Does plain substring matching
    against the knowledge base instead of real embeddings.
    """
    def __init__(self, templates: dict):
        self.templates = templates


    def query(self, text, k=1):
        text_lower = text.lower()
        scored = []

        for tid, template in self.templates.items():
            overlap = sum(1 for w in template.lower().split() if w in text_lower)
            scored.append((overlap, tid))

        scored.sort(reverse=True)

        return [{"template_id": tid, "text": self.templates[tid], "distance": 0.0} for _, tid in scored[:k]]


def _build_kb(kb_entries):
    if MOCK_LLM:
        templates = {i: entry["issue"] for i, entry in enumerate(kb_entries)}
        kb_lookup = {i: entry for i, entry in enumerate(kb_entries)}

        return _FakeKbStore(templates), kb_lookup

    from logsage_common.vector_store_qdrant import QdrantVectorStore

    kb_store = QdrantVectorStore(collection_name="eval_knowledge_base")
    kb_lookup = build_kb_index(kb_store, kb_entries)

    return kb_store, kb_lookup


def _get_baseline_store():
    """
    Returns a live QdrantVectorStore pointed at the "baseline" collection
    (the same one ingestion_service's /upload/baseline populates), or None
    if unreachable / not yet built. None means negative cases get skipped
    with a warning rather than silently counted as passing.
    """
    if MOCK_LLM:
        return None

    from logsage_common.vector_store_qdrant import QdrantVectorStore

    try:
        store = QdrantVectorStore(collection_name="baseline")
        store.load()

        return store

    except Exception as exc:
        print(f"WARNING: could not load baseline Qdrant collection ({exc}); "
              f"negative cases will be skipped, not scored.")

        return None


def run_eval() -> dict:
    cases = json.loads(DATASET_PATH.read_text())
    kb_entries = load_knowledge_base(str(_ROOT / "services" / "analysis_service" / "data" / "knowledge_base.json"))
    kb_store, kb_lookup = _build_kb(kb_entries)
    baseline_store = _get_baseline_store()

    positive_results = []
    negative_results = []
    skipped_negatives = 0

    for case in cases:
        is_negative = case["expected_root_cause"] is None

        if is_negative:
            if baseline_store is None:
                skipped_negatives += 1
                continue

            anomalous, _ = is_anomalous(case["raw_log"], baseline_store)
            negative_results.append(
                {
                    "id": case["id"],
                    "raw_log": case["raw_log"],
                    "flagged_anomalous": anomalous,
                    "false_positive": anomalous,
                }
            )

            # Per the harness contract: the diagnostic pipeline never runs
            # on a case correctly identified as normal, and a false
            # positive still isn't judged against a reference diagnosis
            # that doesn't exist -- being flagged at all IS the failure.
            continue

        if baseline_store is not None:
            anomalous, _ = is_anomalous(case["raw_log"], baseline_store)

            if not anomalous:
                # A true anomaly the detector missed: record it as a
                # judge_score of 0.0 (diagnostic pipeline never got a
                # chance to run) rather than silently dropping it.
                positive_results.append(
                    {
                        "id": case["id"],
                        "raw_log": case["raw_log"],
                        "expected_root_cause": case["expected_root_cause"],
                        "actual_root_cause": None,
                        "actual_fix": None,
                        "pipeline_confidence": None,
                        "judge_score": 0.0,
                        "judge_verdict": "fail",
                        "judge_rationale": "missed by is_anomalous() -- diagnostic pipeline never ran",
                        "detector_missed": True,
                    }
                )

                continue

        pipeline_output = run_diagnostic_pipeline(case["raw_log"], kb_store, kb_lookup)
        analysis = pipeline_output["analysis"]
        verdict = judge_diagnosis(case, analysis)

        positive_results.append(
            {
                "id": case["id"],
                "raw_log": case["raw_log"],
                "expected_root_cause": case["expected_root_cause"],
                "actual_root_cause": analysis.get("root_cause"),
                "actual_fix": analysis.get("suggested_fix"),
                "pipeline_confidence": analysis.get("confidence"),
                "judge_score": verdict["score"],
                "judge_verdict": verdict["verdict"],
                "judge_rationale": verdict["rationale"],
                "detector_missed": False,
            }
        )

    scores = [r["judge_score"] for r in positive_results]
    accuracy = sum(1 for s in scores if s >= PASS_THRESHOLD) / len(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    fp_count = sum(1 for r in negative_results if r["false_positive"])
    false_positive_rate = fp_count / len(negative_results) if negative_results else None

    summary = {
        "num_positive_cases": len(positive_results),
        "accuracy": round(accuracy, 4),
        "mean_score": round(mean_score, 4),
        "pass_threshold": PASS_THRESHOLD,
        "num_negative_cases": len(negative_results),
        "false_positive_rate": round(false_positive_rate, 4) if false_positive_rate is not None else None,
        "skipped_negatives_no_baseline": skipped_negatives,
        "positive_cases": positive_results,
        "negative_cases": negative_results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    return summary


def _print_summary(summary: dict):
    print(f"\n{'CASE':<28}{'SCORE':<8}{'VERDICT':<8}RATIONALE")

    for r in summary["positive_cases"]:
        print(f"{r['id']:<28}{r['judge_score']:<8}{r['judge_verdict']:<8}{r['judge_rationale']}")

    print(
        f"\n{summary['num_positive_cases']} positive cases | "
        f"accuracy={summary['accuracy']*100:.1f}% (threshold={summary['pass_threshold']}) | "
        f"mean_score={summary['mean_score']:.2f}"
    )

    if summary["negative_cases"]:
        print(f"\n{'CASE':<35}FLAGGED_ANOMALOUS")

        for r in summary["negative_cases"]:
            print(f"{r['id']:<35}{r['flagged_anomalous']}")

        fp_pct = summary["false_positive_rate"] * 100
        print(
            f"\n{summary['num_negative_cases']} negative cases | "
            f"false_positive_rate={fp_pct:.1f}%"
        )

    elif summary["skipped_negatives_no_baseline"]:
        print(
            f"\n{summary['skipped_negatives_no_baseline']} negative cases SKIPPED "
            f"(no reachable baseline Qdrant collection -- false_positive_rate not computed)"
        )

    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    summary = run_eval()
    _print_summary(summary)