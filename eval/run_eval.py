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

Writes eval_results.json (per-case scores + aggregate) next to this script
and prints a summary, e.g.:
    Achieved a 92% diagnostic accuracy score using LLM-as-a-Judge evaluation
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


def run_eval() -> dict:
    cases = json.loads(DATASET_PATH.read_text())
    kb_entries = load_knowledge_base(str(_ROOT / "services" / "analysis_service" / "data" / "knowledge_base.json"))
    kb_store, kb_lookup = _build_kb(kb_entries)

    results = []

    for case in cases:
        pipeline_output = run_diagnostic_pipeline(case["raw_log"], kb_store, kb_lookup)
        analysis = pipeline_output["analysis"]
        verdict = judge_diagnosis(case, analysis)

        results.append(
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
            }
        )

    scores = [r["judge_score"] for r in results]
    accuracy = sum(1 for s in scores if s >= PASS_THRESHOLD) / len(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "num_cases": len(results),
        "accuracy": round(accuracy, 4),
        "mean_score": round(mean_score, 4),
        "pass_threshold": PASS_THRESHOLD,
        "cases": results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    return summary


def _print_summary(summary: dict):
    print(f"\n{'CASE':<28}{'SCORE':<8}{'VERDICT':<8}RATIONALE")

    for r in summary["cases"]:
        print(f"{r['id']:<28}{r['judge_score']:<8}{r['judge_verdict']:<8}{r['judge_rationale']}")

    print(
        f"\n{summary['num_cases']} cases | "
        f"accuracy={summary['accuracy']*100:.1f}% (threshold={summary['pass_threshold']}) | "
        f"mean_score={summary['mean_score']:.2f}"
    )
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    summary = run_eval()
    _print_summary(summary)