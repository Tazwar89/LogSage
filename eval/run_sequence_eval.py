"""
End-to-end eval of the SEQUENCE path: block context -> DeepLog flag -> agentic
diagnosis -> LLM judge. Complements eval/run_eval.py (which only covers the
legacy per-line path and a small hand-written golden set).

Why reference-free: Loghub's anomaly_label.csv only says Normal/Anomaly per block,
there are no ground-truth root causes. So the judge grades each diagnosis against
the block's own log lines (the only evidence), not against a self-written answer.
Report the number as "judge-rated grounded diagnosis rate", not as accuracy.

Bias controls:
  - JUDGE_MODEL must differ from LLM_MODEL (refuses to run otherwise).
  - Blocks are sampled with a fixed seed, so runs are repeatable.
  - Cases are real HDFS_v1 blocks, not synthetic or duplicated lines.
  - Deterministic checks run alongside the judge (ungrounded paths, destructive
    commands, guardrail flags).

Needs the FULL HDFS_v1 HDFS.log (the 2k sample has ~1 line per block, so no sequences),
a reachable Qdrant (QDRANT_URL), GROQ_API_KEY and HF_TOKEN.

Usage (repo root):
    python -m eval.run_sequence_eval --log ~/Desktop/HDFS_v1/HDFS.log \
        --labels eval/anomaly_label.csv --n-anomalous 60 --n-normal 60

    # detection + case selection only, no LLM calls:
    python -m eval.run_sequence_eval --log ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "services"))
sys.path.insert(0, str(_ROOT / "libs" / "logsage_common"))
sys.path.insert(0, str(_ROOT))

from ingestion_service.app.parsing import parse_line
from ingestion_service.app.sequence import (
    SequenceScorer,
    build_block_entry,
)
from logsage_common.sequence_parsing import BLOCK_ID_RE

from eval.precision_recall import load_ground_truth

RESULTS_DIR = Path(__file__).parent
DEFAULT_JUDGE = "openai/gpt-oss-120b"
PASS_THRESHOLD = 0.7

JUDGE_PROMPT = """You are auditing an automated HDFS log diagnostic system.

The ONLY evidence is this block's event log (a session flagged as anomalous):
{context}

System diagnosis:
- Root cause: {root_cause}
- Suggested fix: {fix}

Grade using the log lines only. Score 0.0-1.0:
- 1.0: root cause names the abnormal event(s) actually present in the lines (e.g. an
  exception, timeout, missing completion) and the fix is a plausible, non-destructive
  investigative or remedial step.
- 0.5: partly right, vague, or generic.
- 0.0: contradicts the lines, blames something not in the evidence, invents specifics
  (paths, hosts, IDs not in the lines), or proposes deleting data.

Respond ONLY in JSON: {{"score": <float>, "verdict": "pass"|"fail", "rationale": "<one sentence>"}}"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)

    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

    return (round((c - m) / d, 4), round((c + m) / d, 4))


def sample_blocks(truth: dict, n_anom: int, n_norm: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    anom = sorted(b for b, a in truth.items() if a)
    norm = sorted(b for b, a in truth.items() if not a)

    return set(rng.sample(anom, min(n_anom, len(anom)))) | set(rng.sample(norm, min(n_norm, len(norm))))


def collect_lines(log_path: str, selected: set[str]) -> list[dict]:
    """One pass over the full log, keeping every parsed line that mentions a selected block, in order."""
    kept = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not any(b in raw for b in ("blk_",)):
                continue

            if selected.isdisjoint(BLOCK_ID_RE.findall(raw)):
                continue

            parsed = parse_line(raw)

            if parsed is not None:
                kept.append(parsed)

    return kept


def deterministic_checks(context: str, analysis: dict) -> dict:
    from analysis_service.app.agentic_pipeline import _DESTRUCTIVE_RE, _PATH_RE

    fix = str(analysis.get("suggested_fix", ""))
    ungrounded = [p for p in _PATH_RE.findall(fix) if p.rstrip(".,;:)") not in context]

    return {
        "ungrounded_paths": ungrounded,
        "destructive_command": bool(_DESTRUCTIVE_RE.search(fix)),
        "guardrail_flags": analysis.get("guardrail_flags", []),
    }


def judge(context: str, analysis: dict, model: str) -> dict:
    import time

    from openai import BadRequestError, OpenAI, RateLimitError

    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        max_retries=0,
    )
    prompt = JUDGE_PROMPT.format(
        context=context, 
        root_cause=analysis.get("root_cause", ""), 
        fix=analysis.get("suggested_fix", "")
    )

    # Track whether to use the extra reasoning body param
    use_reasoning_effort = True

    for attempt in range(6):
        try:
            try:
                if use_reasoning_effort:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        max_tokens=1200,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}],
                        extra_body={"reasoning_effort": "none"} # Explicitly pass the parameter
                    )

                else:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        max_tokens=1200,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}]
                    )

            except BadRequestError:
                use_reasoning_effort = False  # Switch mode flag; retry without reasoning_effort
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}]
                )

            break  # Break the retry loop on a successful request

        except RateLimitError:
            time.sleep(15 * (attempt + 1))

    else:
        raise RuntimeError("Judge rate-limited after 6 retries")

    out = json.loads(resp.choices[0].message.content or "{}")
    out["score"] = float(out.get("score", 0.0))
    out.setdefault("rationale", "")
    time.sleep(3)  # pace calls under the per-minute output-token cap

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Full HDFS_v1 HDFS.log")
    p.add_argument("--labels", default=str(_ROOT / "eval" / "anomaly_label.csv"))
    p.add_argument("--n-anomalous", type=int, default=60)
    p.add_argument("--n-normal", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", default=str(_ROOT / "models" / "sequence"))
    p.add_argument("--dry-run", action="store_true", help="Skip all LLM calls")
    args = p.parse_args()

    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE)
    gen_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    if not args.dry_run:
        from openai import OpenAI
        available = {m.id for m in OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        ).models.list().data}

        for name in (judge_model, gen_model):
            if name not in available:
                raise SystemExit(f"Model '{name}' is not available to this API key.")

    if judge_model == gen_model and not args.dry_run:
        raise SystemExit(f"JUDGE_MODEL must differ from LLM_MODEL (both '{gen_model}').")

    truth = load_ground_truth(Path(args.labels))
    selected = sample_blocks(truth, args.n_anomalous, args.n_normal, args.seed)
    parsed_logs = collect_lines(args.log, selected)
    print(f"Sampled {len(selected)} blocks, collected {len(parsed_logs):,} lines")

    scorer = SequenceScorer(args.model_dir)
    results = {r.block_id: r for r in scorer.score_logs(parsed_logs) if r.block_id in selected}
    missing = selected - set(results)

    tp = [b for b in results if truth[b] and results[b].is_anomalous]
    fn = [b for b in results if truth[b] and not results[b].is_anomalous]
    fp = [b for b in results if not truth[b] and results[b].is_anomalous]
    n_norm = sum(1 for b in results if not truth[b])
    summary: dict = {
        "seed": args.seed, "judge_model": judge_model, "generator_model": gen_model,
        "blocks_sampled": len(selected), "blocks_not_found_in_log": len(missing),
        "detector_on_sample": {"tp": len(tp), "fn": len(fn), "fp": len(fp), "normal_scored": n_norm},
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))

        return

    from analysis_service.app.agentic_pipeline import run_diagnostic_pipeline
    from analysis_service.app.rag import build_kb_index, load_knowledge_base
    from logsage_common.vector_store_qdrant import QdrantVectorStore

    kb_entries = load_knowledge_base(str(_ROOT / "services" / "analysis_service" / "data" / "knowledge_base.json"))
    kb_store = QdrantVectorStore(collection_name="eval_knowledge_base")
    kb_lookup = build_kb_index(kb_store, kb_entries)

    cases, errors = [], []

    from openai import OpenAIError
    import time as _time

    for b in tp:
        try:
            _time.sleep(2)  # stay under Groq free-tier RPM across triage+report+judge calls
            entry = build_block_entry(parsed_logs, results[b])
            out = run_diagnostic_pipeline(entry["message"], kb_store, kb_lookup)
            analysis = out["analysis"]
            verdict = judge(entry["message"], analysis, judge_model)

        except (OpenAIError, json.JSONDecodeError) as exc:
            errors.append({"block_id": b, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"{b}  ERROR {type(exc).__name__}")

            continue

        cases.append({
            "block_id": b, "n_events": entry["n_events"], "sequence_score": entry["sequence_anomaly_score"],
            "kb_matches_used": analysis.get("kb_matches_used", len(out["retrieved_context"])),
            "root_cause": analysis.get("root_cause"), "suggested_fix": analysis.get("suggested_fix"),
            "judge_score": verdict["score"], "judge_rationale": verdict["rationale"],
            "checks": deterministic_checks(entry["message"], analysis),
        })
        print(f"{b}  score={verdict['score']:.2f}")

    n = len(cases)
    passed = sum(1 for c in cases if c["judge_score"] >= PASS_THRESHOLD)
    n_anom = len(tp) + len(fn)
    summary.update({
        "diagnosed_cases": n,
        "errored_cases_count": len(errors),
        "error_rate_of_tp": round(len(errors) / len(tp), 4) if tp else None,
        "judge_pass_rate": round(passed / n, 4) if n else None,
        "judge_pass_rate_95ci": wilson(passed, n),
        "mean_judge_score": round(sum(c["judge_score"] for c in cases) / n, 4) if n else None,
        "end_to_end_rate_incl_missed": round(passed / n_anom, 4) if n_anom else None,
        "ungrounded_path_rate": round(sum(1 for c in cases if c["checks"]["ungrounded_paths"]) / n, 4) if n else None,
        "destructive_command_rate": round(sum(1 for c in cases if c["checks"]["destructive_command"]) / n, 4) if n else None,
        "cases_with_kb_match": sum(1 for c in cases if c["kb_matches_used"] > 0),
        "cases": cases,
        "errored_cases": errors,
    })
    (RESULTS_DIR / f"sequence_judge_results_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()