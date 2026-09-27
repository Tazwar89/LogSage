"""
Blinds judge verdicts from sequence_judge_results_seed*.json for manual review.

Usage:
    python -m eval.blind_review --n 18 --seed 42 \
        --sources eval/sequence_judge_results_seed7.json eval/sequence_judge_results_seed13.json \
        --out eval/human_review_blind.json --answer-key eval/human_review_answer_key.json

Samples N cases (mixed pass/fail per the judge, so you're not just reviewing
passes) across the given source files, writes:
  - human_review_blind.json: block_id, n_events, sequence_score, root_cause,
    suggested_fix (NO judge_score/judge_rationale/checks) -- fill in your
    own "human_score" (0.0/0.5/1.0) and "human_rationale" per case.
  - human_review_answer_key.json: same case ids mapped to the judge's actual
    score/verdict/rationale, kept separate so you don't peek while grading.

Score/pass threshold matches eval/run_sequence_eval.py: pass if score >= 0.7.
"""
import argparse
import json
import random
from pathlib import Path

PASS_THRESHOLD = 0.7


def load_cases(paths):
    cases = []

    for p in paths:
        data = json.loads(Path(p).read_text())
        seed = data.get("seed", Path(p).stem)

        for c in data.get("cases", []):
            c = dict(c)
            c["_source_seed"] = seed
            c["_case_key"] = f"seed{seed}:{c['block_id']}"
            cases.append(c)

    return cases


def stratified_sample(cases, n, seed):
    rng = random.Random(seed)
    passed = [c for c in cases if c["judge_score"] >= PASS_THRESHOLD]
    failed = [c for c in cases if c["judge_score"] < PASS_THRESHOLD]
    rng.shuffle(passed)
    rng.shuffle(failed)

    # Aim for a mix; if one bucket is small (e.g. few fails), take what's there
    # and top up from the other bucket rather than erroring.
    n_fail = min(len(failed), max(1, n // 3)) if failed else 0
    n_pass = min(len(passed), n - n_fail)
    n_fail = min(len(failed), n - n_pass)  # top up if pass bucket was short

    sample = passed[:n_pass] + failed[:n_fail]
    rng.shuffle(sample)

    return sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval/human_review_blind.json")
    p.add_argument("--answer-key", default="eval/human_review_answer_key.json")
    args = p.parse_args()

    cases = load_cases(args.sources)
    sample = stratified_sample(cases, args.n, args.seed)

    blind = []
    answer_key = {}

    for c in sample:
        key = c["_case_key"]
        blind.append({
            "case_key": key,
            "n_events": c.get("n_events"),
            "sequence_score": c.get("sequence_score"),
            "kb_matches_used": c.get("kb_matches_used"),
            "root_cause": c.get("root_cause"),
            "suggested_fix": c.get("suggested_fix"),
            "human_score": None,        # fill in: 0.0 / 0.5 / 1.0
            "human_rationale": "",      # fill in: one sentence
        })
        answer_key[key] = {
            "judge_score": c["judge_score"],
            "judge_verdict": "pass" if c["judge_score"] >= PASS_THRESHOLD else "fail",
            "judge_rationale": c.get("judge_rationale"),
        }

    Path(args.out).write_text(json.dumps(blind, indent=2))
    Path(args.answer_key).write_text(json.dumps(answer_key, indent=2))

    print(f"Sampled {len(sample)} cases ({sum(1 for c in sample if c['judge_score']>=PASS_THRESHOLD)} judge-pass, "
          f"{sum(1 for c in sample if c['judge_score']<PASS_THRESHOLD)} judge-fail)")
    print(f"Blind file: {args.out}  <- fill in human_score/human_rationale here")
    print(f"Answer key: {args.answer_key}  <- don't open until you've graded everything")


if __name__ == "__main__":
    main()