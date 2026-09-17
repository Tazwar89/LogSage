# LLM-as-Judge Eval Harness

Measures diagnostic accuracy of `analysis_service`'s LangGraph pipeline
(`agentic_pipeline.run_diagnostic_pipeline`) against a small hand-labeled
golden dataset, using a second LLM call as the judge.

## Files

- `golden_dataset.json` — 23 cases (`source: synthetic` or `hdfs_2k`).
  13 positive cases have a human-verified `expected_root_cause`/`expected_fix`
  (5 synthetic, covering fault types not present in the 2k-line HDFS sample
  itself — OOM, disk I/O, under-replication, security breach; 8 pulled
  verbatim from `HDFS_2k.log`'s only real WARN pattern, "Got exception while
  serving"). 10 negative cases (`expected_root_cause: null`) are real,
  verbatim `INFO`-level HDFS_2k lines that should never be flagged
  anomalous or diagnosed.
- `judge.py` — sends the pipeline's output + the reference answer to an LLM
  and asks for a 0–1 score, pass/fail verdict, and one-sentence rationale.
  Only ever called on positive cases.
- `run_eval.py` — for each case: `is_anomalous()` first (if a baseline
  Qdrant collection is reachable). Negative cases stop there — a flag is a
  false positive, full stop. Positive cases only proceed to the diagnostic
  pipeline + judge if flagged anomalous; a positive case the detector
  misses is scored 0.0 without the pipeline ever running. Writes
  `eval_results.json` with `accuracy`/`mean_score` (over positive cases
  only) and `false_positive_rate` (over negative cases only) as separate
  numbers — don't fold one into the other.
- `precision_recall.py` — separate, heavier metric: block-level
  precision/recall of the detector against Loghub's official HDFS ground
  truth. Requires the full HDFS_1 log + `anomaly_label.csv`, which Loghub
  only releases via a Zenodo request (see the script's docstring) — it is
  **not** available for the 2k-line sample used everywhere else in this
  repo, so this is a separate, occasional run, not part of `run_eval.py`.

## Running

Offline sanity check (no API key, no Qdrant, deterministic keyword-overlap
judge — confirms the harness itself is wired correctly, not real accuracy):

```bash
MOCK_LLM=true python -m eval.run_eval
```

Real eval run (needs `GROQ_API_KEY`; point `QDRANT_URL` at either a local
Qdrant or the deployed `logsage-qdrant` instance):

```bash
export GROQ_API_KEY=...
export QDRANT_URL=https://logsage-qdrant.fly.dev
python -m eval.run_eval
```

Real runs use a scratch Qdrant collection (`eval_knowledge_base`) so they
never touch the `knowledge_base` collection used by the live service.
For `false_positive_rate` to be computed (rather than skipped), the
`baseline` collection also needs a real baseline loaded first — run
ingestion_service's `/upload/baseline` against `HDFS_2k.log` against the
same `QDRANT_URL` before running the eval.

`MOCK_LLM` output isn't a real accuracy signal, but it's real and useful
for wiring negatives: 13 positive cases run the mocked judge (all `fail` at
the offline keyword-overlap heuristic — expected, not a regression), and
the 10 negative cases are reported as skipped (no baseline reachable
offline), not silently passed.

## Output

Prints a per-case table for positive cases (score/verdict/rationale) and a
second table for negative cases (flagged or not), and writes full detail to
`eval_results.json`:

```
CASE                        SCORE   VERDICT RATIONALE
oom_fsnamesystem            1.0     pass    Correctly identifies heap exhaustion and the right fix.
...
13 positive cases | accuracy=92.3% (threshold=0.7) | mean_score=0.90

CASE                               FLAGGED_ANOMALOUS
hdfs2k_normal_addstoredblock       False
...
10 negative cases | false_positive_rate=10.0%
```

`accuracy`/`mean_score` are computed over positive cases only.
`false_positive_rate` is computed over negative cases only — quote both,
not one folded into the other. Neither number alone justifies an
anomaly-detector claim on its own: a high `accuracy` with an unreported
`false_positive_rate` says nothing about how often normal traffic gets
needlessly escalated to the LLM, and a low `false_positive_rate` achieved
by flagging almost nothing would also show up as a low `recall` in
`precision_recall.py` (see below) — report them together.

Because Groq's output isn't deterministic, run `run_eval.py` 2–3 times
and report the range of `mean_score` (e.g. "0.85–0.95 across 3 runs"),
not a single point estimate:

```bash
for i in 1 2 3; do
  GROQ_API_KEY=... QDRANT_URL=https://logsage-qdrant.fly.dev python -m eval.run_eval
  cp eval/eval_results.json "eval/eval_results_run${i}.json"
done
```

## Extending

Add more cases to `golden_dataset.json` as real anomalies show up in
production logs — that's more valuable than hand-crafting synthetic ones,
since it keeps the eval set representative of what the pipeline actually
has to handle. Keep tagging each case's `source` (`synthetic` vs
`hdfs_2k`/production) so it stays clear which numbers are backed by real
traffic.