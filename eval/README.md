# LLM-as-Judge Eval Harness

Measures diagnostic accuracy of `analysis_service`'s LangGraph pipeline
(`agentic_pipeline.run_diagnostic_pipeline`) against a small hand-labeled
golden dataset, using a second LLM call as the judge.

## Files

- `golden_dataset.json` — anomalous log lines with a human-verified expected
  root cause and fix. Two entries intentionally match `knowledge_base.json`
  (tests RAG retrieval); the other three have no KB match (tests the
  pipeline's ability to reason from the log line alone).
- `judge.py` — sends the pipeline's output + the reference answer to an LLM
  and asks for a 0–1 score, pass/fail verdict, and one-sentence rationale.
- `run_eval.py` — orchestrates: run pipeline on each case → judge each
  result → aggregate → write `eval_results.json`.

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

## Output

Prints a per-case table and an aggregate accuracy, and writes the full
detail to `eval_results.json`:

```
CASE                        SCORE   VERDICT RATIONALE
oom_fsnamesystem            1.0     pass    Correctly identifies heap exhaustion and the right fix.
...
5 cases | accuracy=80.0% (threshold=0.7) | mean_score=0.84
```

`accuracy` = fraction of cases scoring ≥ 0.7. `mean_score` = average judge
score across all cases. Use whichever reads better for a resume bullet,
e.g. "Achieved an 80% diagnostic accuracy score using LLM-as-a-Judge
evaluation."

## Extending

Add more cases to `golden_dataset.json` as real anomalies show up in
production logs — that's more valuable than hand-crafting synthetic ones,
since it keeps the eval set representative of what the pipeline actually
has to handle.
