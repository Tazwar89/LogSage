"""
LLM-as-judge scorer for the diagnostic pipeline's eval harness.

Given a golden test case (raw log + expected root cause/fix) and the
pipeline's actual output, asks an LLM to grade the diagnosis on a 0-1 scale
and return a pass/fail verdict plus rationale. Kept separate from
agentic_pipeline.py's _call_llm_json so the judge prompt/parsing can evolve
independently of the diagnostic pipeline itself.

Respects the same MOCK_LLM / GROQ_API_KEY / LLM_BASE_URL conventions as
agentic_pipeline.py so `MOCK_LLM=true` runs the whole harness offline.
"""
import json
import os

from openai import OpenAI

JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

JUDGE_PROMPT_TEMPLATE = """You are grading an automated log-diagnostic system.

Original log line:
{raw_log}

Reference (human-verified) diagnosis:
- Root cause: {expected_root_cause}
- Suggested fix: {expected_fix}

System's actual diagnosis:
- Root cause: {actual_root_cause}
- Suggested fix: {actual_fix}
- Confidence reported by system: {actual_confidence}

Grade the system's diagnosis against the reference. It does not need to
match word-for-word -- judge whether it identifies the same underlying
technical issue and proposes a fix that would plausibly resolve it.

Respond ONLY in JSON with keys:
- "score": float between 0.0 and 1.0 (1.0 = fully correct root cause and
  reasonable fix, 0.5 = partially correct or vague, 0.0 = wrong or
  nonsensical)
- "verdict": "pass" if score >= 0.7 else "fail"
- "rationale": one sentence explaining the score
"""


def _get_client():
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )


def judge_diagnosis(case: dict, actual: dict) -> dict:
    """
    case: one entry from golden_dataset.json
    actual: the "analysis" dict returned by run_diagnostic_pipeline
            (keys: root_cause, suggested_fix, confidence)
    Returns: {"score": float, "verdict": "pass"|"fail", "rationale": str}
    """
    if MOCK_LLM:
        # Deterministic offline stand-in: exact-ish match on root cause
        # keywords gives a pass, otherwise a mid-range score. Lets the
        # harness be exercised in CI without spending API credits.
        expected = case["expected_root_cause"].lower()
        got = str(actual.get("root_cause", "")).lower()
        overlap = any(word in got for word in expected.split() if len(word) > 4)
        score = 0.9 if overlap else 0.4

        return {
            "score": score,
            "verdict": "pass" if score >= 0.7 else "fail",
            "rationale": "mock judge: keyword overlap heuristic (MOCK_LLM=true)",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        raw_log=case["raw_log"],
        expected_root_cause=case["expected_root_cause"],
        expected_fix=case["expected_fix"],
        actual_root_cause=actual.get("root_cause", ""),
        actual_fix=actual.get("suggested_fix", ""),
        actual_confidence=actual.get("confidence", ""),
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content or "{}")

    # Defensive defaults in case the judge model omits a key
    result.setdefault("score", 0.0)
    result.setdefault("verdict", "fail")
    result.setdefault("rationale", "")

    return result