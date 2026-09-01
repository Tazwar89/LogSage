import os, json
from openai import OpenAI
from redact import redact

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT_TEMPLATE = """You are a log diagnostic assistant.

Anomalous log entry:
{log_entry}

Related historical issues/fixes:
{context}

Respond ONLY in JSON with keys: root_cause, suggested_fix, confidence (0-1)."""

def analyze_log(log_entry, context_entries):
    safe_log = redact(log_entry)
    context_str = "\n".join(
        f"- Issue: {c['issue']} | Fix: {c['fix']}" for c in context_entries
    ) or "No related historical issues found."

    prompt = PROMPT_TEMPLATE.format(log_entry=safe_log, context=context_str)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content or "{}")