"""
Agentic diagnostic pipeline using LangGraph.

3-node graph, each node with a distinct responsibility:

  1. triage_node   -- redacts sensitive data and extracts key entities from the
                       anomalous log entry (or anomalous block's event context).
  2. research_node -- retrieves relevant historical fixes via RAG
                       (wraps rag.retrieve_context(), which now drops matches
                       beyond a distance cutoff instead of always returning k).
  3. report_node   -- synthesizes triage + research into a final
                       root_cause / suggested_fix / confidence verdict, then
                       post-validates the suggested fix (no paths that were not
                       in the input, no destructive commands).

State is passed between nodes via a typed dict, matching LangGraph's
standard state-graph pattern.
"""
import os, json, re
from functools import partial
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI

from logsage_common.redact import redact

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

# Absolute filesystem / HDFS paths, e.g. /user/root/rand/part-00345
_PATH_RE = re.compile(r"(?<![\w:/.\-])/[\w.\-<>*$]+(?:/[\w.\-<>*$]+)*")
# Commands that delete or reformat data; a log diagnosis should never propose them.
_DESTRUCTIVE_RE = re.compile(
    r"(hdfs\s+dfs\s+-rm[^\n;|`]*|hadoop\s+fs\s+-rm[^\n;|`]*|\brm\s+-[^\n;|`]*|"
    r"hdfs\s+fsck[^\n;|`]*-delete[^\n;|`]*|namenode\s+-format[^\n;|`]*)",
    re.IGNORECASE,
)


class DiagnosticState(TypedDict):
    raw_log: str
    redacted_log: str
    entities: Dict[str, Any]
    retrieved_context: List[Dict[str, str]]
    final_analysis: Dict[str, Any]


def _get_client():
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )


def _call_llm_json(prompt: str, mock_response: dict) -> dict:
    """
    Wraps every LLM call in this pipeline. When MOCK_LLM=true (set in CI to
    avoid spending real API credits on every push, or locally when testing
    without a key), returns a fixed canned response instead of calling out
    to Groq/OpenAI.
    """
    if MOCK_LLM:
        return mock_response

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content or "{}")


def sanitize_suggested_fix(fix: str, allowed_text: str) -> tuple[str, list[str]]:
    """
    Deterministic guard on the LLM's suggested_fix. Returns (clean_fix, flags).
      - any absolute path not present verbatim in allowed_text -> replaced by <path>
      - destructive commands (rm/delete/format) -> replaced by a placeholder
    """
    flags: list[str] = []

    if not isinstance(fix, str):
        return fix, flags

    def _replace_destructive(_m):
        if "destructive_command_removed" not in flags:
            flags.append("destructive_command_removed")

        return "[destructive command removed]"

    fix = _DESTRUCTIVE_RE.sub(_replace_destructive, fix)

    def _replace_path(m):
        path = m.group(0).rstrip(".,;:)")
        if path in allowed_text:
            return m.group(0)

        if "unsupported_path_removed" not in flags:
            flags.append("unsupported_path_removed")

        return "<path>" + m.group(0)[len(path):]

    return _PATH_RE.sub(_replace_path, fix), flags


def triage_node(state: DiagnosticState) -> DiagnosticState:
    """Node 1: redact sensitive data, extract key entities from the log entry."""
    redacted = redact(state["raw_log"])

    prompt = f"""Extract key entities from this system log entry. Respond ONLY in JSON
with keys: component, error_keywords (list), severity_guess (low/medium/high).

Log entry: {redacted}"""

    entities = _call_llm_json(
        prompt,
        mock_response={
            "component": "mock-component",
            "error_keywords": ["mock-error"],
            "severity_guess": "medium",
        },
    )

    return {**state, "redacted_log": redacted, "entities": entities}


def research_node(state: DiagnosticState, kb_store, kb_lookup) -> DiagnosticState:
    """Node 2: retrieve related historical issues/fixes via RAG (distance-filtered)."""
    from .rag import retrieve_context

    context = retrieve_context(state["redacted_log"], kb_store, kb_lookup, k=3)

    return {**state, "retrieved_context": context}


def report_node(state: DiagnosticState) -> DiagnosticState:
    """Node 3: synthesize triage + research into a final diagnosis."""
    context = state["retrieved_context"]
    context_str = "\n".join(f"- Issue: {c['issue']} | Fix: {c['fix']}" for c in context) \
        or "None. No knowledge-base entry matched closely enough; do not invent one."

    prompt = f"""You are a log diagnostic assistant.

Anomalous log entry (this is the ONLY evidence you may cite):
{state['redacted_log']}

Extracted entities: {json.dumps(state['entities'])}

Related historical issues/fixes from the knowledge base:
{context_str}

Rules:
- Base root_cause on the log entry above. Use a knowledge-base item only if it clearly matches this entry; otherwise ignore it.
- suggested_fix must be generic investigative or remedial guidance. Do NOT include any file path, hostname, IP address, or block ID unless it appears verbatim in the log entry above.
- NEVER suggest commands that delete, overwrite, or format data (rm, -delete, format).
- If the evidence is insufficient, say so in root_cause and set confidence below 0.5.

Respond ONLY in JSON with keys: root_cause, suggested_fix, confidence (0-1)."""

    analysis = _call_llm_json(
        prompt,
        mock_response={
            "root_cause": "mock-root-cause (MOCK_LLM=true)",
            "suggested_fix": "mock-suggested-fix (MOCK_LLM=true)",
            "confidence": 0.5,
        },
    )

    allowed = state["redacted_log"] + "\n" + context_str
    clean_fix, flags = sanitize_suggested_fix(analysis.get("suggested_fix", ""), allowed)
    analysis["suggested_fix"] = clean_fix
    analysis["kb_matches_used"] = len(context)

    if flags:
        analysis["guardrail_flags"] = flags

    return {**state, "final_analysis": analysis}


def build_diagnostic_graph(kb_store, kb_lookup):
    """
    Builds and compiles the LangGraph state graph.
    kb_store / kb_lookup are bound via closure since LangGraph nodes take
    only (state) as input.
    """
    graph = StateGraph(DiagnosticState)
    bound_research_node = partial(research_node, kb_store=kb_store, kb_lookup=kb_lookup)

    graph.add_node("triage", triage_node)
    graph.add_node("research", bound_research_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "research")
    graph.add_edge("research", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_diagnostic_pipeline(raw_log: str, kb_store, kb_lookup) -> dict:
    """Public entry point: runs the full agentic pipeline on one log entry or block context."""
    pipeline = build_diagnostic_graph(kb_store, kb_lookup)
    initial_state: DiagnosticState = {
        "raw_log": raw_log,
        "redacted_log": "",
        "entities": {},
        "retrieved_context": [],
        "final_analysis": {},
    }
    final_state = pipeline.invoke(initial_state)

    return {
        "entities": final_state["entities"],
        "retrieved_context": final_state["retrieved_context"],
        "analysis": final_state["final_analysis"],
    }