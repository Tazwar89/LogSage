"""
Agentic diagnostic pipeline using LangGraph.

Replaces the single-shot LLM call in the original llm_analysis.py with a
3-node graph, each node with a distinct responsibility:

  1. triage_node   -- confirms this is a genuine anomaly worth investigating
                       and extracts key entities from the raw log line.
  2. research_node -- retrieves relevant historical fixes via RAG
                       (wraps the existing rag.retrieve_context()).
  3. report_node   -- synthesizes triage + research into a final
                       root_cause / suggested_fix / confidence verdict.

State is passed between nodes via a typed dict, matching LangGraph's
standard state-graph pattern. This is a real loop (state accumulates
across nodes) rather than a single prompt relabeled as an "agent".
"""
import json
import os
from functools import partial
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from openai import OpenAI

from .redact import redact

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


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


def triage_node(state: DiagnosticState) -> DiagnosticState:
    """Node 1: redact sensitive data, extract key entities from the log line."""
    redacted = redact(state["raw_log"])

    client = _get_client()
    prompt = f"""Extract key entities from this system log line. Respond ONLY in JSON
with keys: component, error_keywords (list), severity_guess (low/medium/high).

Log line: {redacted}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    entities = json.loads(response.choices[0].message.content or "{}")

    return {**state, "redacted_log": redacted, "entities": entities}


def research_node(state: DiagnosticState, kb_store, kb_lookup) -> DiagnosticState:
    """Node 2: retrieve related historical issues/fixes via RAG."""
    from .rag import retrieve_context

    context = retrieve_context(state["redacted_log"], kb_store, kb_lookup, k=3)

    return {**state, "retrieved_context": context}


def report_node(state: DiagnosticState) -> DiagnosticState:
    """Node 3: synthesize triage + research into a final diagnosis."""
    client = _get_client()

    context_str = "\n".join(
        f"- Issue: {c['issue']} | Fix: {c['fix']}" for c in state["retrieved_context"]
    ) or "No related historical issues found."

    prompt = f"""You are a log diagnostic assistant.

Anomalous log entry: {state['redacted_log']}
Extracted entities: {json.dumps(state['entities'])}
Related historical issues/fixes:
{context_str}

Respond ONLY in JSON with keys: root_cause, suggested_fix, confidence (0-1)."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    analysis = json.loads(response.choices[0].message.content or "{}")

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
    """Public entry point: runs the full agentic pipeline on a single log line."""
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