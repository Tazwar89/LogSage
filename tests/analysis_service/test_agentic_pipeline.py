"""
Tests for app/agentic_pipeline.py

The LLM client (_get_client) is mocked throughout so these tests run
offline, deterministically, and without spending API credits. They verify
the graph wiring and state propagation across nodes, not the LLM's actual
reasoning quality.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from app.agentic_pipeline import (
    build_diagnostic_graph,
    run_diagnostic_pipeline,
    triage_node,
    report_node,
    DiagnosticState
)


def make_fake_llm_response(payload: dict):
    """Builds a MagicMock shaped like an OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)

    return response


@pytest.fixture
def mock_llm_client():
    """
    Patches _get_client so every call returns the same fake client, and lets
    the caller configure what JSON payload each response contains.
    """
    with patch("app.agentic_pipeline._get_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        yield client


class TestTriageNode:
    def test_extracts_entities_and_redacts_log(self, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = make_fake_llm_response(
            {"component": "DataNode", "error_keywords": ["timeout"], "severity_guess": "high"}
        )

        state: DiagnosticState = {
            "raw_log": "connection to 10.0.0.5 timed out",
            "redacted_log": "",
            "entities": {},
            "retrieved_context": [],
            "final_analysis": {},
        }
        result = triage_node(state)

        assert result["entities"]["component"] == "DataNode"
        assert "10.0.0.5" not in result["redacted_log"]  # IP should be redacted before leaving this node


class TestReportNode:
    def test_synthesizes_final_analysis_from_state(self, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = make_fake_llm_response(
            {"root_cause": "disk failure", "suggested_fix": "replace disk", "confidence": 0.9}
        )

        state: DiagnosticState = {
            "raw_log": "raw",
            "redacted_log": "disk error on node 3",
            "entities": {"component": "DataNode"},
            "retrieved_context": [{"issue": "disk failure", "fix": "replace disk"}],
            "final_analysis": {},
        }
        result = report_node(state)

        assert result["final_analysis"]["root_cause"] == "disk failure"
        assert result["final_analysis"]["confidence"] == 0.9


class TestFullPipeline:
    def test_pipeline_runs_all_three_nodes_in_order(self, mock_llm_client):
        # Distinguish node outputs by returning different payloads per call
        mock_llm_client.chat.completions.create.side_effect = [
            make_fake_llm_response(
                {"component": "PacketResponder", "error_keywords": ["breach"], "severity_guess": "high"}
            ),
            make_fake_llm_response(
                {"root_cause": "security breach", "suggested_fix": "rotate keys", "confidence": 0.85}
            ),
        ]

        fake_kb_store = MagicMock()
        fake_kb_store.query.return_value = [{"template_id": 0, "text": "known issue", "distance": 0.1}]
        fake_kb_lookup = {0: {"issue": "known issue", "fix": "known fix"}}

        result = run_diagnostic_pipeline(
            "CRITICAL SECURITY BREACH detected", kb_store=fake_kb_store, kb_lookup=fake_kb_lookup
        )

        assert result["entities"]["component"] == "PacketResponder"
        assert result["analysis"]["root_cause"] == "security breach"
        assert result["retrieved_context"] == [{"issue": "known issue", "fix": "known fix"}]
        # Confirms both triage and report LLM calls actually happened
        assert mock_llm_client.chat.completions.create.call_count == 2


    def test_graph_compiles_without_kb_store(self):
        """The graph object itself should compile successfully regardless of
        what kb_store/kb_lookup are bound to -- research_node's actual use of
        them is only exercised when the graph is invoked (see test above)."""
        graph = build_diagnostic_graph(kb_store=None, kb_lookup=None)
        assert graph is not None