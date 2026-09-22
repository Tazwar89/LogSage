import os
import pytest


@pytest.fixture(autouse=True)
def _mock_llm_env(monkeypatch):
    """
    Forces MOCK_LLM=true for tests in this directory only, so run_eval,
    agentic_pipeline, and judge (which all read MOCK_LLM at import time)
    exercise their offline/mocked code paths without needing real API keys.
    Scoped as an autouse fixture rather than a module-level os.environ
    assignment so it can't leak into test sessions that collect this
    conftest.py but don't run eval/ tests.
    """
    monkeypatch.setenv("MOCK_LLM", "true")