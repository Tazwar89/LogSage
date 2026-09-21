"""
REST contract tests for analysis_service.

Heavy dependencies (Qdrant, Redis, embedding model, LLM pipeline) are stubbed
in sys.modules before main.py is imported, so these run instantly and offline.
"""
import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


class FakeLogStore:
    entries = {}

    def get(self, tid):
        return self.entries.get(tid)


    def list_trace_ids(self):
        return sorted(self.entries)


class FakeQdrant:
    missing = False

    def __init__(self, collection_name):
        self.collection_name = collection_name


    def load(self, path=None):
        if FakeQdrant.missing:
            raise FileNotFoundError("no collection")


@pytest.fixture
def api(monkeypatch):
    FakeLogStore.entries = {"t1": {"message": "disk failure on node 3"}, "t2": {"message": "ok"}, "t3": {"message": "ok2"}}
    FakeQdrant.missing = False

    stubs = {
        "logsage_common.vector_store_qdrant": types.SimpleNamespace(QdrantVectorStore=FakeQdrant),
        "logsage_common.log_store": types.SimpleNamespace(LogStore=FakeLogStore),
        "analysis_service.app.rag": types.SimpleNamespace(
            load_knowledge_base=list, build_kb_index=lambda s, e: {}, retrieve_context=lambda *a, **k: []
        ),
        "analysis_service.app.agentic_pipeline": types.SimpleNamespace(
            run_diagnostic_pipeline=lambda msg, store, lookup: {"root_cause": "disk", "suggested_fix": "replace disk"}
        ),
        "analysis_service.app.analytics": types.SimpleNamespace(compute_log_stats=lambda entries: {"total": len(entries)}),
    }

    for name, mod in stubs.items():
        m = types.ModuleType(name)
        m.__dict__.update(vars(mod))
        monkeypatch.setitem(sys.modules, name, m)

    sys.modules.pop("analysis_service.app.main", None)
    main = importlib.import_module("analysis_service.app.main")
    monkeypatch.setattr(main, "is_anomalous", lambda text, store: ("disk" in text, {"template_id": 1}))
    monkeypatch.delenv("LOGSAGE_API_KEY", raising=False)
    yield main, TestClient(main.app)
    sys.modules.pop("analysis_service.app.main", None)


def test_health_v1(api):
    _, c = api
    r = c.get("/api/v1/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "service": "analysis"}


def test_analyze_anomalous_includes_pipeline_fields(api):
    _, c = api
    body = c.get("/api/v1/analyze/t1").json()
    assert body["anomalous"] is True and body["root_cause"] == "disk" and body["trace_id"] == "t1"


def test_analyze_normal_returns_nearest_match(api):
    _, c = api
    body = c.get("/api/v1/analyze/t2").json()
    assert body["anomalous"] is False and body["nearest_match"] == {"template_id": 1}


def test_analyze_unknown_trace_is_404(api):
    _, c = api
    assert c.get("/api/v1/analyze/nope").status_code == 404


def test_analyze_without_baseline_is_503(api):
    _, c = api
    FakeQdrant.missing = True
    assert c.get("/api/v1/analyze/t1").status_code == 503


def test_logs_pagination(api):
    _, c = api
    body = c.get("/api/v1/logs?limit=2&offset=1").json()
    assert body == {"total": 3, "limit": 2, "offset": 1, "items": ["t2", "t3"]}


@pytest.mark.parametrize("qs", ["limit=0", "limit=501", "offset=-1"])
def test_logs_rejects_bad_pagination(api, qs):
    _, c = api
    assert c.get(f"/api/v1/logs?{qs}").status_code == 422


def test_stats_v1(api):
    _, c = api
    assert c.get("/api/v1/stats").json() == {"total": 3}


def test_legacy_routes_still_work(api):
    _, c = api
    assert c.get("/health").status_code == 200
    assert c.get("/logs").json() == ["t1", "t2", "t3"]
    assert c.get("/analyze/t2").json()["anomalous"] is False
    assert c.get("/stats").status_code == 200


def test_api_key_enforced_when_configured(api, monkeypatch):
    _, c = api
    monkeypatch.setenv("LOGSAGE_API_KEY", "s3cret")
    assert c.get("/api/v1/logs").status_code == 401
    assert c.get("/api/v1/logs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/v1/logs", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/logs").status_code == 401  # legacy routes are protected too
    assert c.get("/health").status_code == 200  # liveness stays open


def test_openapi_documents_v1_routes_and_deprecates_legacy(api):
    _, c = api
    spec = c.get("/openapi.json").json()
    assert "/api/v1/analyze/{trace_id}" in spec["paths"]
    assert spec["paths"]["/logs"]["get"]["deprecated"] is True
    assert "AnalyzeResponse" in spec["components"]["schemas"]