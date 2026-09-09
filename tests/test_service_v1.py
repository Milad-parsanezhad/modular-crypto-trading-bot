from __future__ import annotations

from fastapi.testclient import TestClient

from research_bot.service import app, SERVICE_VERSION


client = TestClient(app)


def test_v1_health_and_dashboard_are_paper_only():
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["version"] == SERVICE_VERSION
    assert body["execution_mode"] == "PAPER"
    assert body["live_execution"] is False
    d = client.get("/dashboard")
    assert d.status_code == 200
    assert "LIVE disabled" in d.text
    assert "Thesis Research Dashboard" in d.text


def test_v1_research_status_retains_negative_v12_result():
    r = client.get("/research/status")
    assert r.status_code == 200
    body = r.json()
    assert body["latest_research_decision"] == "NO_INCREMENTAL_DERIVATIVES_EVIDENCE"
    assert body["forward_paper_label"] == "HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA"
    assert body["live_execution"] is False


def test_paper_run_once_is_disabled_without_explicit_environment_flag():
    r = client.post("/paper/run-once")
    assert r.status_code == 200
    assert r.json()["status"] == "DISABLED"
    assert r.json()["live_execution"] is False
