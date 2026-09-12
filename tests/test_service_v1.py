from __future__ import annotations

from fastapi.testclient import TestClient

from research_bot.service import app, SERVICE_VERSION


client = TestClient(app)


def test_v1_legacy_dashboard_now_reports_research_only():
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["version"] == SERVICE_VERSION
    assert body["execution_mode"] == "RESEARCH_ONLY"
    assert body["paper_execution"] is False
    assert body["live_execution"] is False
    d = client.get("/dashboard")
    assert d.status_code == 200
    assert "LIVE=false" in d.text
    assert "Thesis Research Dashboard" in d.text


def test_v1_status_exposes_current_v50_negative_result():
    r = client.get("/research/status")
    assert r.status_code == 200
    body = r.json()
    assert body["latest_completed_experiment"] == "v0.50"
    assert body["latest_completed_decision"] == "V50_NONOVERLAP_FAILURE_SUPPORTED"
    assert body["paper_execution"] is False
    assert body["live_execution"] is False


def test_paper_run_once_is_disabled_without_explicit_environment_flag():
    r = client.post("/paper/run-once")
    assert r.status_code == 423
    assert "disabled" in r.json()["detail"].lower()
