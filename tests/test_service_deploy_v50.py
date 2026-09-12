from __future__ import annotations

from fastapi.testclient import TestClient

from research_bot.service import (
    app,
    LATEST_COMPLETED_DECISION,
    LATEST_COMPLETED_EXPERIMENT,
    LIVE_EXECUTION,
    PAPER_EXECUTION,
    SERVICE_VERSION,
)

client = TestClient(app)


def test_health_is_fail_closed_v50():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == SERVICE_VERSION == "1.0.0-rc3"
    assert body["execution_mode"] == "RESEARCH_ONLY"
    assert body["latest_completed_experiment"] == LATEST_COMPLETED_EXPERIMENT == "v0.50"
    assert body["paper_execution"] is PAPER_EXECUTION is False
    assert body["live_execution"] is LIVE_EXECUTION is False
    assert body["kraken_holdout"] == "SEALED"


def test_research_status_pins_canonical_v50_result():
    r = client.get("/research/status")
    assert r.status_code == 200
    body = r.json()
    assert body["latest_completed_decision"] == LATEST_COMPLETED_DECISION == "V50_NONOVERLAP_FAILURE_SUPPORTED"
    assert body["v50_provenance"]["workflow_run"] == 34707823108
    assert body["v50_provenance"]["artifact_id"] == 10302830689
    assert body["paper_execution"] is False
    assert body["live_execution"] is False


def test_execution_endpoints_remain_locked():
    assert client.post("/decision/evaluate", json={}).status_code == 423
    assert client.post("/paper/run-once").status_code == 423
