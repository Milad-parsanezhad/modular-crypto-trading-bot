from __future__ import annotations

from fastapi.testclient import TestClient

from research_bot.service import app


client = TestClient(app)


def test_health_is_paper_only():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution_mode"] == "PAPER"
    assert body["live_execution"] is False


def test_status_exposes_evidence_contract_not_profit_claim():
    response = client.get("/research/status")
    assert response.status_code == 200
    body = response.json()
    assert body["principle"] == "Evidence Before Opinion"
    assert body["live_execution"] is False
    assert "paper_execution" in body["active_modules"]


def test_decision_endpoint_can_execute_paper_fill():
    payload = {
        "asset": "BTC",
        "symbol": "BTC/USDT",
        "expected_return": 0.012,
        "expected_cost": 0.0012,
        "risk_penalty": 0.001,
        "uncertainty_penalty": 0.001,
        "confidence": 0.80,
        "currently_long": False,
        "equity": 10000,
        "peak_equity": 10000,
        "gross_exposure": 0.0,
        "asset_weight": 0.0,
        "turnover": 0.0,
        "spread_bps": 4,
        "slippage_bps": 2,
        "reference_price": 100000,
        "quantity": 0.02,
        "client_order_id": "service-test-btc-001",
        "recent_returns": [0.001] * 30,
    }
    response = client.post("/decision/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "EXECUTED_PAPER"
    assert body["fill"]["mode"] == "PAPER"
    assert body["live_execution"] is False
