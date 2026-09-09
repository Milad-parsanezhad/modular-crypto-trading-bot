from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_bot.forward_evidence_v15 import build_forward_evidence, evidence_markdown


def _payload(observation_count: int = 2, fill_count: int = 1, hours: int = 4):
    start = datetime(2026, 9, 9, 8, tzinfo=timezone.utc)
    observations = []
    for i in range(observation_count):
        observations.append({
            "observed_at": (start + timedelta(hours=(hours * i / max(1, observation_count - 1)))).isoformat(),
            "bar_timestamp": start.isoformat(),
            "symbol": "BTC/USDT" if i % 2 == 0 else "ETH/USDT",
            "action": "NO_TRADE" if i % 2 == 0 else "HOLD",
            "rule_score": 0.6,
            "spread_bps": 4.0,
        })
    fills = []
    for i in range(fill_count):
        fills.append({
            "timestamp": start.isoformat(),
            "symbol": "ETH/USDT",
            "side": "BUY",
            "filled_quantity": 0.5,
            "fill_price": 2500.0,
            "fee_paid": 1.25,
        })
    return {
        "health": {"status": "ok", "version": "1.0.0-rc1", "execution_mode": "PAPER", "forward_paper_enabled": True, "live_execution": False},
        "research": {"live_execution": False, "latest_completed_research_gate": "v0.12 external derivatives holdout", "latest_research_decision": "NO_INCREMENTAL_DERIVATIVES_EVIDENCE", "forward_paper_label": "HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA"},
        "paper": {"status": "RUNNING", "paper_execution_enabled": True, "live_execution": False, "strategy_version": "ICHIMOKU_SHADOW_V14", "cumulative_return": -0.0003, "current_drawdown": -0.0003, "store": {"backend": "postgres", "account": {"cash": 8000.0, "equity": 9997.0, "peak_equity": 10000.0}, "positions": [{"symbol": "ETH/USDT", "quantity": 0.5, "avg_price": 2500.0}], "observations": observation_count, "fills": fill_count, "equity_points": observation_count}},
        "observations_payload": {"items": observations},
        "fills_payload": {"items": fills},
    }


def test_small_sample_is_not_promotable():
    evidence = build_forward_evidence(**_payload())
    assert evidence["safety_contract"]["ok"] is True
    assert evidence["scientific_state"]["evidence_state"] == "INSUFFICIENT_FORWARD_SAMPLE"
    assert evidence["scientific_state"]["live_promotion"] == "PROHIBITED"
    assert evidence["sample_gate"]["passed"] is False
    text = evidence_markdown(evidence)
    assert "No alpha" in text


def test_large_descriptive_sample_still_requires_formal_review():
    evidence = build_forward_evidence(**_payload(observation_count=120, fill_count=12, hours=200))
    assert evidence["sample_gate"]["passed"] is True
    assert evidence["scientific_state"]["evidence_state"] == "READY_FOR_FORMAL_STATISTICAL_EVALUATION"
    assert evidence["scientific_state"]["live_promotion"] == "PROHIBITED"


def test_live_flag_breaks_safety_contract():
    payload = _payload()
    payload["health"]["live_execution"] = True
    evidence = build_forward_evidence(**payload)
    assert evidence["safety_contract"]["ok"] is False
    assert evidence["scientific_state"]["evidence_state"] == "SAFETY_CONTRACT_FAILURE"
    assert evidence["scientific_state"]["live_promotion"] == "PROHIBITED"
