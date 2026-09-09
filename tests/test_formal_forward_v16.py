from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_bot.formal_forward_v16 import build_formal_forward_evaluation, chapter4_markdown


START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _snapshot(i: int, *, total: int = 2, fills: int = 1, gate: bool = False, safety: bool = True, equity: float | None = None) -> dict:
    equity = float(equity if equity is not None else 10_000.0 + i * 5.0)
    peak = max(10_000.0, equity)
    return {
        "evidence_version": "v0.15",
        "captured_at": (START + timedelta(hours=4 * i)).isoformat(),
        "service": {"version": "1.0.0-rc1", "execution_mode": "PAPER", "backend": "postgres"},
        "scientific_state": {"evidence_state": "READY_FOR_FORMAL_STATISTICAL_EVALUATION" if gate else "INSUFFICIENT_FORWARD_SAMPLE", "live_promotion": "PROHIBITED"},
        "paper_account": {
            "equity": equity,
            "cash": equity,
            "peak_equity": peak,
            "cumulative_return": equity / 10_000.0 - 1.0,
            "current_drawdown": equity / peak - 1.0,
            "positions": [],
        },
        "forward_counts": {"observations_total": total, "fills_total": fills, "equity_points_total": total},
        "safety_contract": {"ok": safety},
        "sample_gate": {
            "passed": gate,
            "observed_forward_hours_recent_window": 168.0 if gate else 4.0 * i,
            "minimum_forward_hours": 168.0,
            "minimum_observations": 100,
            "minimum_fills": 10,
        },
    }


def test_small_sample_withholds_formal_risk_ratios_and_live_promotion():
    result = build_formal_forward_evaluation([_snapshot(0), _snapshot(1)])
    assert result["formal_state"] == "INSUFFICIENT_FORWARD_SAMPLE"
    assert result["metric_policy"]["risk_ratios_authorized"] is False
    assert result["paper_performance"]["sharpe"] is None
    assert result["paper_performance"]["sortino"] is None
    assert result["live_promotion"] == "PROHIBITED"
    assert "WITHHELD_UNTIL_GATE" in chapter4_markdown(result)


def test_safety_failure_is_fail_closed_even_when_latest_gate_says_passed():
    rows = [_snapshot(i, total=120, fills=12, gate=True) for i in range(25)]
    rows[5]["safety_contract"]["ok"] = False
    result = build_formal_forward_evaluation(rows)
    assert result["formal_state"] == "SAFETY_CONTRACT_FAILURE"
    assert result["sample_progress"]["preregistered_gate_passed"] is False
    assert result["metric_policy"]["alpha_claim_authorized"] is False
    assert result["live_promotion"] == "PROHIBITED"


def test_sufficient_forward_sample_unlocks_review_metrics_but_not_alpha_or_live():
    rows = []
    for i in range(30):
        equity = 10_000.0 * (1.0 + 0.0005 * i + (0.0002 if i % 2 == 0 else -0.0001))
        rows.append(_snapshot(i, total=130, fills=14, gate=True, equity=equity))
    result = build_formal_forward_evaluation(rows)
    assert result["formal_state"] == "READY_FOR_FORMAL_FORWARD_REVIEW"
    assert result["sample_progress"]["preregistered_gate_passed"] is True
    assert result["metric_policy"]["risk_ratios_authorized"] is True
    assert result["paper_performance"]["period_returns"] >= 20
    assert result["paper_performance"]["sharpe"] is not None
    assert result["metric_policy"]["profitability_claim_authorized"] is False
    assert result["metric_policy"]["alpha_claim_authorized"] is False
    assert result["live_promotion"] == "PROHIBITED"


def test_duplicate_snapshot_timestamp_is_deduplicated():
    a = _snapshot(0)
    b = _snapshot(1)
    b["captured_at"] = a["captured_at"]
    result = build_formal_forward_evaluation([a, b])
    assert result["source_contract"]["snapshot_count"] == 1
