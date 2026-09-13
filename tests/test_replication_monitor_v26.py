from __future__ import annotations

import pandas as pd

from research_bot.replication_monitor_v26 import (
    V25_SELECTION_LOCK_UTC,
    compact_replication_metrics,
    filter_fresh_temporal_attempts,
    qualification_decision,
)


def _metrics(prefix: str, *, trades: int = 250, passed: bool = True) -> dict:
    return {
        f"{prefix}_trades": trades,
        f"{prefix}_profit_factor": 1.30,
        f"{prefix}_expectancy_r": 0.20,
        f"{prefix}_max_drawdown": -0.03,
        f"{prefix}_positive_asset_fraction": 0.70,
        f"{prefix}_block_ci_low": 0.0002,
        f"{prefix}_pass": passed,
    }


def test_fresh_filter_excludes_prelock_and_equal_lock_signal() -> None:
    a = V25_SELECTION_LOCK_UTC
    frame = pd.DataFrame(
        {
            "signal_time": [a - pd.Timedelta(hours=1), a, a + pd.Timedelta(hours=1)],
            "entry_time": [a, a + pd.Timedelta(minutes=1), a + pd.Timedelta(hours=2)],
        }
    )
    out = filter_fresh_temporal_attempts(frame)
    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["signal_time"]) == a + pd.Timedelta(hours=1)


def test_external_fail_blocks_promotion_even_if_temporal_passes() -> None:
    ext = _metrics("external", passed=False)
    ext["external_profit_factor"] = 0.90
    tmp = _metrics("temporal", passed=True)
    decision = qualification_decision(ext, tmp)
    assert decision["decision"] == "REJECTED_EXTERNAL_REPLICATION"
    assert decision["forward_paper_candidate_authorized"] is False
    assert decision["live_execution_authorized"] is False


def test_hard_drawdown_rejects_before_minimum_trade_count() -> None:
    ext = _metrics("external", trades=38, passed=False)
    ext["external_profit_factor"] = 0.1089744791033107
    ext["external_expectancy_r"] = -0.9146516547945169
    ext["external_positive_asset_fraction"] = 0.0625
    ext["external_max_drawdown"] = -0.05090803436384295
    ext["external_block_ci_low"] = None
    tmp = _metrics("temporal", trades=0, passed=False)
    decision = qualification_decision(ext, tmp)
    assert decision["decision"] == "REJECTED_EXTERNAL_REPLICATION"
    assert decision["external_hard_drawdown_breached"] is True
    assert "5% hard drawdown" in decision["reason"]
    assert decision["forward_paper_candidate_authorized"] is False


def test_external_pass_waits_for_fresh_temporal_sample() -> None:
    ext = _metrics("external", passed=True)
    tmp = _metrics("temporal", trades=12, passed=False)
    decision = qualification_decision(ext, tmp)
    assert decision["decision"] == "EXTERNAL_PASS_TEMPORAL_ACCUMULATING"
    assert decision["forward_paper_candidate_authorized"] is False


def test_both_pass_authorizes_only_forward_paper_candidate() -> None:
    decision = qualification_decision(_metrics("external"), _metrics("temporal"))
    assert decision["decision"] == "FORWARD_PAPER_CANDIDATE"
    assert decision["forward_paper_candidate_authorized"] is True
    assert decision["paper_replacement_authorized"] is False
    assert decision["live_execution_authorized"] is False


def test_compact_replication_metrics_preserves_v25_gate() -> None:
    raw = {
        "external_trades": 222,
        "external_total_return": 0.42,
        "external_profit_factor": 1.4,
        "external_win_rate": 0.48,
        "external_expectancy_r": 0.21,
        "external_max_drawdown": -0.04,
        "external_positive_asset_fraction": 0.75,
        "external_block_ci_low": 0.0001,
        "external_block_ci_high": 0.001,
        "external_pass_v25": True,
    }
    out = compact_replication_metrics(raw, prefix="external")
    assert out["external_trades"] == 222
    assert out["external_pass"] is True
