import pandas as pd

from research_bot.fresh_temporal_oos_v28 import (
    REQUIRED_V27_DECISION,
    filter_strictly_post_lock,
    v27_prerequisite_failures,
    v28_decision,
)


def good_v27():
    return {
        "decision": REQUIRED_V27_DECISION,
        "holdout_used": True,
        "winner": "H4_X",
        "timeframe": "4h",
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "holdout_winner_reselection": False,
    }


def good_lock():
    return {
        "winner": "H4_X",
        "timeframe": "4h",
        "locked_at_utc": "2026-09-11T18:00:00Z",
    }


def good_metrics(**overrides):
    x = {
        "temporal_trades": 250,
        "temporal_profit_factor": 1.20,
        "temporal_expectancy_r": 0.10,
        "temporal_max_drawdown": -0.04,
        "temporal_positive_asset_fraction": 0.70,
        "temporal_block_ci_low": 0.0001,
    }
    x.update(overrides)
    return x


def test_strictly_post_lock_filter_excludes_equal_and_prior_rows():
    attempts = pd.DataFrame({
        "signal_time": [
            "2026-09-11T17:59:59Z",
            "2026-09-11T18:00:00Z",
            "2026-09-11T18:00:01Z",
        ],
        "x": [1, 2, 3],
    })
    out = filter_strictly_post_lock(attempts, good_lock())
    assert out["x"].tolist() == [3]


def test_v28_is_blocked_unless_v27_untouched_holdout_passed():
    v27 = good_v27()
    v27["decision"] = "ROBUST_CANDIDATE_REJECTED_HOLDOUT"
    result = v28_decision(v27, good_lock(), good_metrics())
    assert result["decision"] == "BLOCKED_BY_V27"
    assert result["forward_paper_candidate_authorized"] is False


def test_v28_rejects_lock_winner_mismatch():
    lock = good_lock()
    lock["winner"] = "OTHER"
    failures = v27_prerequisite_failures(good_v27(), lock)
    assert "V27_LOCK_MISMATCH" in failures


def test_v28_accumulates_without_minimum_trade_count():
    result = v28_decision(good_v27(), good_lock(), good_metrics(temporal_trades=199))
    assert result["decision"] == "FRESH_TEMPORAL_OOS_ACCUMULATING"
    assert result["forward_paper_candidate_authorized"] is False


def test_hard_drawdown_rejects_even_below_minimum_trade_count():
    result = v28_decision(
        good_v27(),
        good_lock(),
        good_metrics(temporal_trades=40, temporal_max_drawdown=-0.051),
    )
    assert result["decision"] == "REJECTED_FRESH_TEMPORAL_OOS"
    assert result["hard_drawdown_breached"] is True


def test_full_frozen_gate_pass_is_only_forward_paper_candidate():
    result = v28_decision(good_v27(), good_lock(), good_metrics())
    assert result["decision"] == "FORWARD_PAPER_CANDIDATE"
    assert result["forward_paper_candidate_authorized"] is True
    assert result["paper_replacement_authorized"] is False
    assert result["live_execution_authorized"] is False
