from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.temporal_consensus_v38 import (
    MODEL_NAMES_V38,
    TOTAL_EFFECTIVE_TRIALS_V38,
    V38_CANDIDATES,
    apply_candidate_v38,
    build_consensus_panel_v38,
    decision_v38,
    preregistration_manifest_v38,
    screen_three_venues_v38,
)


def _panel(model: str, n: int = 240) -> pd.DataFrame:
    t = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    offset = {"V36_EXPECTED_NETR_RIDGE": -0.05, "V36_EXPECTED_NETR_HUBER": 0.0, "V36_EXPECTED_NETR_HISTGB": 0.05}[model]
    u = np.linspace(-1.0, 1.0, n) + offset
    return pd.DataFrame({
        "signal_time": t,
        "entry_time": t + pd.Timedelta(days=1),
        "exit_time": t + pd.Timedelta(days=3),
        "symbol": [f"S{i%20}/USDT" for i in range(n)],
        "base_strategy_v36": [f"B{i%6}" for i in range(n)],
        "side": np.where(np.arange(n)%2==0, 1, -1),
        "entry": 100.0,
        "stop": 95.0,
        "r_multiple": 0.25 + 0.05*np.sin(np.arange(n)),
        "f_regime_v36": np.where(np.arange(n)%3==0, -1, 1),
        "utility_v36": u,
        "selected_v36": u >= np.median(u),
        "lower_r_v36": u - 0.1,
        "predicted_r_v36": u + 0.2,
        "conformal_buffer_v36": 0.3,
    })


def test_manifest_is_fail_closed_and_counts_trials():
    m = preregistration_manifest_v38()
    assert m["candidate_count"] == 5
    assert m["total_effective_trials"] == TOTAL_EFFECTIVE_TRIALS_V38 == 137
    assert m["reserved_holdout_venue"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["live_execution_authorized"] is False


def test_consensus_merge_is_one_to_one():
    x = build_consensus_panel_v38({m: _panel(m) for m in MODEL_NAMES_V38})
    assert len(x) == 240
    assert x["utility_histgb_v38"].notna().all()


def test_quarter_balancing_preserves_multiple_time_blocks():
    x = build_consensus_panel_v38({m: _panel(m) for m in MODEL_NAMES_V38})
    c = next(c for c in V38_CANDIDATES if c.name == "V38_MEDIAN_UTILITY_QUARTER_BALANCED")
    z = apply_candidate_v38(x, c)
    q = pd.to_datetime(z["signal_time"], utc=True).dt.to_period("Q").astype(str)
    counts = z.assign(q=q).groupby("q")["selected_v38"].sum()
    assert (counts > 0).all()
    assert z["selected_v38"].sum() >= 100


def test_two_of_three_quorum_uses_only_frozen_selection_flags():
    x = build_consensus_panel_v38({m: _panel(m) for m in MODEL_NAMES_V38})
    c = next(c for c in V38_CANDIDATES if c.name == "V38_TWO_OF_THREE_SELECTION_QUORUM")
    z = apply_candidate_v38(x, c)
    expected = z[["selected_ridge_v38", "selected_huber_v38", "selected_histgb_v38"]].astype(bool).sum(axis=1) >= 2
    assert z["selected_v38"].equals(expected)


def test_screen_requires_all_frozen_gates():
    good = {
        "selected_events": 220,
        "profit_factor": 1.2,
        "expectancy_r": 0.1,
        "positive_asset_fraction": 0.7,
        "block_ci_low": 0.0001,
        "positive_quarter_fraction": 0.8,
        "stress_36bps_profit_factor": 1.1,
    }
    metrics = {v: dict(good) for v in ("coinex_consumed", "okx_consumed", "kucoin_consumed")}
    assert screen_three_venues_v38(metrics)["development_eligible_v38"] is True
    metrics["coinex_consumed"]["block_ci_low"] = -1e-6
    assert screen_three_venues_v38(metrics)["development_eligible_v38"] is False


def test_no_winner_never_authorizes_holdout_or_live():
    d = decision_v38(None)
    assert d["kraken_touched"] is False
    assert d["holdout_authorized_to_run_in_this_stage"] is False
    assert d["paper_replacement_authorized"] is False
    assert d["live_execution_authorized"] is False
