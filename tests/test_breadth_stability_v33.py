from __future__ import annotations

import pandas as pd

from research_bot.breadth_stability_v33 import (
    V33_CANDIDATES, apply_breadth_gate, preregistration_manifest_v33,
    screen_three_venues, select_v33_winner, v33_decision,
)


def test_registry_is_frozen_before_kraken():
    m = preregistration_manifest_v33()
    assert len(V33_CANDIDATES) == 6
    assert m["reserved_holdout_venue"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["total_effective_trials"] == 114


def test_breadth_gate_is_causal_shape_preserving():
    f = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=4, tz="UTC"), "close": [1,1,1,1]})
    b = pd.DataFrame({"timestamp": f["timestamp"], "breadth_v33": [0.7,0.7,0.3,0.3]})
    d = pd.Series([1,1,-1,-1], dtype="int8")
    out, feat = apply_breadth_gate(d, f, b, V33_CANDIDATES[0])
    assert len(out) == len(d)
    assert out.tolist() == [1,1,-1,-1]
    assert "breadth_smoothed_v33" in feat


def test_all_three_consumed_venues_must_pass_full_gate():
    good = {"trades": 250, "profit_factor": 1.2, "expectancy_r": 0.1, "positive_asset_fraction": 0.65, "max_drawdown": -0.04, "block_ci_low": 0.0001}
    metrics = {"coinex_consumed": dict(good), "okx_consumed": dict(good), "kucoin_consumed": dict(good)}
    assert screen_three_venues(metrics)["development_eligible_v33"] is True
    metrics["kucoin_consumed"]["positive_asset_fraction"] = 0.55
    assert screen_three_venues(metrics)["development_eligible_v33"] is False


def test_no_winner_never_touches_holdout_or_authorizes_live():
    d = v33_decision(select_v33_winner([]))
    assert d["decision"] == "NO_V33_ROBUST_DEVELOPMENT_CANDIDATE"
    assert d["kraken_touched"] is False
    assert d["live_execution_authorized"] is False
