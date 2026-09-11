from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.regime_event_alpha_v30 import (
    DEVELOPMENT_VENUES_V30,
    FINAL_HOLDOUT_VENUE_V30,
    TOTAL_EFFECTIVE_TRIALS_V30,
    V30_CANDIDATES,
    cross_sectional_dispersion,
    generate_direction_v30,
    persistent_market_regime,
    preregistration_manifest_v30,
)
from research_bot.qualification_v30 import select_v30_winner, v30_decision


def frame(seed: int = 1, n: int = 520, *, freq: str = "4h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.00025, 0.0075, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.0015, 0.009, n)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=8.0, sigma=0.45, size=n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def mutate_future(x: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    y = x.copy()
    idx = y.index > cutoff
    y.loc[idx, ["open", "high", "low", "close"]] *= 3.0
    y.loc[idx, "volume"] *= 20.0
    return y


def test_registry_is_preregistered_and_small() -> None:
    assert len(V30_CANDIDATES) == 12
    assert len({c.name for c in V30_CANDIDATES}) == 12
    assert {c.timeframe for c in V30_CANDIDATES} == {"4h", "1d"}
    assert {c.family for c in V30_CANDIDATES} == {"regime_momentum", "cusum_breakout", "ichimoku_regime"}


def test_manifest_preserves_untouched_kucoin_and_frozen_risk() -> None:
    m = preregistration_manifest_v30()
    assert tuple(m["development_venues"]) == DEVELOPMENT_VENUES_V30
    assert FINAL_HOLDOUT_VENUE_V30 == "kucoin"
    assert FINAL_HOLDOUT_VENUE_V30 not in m["development_venues"]
    assert m["candidate_count"] == 12
    assert m["total_effective_trials"] == TOTAL_EFFECTIVE_TRIALS_V30 == 96
    assert m["round_trip_cost_fraction"] == 0.0024
    assert m["aggregate_open_risk_cap"] == 0.02
    assert m["directional_open_risk_cap"] == 0.015
    assert m["max_drawdown"] == 0.05
    assert m["threshold_relaxation"] is False
    assert m["strategy_parameter_retuning"] is False
    assert m["live_execution_authorized"] is False


def test_market_regime_is_future_mutation_invariant() -> None:
    x = frame(11)
    cutoff = 360
    cutoff_ts = x.loc[cutoff, "timestamp"]
    a = persistent_market_regime(x, "4h")
    b = persistent_market_regime(mutate_future(x, cutoff), "4h")
    pd.testing.assert_series_equal(
        a.loc[a["timestamp"] <= cutoff_ts, "market_regime_v30"].reset_index(drop=True),
        b.loc[b["timestamp"] <= cutoff_ts, "market_regime_v30"].reset_index(drop=True),
    )


def test_dispersion_is_future_mutation_invariant() -> None:
    frames = {f"S{i}/USDT": frame(100 + i) for i in range(5)}
    cutoff = 360
    cutoff_ts = next(iter(frames.values())).loc[cutoff, "timestamp"]
    mutated = {k: mutate_future(v, cutoff) for k, v in frames.items()}
    a = cross_sectional_dispersion(frames, "4h")
    b = cross_sectional_dispersion(mutated, "4h")
    am = a["timestamp"] <= cutoff_ts
    bm = b["timestamp"] <= cutoff_ts
    pd.testing.assert_series_equal(
        a.loc[am, "dispersion_v30"].reset_index(drop=True),
        b.loc[bm, "dispersion_v30"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        a.loc[am, "dispersion_cap_v30"].reset_index(drop=True),
        b.loc[bm, "dispersion_cap_v30"].reset_index(drop=True),
        check_names=False,
    )


def test_v30_signal_is_future_mutation_invariant() -> None:
    target = frame(5)
    btc = frame(6)
    peers = {"BTC/USDT": btc, "ETH/USDT": target, "SOL/USDT": frame(7), "XRP/USDT": frame(8)}
    disp = cross_sectional_dispersion(peers, "4h")
    candidate = next(c for c in V30_CANDIDATES if c.name == "V30_H4_CUSUM_BREAKOUT_3")
    cutoff = 360
    cutoff_ts = target.loc[cutoff, "timestamp"]
    d1, f1 = generate_direction_v30(candidate, target, market_frame=btc, dispersion=disp)

    target2 = mutate_future(target, cutoff)
    btc2 = mutate_future(btc, cutoff)
    peers2 = {k: mutate_future(v, cutoff) for k, v in peers.items()}
    disp2 = cross_sectional_dispersion(peers2, "4h")
    d2, f2 = generate_direction_v30(candidate, target2, market_frame=btc2, dispersion=disp2)

    mask1 = f1["timestamp"] <= cutoff_ts
    mask2 = f2["timestamp"] <= cutoff_ts
    pd.testing.assert_series_equal(d1.loc[mask1].reset_index(drop=True), d2.loc[mask2].reset_index(drop=True))
    for col in ["market_regime_v30", "dispersion_v30", "dispersion_cap_v30", "cusum_positive_v30", "cusum_negative_v30"]:
        pd.testing.assert_series_equal(
            f1.loc[mask1, col].reset_index(drop=True),
            f2.loc[mask2, col].reset_index(drop=True),
            check_names=False,
        )


def eligible_row(name: str, ci: float) -> dict:
    return {
        "strategy": name,
        "timeframe": "4h",
        "development_eligible_v27": True,
        "robust_floor_block_ci_low_v27": ci,
        "robust_floor_breadth_v27": 0.7,
        "robust_floor_profit_factor_v27": 1.2,
        "robust_floor_expectancy_r_v27": 0.1,
        "robust_min_trades_v27": 250,
        "robust_worst_drawdown_v27": 0.03,
    }


def test_winner_is_locked_without_holdout_information() -> None:
    winner = select_v30_winner([eligible_row("B", 0.001), eligible_row("A", 0.002)])
    assert winner is not None and winner["strategy"] == "A"


def test_no_development_candidate_keeps_kucoin_untouched() -> None:
    d = v30_decision(None, None)
    assert d["decision"] == "NO_V30_ROBUST_DEVELOPMENT_CANDIDATE"
    assert d["holdout_used"] is False
    assert d["final_holdout_venue"] == "kucoin"
    assert d["live_execution_authorized"] is False


def test_holdout_hard_drawdown_rejects_before_minimum_count() -> None:
    winner = eligible_row("A", 0.002)
    holdout = {
        "trades": 12,
        "profit_factor": 1.5,
        "expectancy_r": 0.2,
        "positive_asset_fraction": 0.8,
        "max_drawdown": -0.051,
        "block_ci_low": 0.001,
    }
    d = v30_decision(winner, holdout)
    assert d["decision"] == "V30_CANDIDATE_REJECTED_HOLDOUT"
    assert "MAX_DRAWDOWN" in d["holdout_failures"]
    assert d["live_execution_authorized"] is False
