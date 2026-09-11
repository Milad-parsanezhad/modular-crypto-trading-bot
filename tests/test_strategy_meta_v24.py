import numpy as np
import pandas as pd

from research_bot.strategy_meta_v24 import (
    V24Contract,
    causal_cusum_features,
    choose_meta_threshold,
    cost_stress_table,
    cscv_pbo_diagnostic,
    deflated_sharpe_probability,
    economic_metrics,
    promotion_decision,
    target_specs,
)


def synthetic_ohlcv(n=1200, seed=24):
    rng = np.random.default_rng(seed)
    drift = 0.00035 + 0.0015 * np.sin(np.arange(n) / 60.0)
    ret = drift + rng.normal(0, 0.006, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = close * (0.002 + rng.random(n) * 0.004)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(10, 0.45, n),
    })


def synthetic_events(n=1000, seed=314):
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.38
    r = np.where(good, rng.uniform(0.4, 3.0, n), -rng.uniform(0.5, 1.25, n))
    entry = np.full(n, 100.0)
    stop = np.where(np.arange(n) % 2 == 0, 99.0, 101.0)
    gross = r * 0.01 + 0.0024
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "signal_time": t,
        "entry_time": t + pd.Timedelta(hours=4),
        "exit_time": t + pd.Timedelta(hours=8),
        "strategy": np.where(np.arange(n) % 2, "A", "B"),
        "symbol": np.where(np.arange(n) % 3, "BTC/USDT", "ETH/USDT"),
        "entry": entry, "stop": stop, "gross_return": gross,
        "r_multiple": r,
    })


def test_target_registry_is_frozen_to_seven_4h_strategies():
    specs = target_specs()
    assert len(specs) == 7
    assert all(s.timeframe == "4h" for s in specs)
    assert {s.name for s in specs} >= {"H4_S6_BREAKOUT", "H4_D1_S6_VOL_RISK", "H4_D1_OB_BOS_RISK"}


def test_cusum_prefix_invariance():
    frame = synthetic_ohlcv()
    cut = 800
    full = causal_cusum_features(frame)
    prefix = causal_cusum_features(frame.iloc[:cut].copy())
    cols = ["f_cusum_event", "f_cusum_abs_event", "f_cusum_threshold", "f_cusum_age"]
    for c in cols:
        assert np.allclose(full.loc[:cut - 1, c].to_numpy(float), prefix[c].to_numpy(float), equal_nan=True)


def test_cusum_future_mutation_does_not_change_past():
    frame = synthetic_ohlcv()
    cut = 800
    original = causal_cusum_features(frame)
    mutated = frame.copy()
    mutated.loc[cut:, "close"] *= np.linspace(0.5, 1.8, len(mutated) - cut)
    changed = causal_cusum_features(mutated)
    assert np.allclose(
        original.loc[:cut - 1, ["f_cusum_event", "f_cusum_threshold"]].to_numpy(float),
        changed.loc[:cut - 1, ["f_cusum_event", "f_cusum_threshold"]].to_numpy(float),
        equal_nan=True,
    )


def test_cost_stress_is_monotonic_for_same_selected_events():
    rows = synthetic_events()
    selected = np.ones(len(rows), dtype=bool)
    table = cost_stress_table(rows, selected, V24Contract())
    mean_r = table["filtered_mean_r"].to_numpy(float)
    assert mean_r[0] > mean_r[1] > mean_r[2]


def test_meta_threshold_can_find_incremental_filter():
    rows = synthetic_events()
    score = np.where(rows["r_multiple"].to_numpy() > 0, 0.8, 0.2)
    threshold, result = choose_meta_threshold(rows, score, V24Contract(min_validation_selected=100))
    assert 0 <= threshold <= 1
    assert result["filtered"]["mean_r"] > result["base"]["mean_r"]
    assert result["filtered"]["profit_factor"] > result["base"]["profit_factor"]


def test_dsr_probability_prefers_strong_return_series():
    rng = np.random.default_rng(1)
    trials = rng.normal(0.02, 0.03, 60)
    good = rng.normal(0.004, 0.008, 500)
    bad = rng.normal(-0.001, 0.008, 500)
    dg = deflated_sharpe_probability(good, trial_sharpes=trials)
    db = deflated_sharpe_probability(bad, trial_sharpes=trials)
    assert 0 <= dg["probability"] <= 1
    assert 0 <= db["probability"] <= 1
    assert dg["probability"] > db["probability"]


def test_cscv_pbo_reports_valid_probability():
    rng = np.random.default_rng(2)
    n = 600
    matrix = pd.DataFrame({
        "stable": rng.normal(0.0015, 0.004, n),
        "noise1": rng.normal(0.0, 0.004, n),
        "noise2": rng.normal(-0.0002, 0.004, n),
        "noise3": rng.normal(0.0001, 0.004, n),
    })
    result = cscv_pbo_diagnostic(matrix, groups=6)
    assert result["paths"] > 0
    assert 0 <= result["pbo"] <= 1
    assert 0 <= result["median_oos_rank_percentile"] <= 1


def test_promotion_fails_closed_for_nonprofitable_filter():
    rows = synthetic_events(400)
    selected = rows["r_multiple"].to_numpy() < 0
    stress = cost_stress_table(rows, selected, V24Contract())
    decision = promotion_decision(
        rows.reset_index(drop=True), selected,
        validation_pbo={"pbo": 0.1},
        dsr={"probability": 0.99},
        bootstrap_ci=(-0.001, 0.001),
        cost_stress=stress,
        contract=V24Contract(min_test_selected=50),
    )
    assert decision["decision"] == "NO_META_MODEL_PROMOTED"
    assert decision["forward_paper_authorized"] is False
    assert decision["live_execution_authorized"] is False


def test_economic_metrics_filter_is_event_level_not_accuracy_metric():
    rows = synthetic_events()
    selected = rows["r_multiple"].to_numpy() > 0
    base = economic_metrics(rows)
    filt = economic_metrics(rows, selected)
    assert filt["selected"] < base["selected"]
    assert filt["mean_r"] > base["mean_r"]
    assert filt["profit_factor"] > base["profit_factor"]
