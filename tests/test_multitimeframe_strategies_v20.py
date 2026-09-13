import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
    apply_candidate_level_policy,
    attach_completed_htf_context,
    generate_direction_v20,
    registry_frame_v20,
)


def synthetic(n=1800, freq="5min", seed=19):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.00005, 0.003, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(0.0008 * close, rng.uniform(0.0003, 0.002, n) * close)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"), "open": open_, "high": high, "low": low, "close": close, "volume": rng.lognormal(8, 0.5, n)})


def test_v20_registry_has_seven_per_timeframe():
    r = registry_frame_v20()
    assert len(r) == 42
    counts = r.groupby("timeframe").size().to_dict()
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        assert counts[tf] == 7


def test_completed_htf_context_is_prefix_invariant():
    df = synthetic(2000, "5min")
    full = attach_completed_htf_context(df, "5m")
    cut = 1400
    prefix = attach_completed_htf_context(df.iloc[:cut].copy(), "5m")
    for c in ["htf_close", "htf_ema50", "htf_cloud_top", "htf_bull_retracement"]:
        assert np.allclose(full.loc[:cut - 1, c].to_numpy(float), prefix[c].to_numpy(float), equal_nan=True)


def test_htf_value_is_unavailable_before_htf_close():
    df = synthetic(500, "5min")
    f = attach_completed_htf_context(df, "5m")
    first_hour = df["timestamp"].iloc[0] + pd.Timedelta(hours=1)
    assert f[f["timestamp"] < first_hour]["htf_close"].isna().all()


def test_all_new_v20_families_generate_valid_direction():
    tf_freq = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    for spec in STRATEGY_REGISTRY_V20[30:]:
        df = synthetic(1800, tf_freq[spec.timeframe])
        d, f = generate_direction_v20(spec, df)
        assert len(d) == len(f) == len(df)
        assert set(pd.Series(d).dropna().unique()).issubset({-1, 0, 1})


def test_loss_streak_cooldown_and_daily_cap_are_deterministic():
    spec = next(s for s in STRATEGY_REGISTRY_V20 if s.timeframe == "5m")
    times = pd.date_range("2025-01-01", periods=12, freq="30min", tz="UTC")
    ledger = pd.DataFrame({
        "strategy": spec.name, "family": spec.family, "timeframe": spec.timeframe, "symbol": ["BTC/USDT"] * 12,
        "signal_time": times - pd.Timedelta(minutes=5), "entry_time": times, "exit_time": times + pd.Timedelta(minutes=5),
        "side": [1] * 12, "entry": [100.0] * 12, "exit": [99.0] * 12, "stop": [99.0] * 12, "target": [103.0] * 12,
        "exit_reason": ["stop"] * 12, "gross_return": [-0.01] * 12, "net_return": [-0.0124] * 12,
        "r_multiple": [-1.1] * 12, "account_return": [-0.00275] * 12, "segment": ["development"] * 12,
        "source_basis": [spec.source_basis] * 12, "risk_scale_volatility": [1.0] * 12,
    })
    out = apply_candidate_level_policy(ledger, spec, RiskPsychologyPolicy())
    assert out["executed_v20"].sum() < len(out)
    assert (out["reject_reason_v20"] == "LOSS_STREAK_COOLDOWN").any() or (out["reject_reason_v20"] == "OVERTRADING_DAILY_CAP").any()
    assert out.loc[out["executed_v20"], "risk_fraction_v20"].max() <= 0.005
