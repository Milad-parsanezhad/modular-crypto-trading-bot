import numpy as np
import pandas as pd

from research_bot.scientific_liquidity_wyckoff_v22d import (
    ALL_LIQUIDITY_WYCKOFF_FEATURES,
    COURSE_HYPOTHESIS_FEATURES,
    SUPPORTED_COMPONENT_FEATURES,
    build_scientific_liquidity_features,
    feature_evidence_table,
)


def synthetic(n=900, seed=314):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.00005, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    half = close * rng.uniform(0.001, 0.006, n)
    high = np.maximum(open_, close) + half
    low = np.minimum(open_, close) - half
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.45, n),
    })


def test_evidence_tiers_are_disjoint_and_complete():
    assert set(SUPPORTED_COMPONENT_FEATURES).isdisjoint(COURSE_HYPOTHESIS_FEATURES)
    assert set(ALL_LIQUIDITY_WYCKOFF_FEATURES) == set(SUPPORTED_COMPONENT_FEATURES) | set(COURSE_HYPOTHESIS_FEATURES)
    table = feature_evidence_table()
    assert set(table["feature"]) == set(ALL_LIQUIDITY_WYCKOFF_FEATURES)
    assert set(table["evidence_tier"]) == {"SUPPORTED_COMPONENT", "COURSE_HYPOTHESIS"}


def test_feature_frame_contains_declared_features_and_is_finite_when_ready():
    out = build_scientific_liquidity_features(synthetic())
    assert set(ALL_LIQUIDITY_WYCKOFF_FEATURES).issubset(out.columns)
    tail = out.iloc[300:]
    # At least most supported continuous features should be populated in a mature history.
    populated = tail[list(SUPPORTED_COMPONENT_FEATURES)].notna().mean()
    assert (populated > 0.90).sum() >= 15


def test_future_mutation_cannot_change_past_features():
    df = synthetic()
    cut = 600
    base = build_scientific_liquidity_features(df).iloc[:cut + 1].reset_index(drop=True)
    changed = df.copy()
    changed.loc[cut + 1:, ["open", "high", "low", "close", "volume"]] *= 5.0
    alt = build_scientific_liquidity_features(changed).iloc[:cut + 1].reset_index(drop=True)
    cols = list(ALL_LIQUIDITY_WYCKOFF_FEATURES)
    np.testing.assert_allclose(base[cols].to_numpy(float), alt[cols].to_numpy(float), equal_nan=True)


def test_completed_daily_level_does_not_use_current_day_future_bars():
    df = synthetic()
    out = build_scientific_liquidity_features(df)
    t = pd.Timestamp("2024-02-15 08:00:00", tz="UTC")
    idx = int(out.index[out["timestamp"] == t][0])
    prior = df[(df["timestamp"] >= t.floor("D") - pd.Timedelta(days=1)) & (df["timestamp"] < t.floor("D"))]
    assert out.loc[idx, "prev_day_high"] == prior["high"].max()
    assert out.loc[idx, "prev_day_low"] == prior["low"].min()


def test_course_proxies_are_binary():
    out = build_scientific_liquidity_features(synthetic()).iloc[300:]
    for c in COURSE_HYPOTHESIS_FEATURES:
        assert set(out[c].dropna().unique()).issubset({0.0, 1.0})
