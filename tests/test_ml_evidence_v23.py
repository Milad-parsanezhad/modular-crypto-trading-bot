import numpy as np
import pandas as pd

from research_bot.ml_evidence_v23 import (
    COURSE_HYPOTHESIS_FEATURES, MLV23Config, TABULAR_FEATURES,
    assign_segments, build_v23_rows, calibrated_probability,
    conformal_binary_quantile, fit_platt, portfolio_backtest,
)


def synthetic(n=900, seed=23):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.008, n) + 0.0002 * np.sin(np.linspace(0, 30, n))
    close = 100 * np.exp(np.cumsum(ret)); open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.002, rng.uniform(0.001, 0.01, n) * close)
    high = np.maximum(open_, close) + spread * 0.5; low = np.minimum(open_, close) - spread * 0.5
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"), "open": open_, "high": high, "low": low, "close": close, "volume": rng.lognormal(9, 0.5, n)})


def test_promotable_feature_set_excludes_course_hypotheses():
    assert not (set(TABULAR_FEATURES) & set(COURSE_HYPOTHESIS_FEATURES))


def test_v23_rows_are_future_mutation_invariant_for_signal_features():
    df = synthetic(); a = build_v23_rows(df, "BTC/USDT"); changed = df.copy(); cut_source = int(a.loc[100, "source_index"])
    changed.loc[cut_source + 1:, ["open", "high", "low", "close", "volume"]] *= 5.0
    b = build_v23_rows(changed, "BTC/USDT"); cols = list(TABULAR_FEATURES)
    assert np.allclose(a.loc[100, cols].astype(float).to_numpy(), b.loc[100, cols].astype(float).to_numpy(), equal_nan=True)


def test_target_starts_after_signal_bar():
    df = synthetic(); cfg = MLV23Config(); rows = build_v23_rows(df, "BTC/USDT", cfg); row = rows.iloc[25]; i = int(row.source_index)
    expected = float(df.open.iloc[i + 2] / df.open.iloc[i + 1] - 1.0)
    assert np.isclose(row.gross_return, expected); assert int(row.target) == int(expected > cfg.roundtrip_cost)


def test_chronological_segments_are_ordered_and_embargoed():
    cfg = MLV23Config(); rows = build_v23_rows(synthetic(), "BTC/USDT", cfg); z = assign_segments(rows, cfg); spans = {}
    for s in ["development", "calibration", "validation", "final_test"]:
        q = z[z.segment == s]; assert len(q) > 0; spans[s] = (q.timestamp.min(), q.timestamp.max())
    assert spans["development"][1] < spans["calibration"][0] < spans["validation"][0] < spans["final_test"][0]
    assert (z.segment == "embargo").sum() >= 3 * cfg.embargo_bars


def test_platt_and_conformal_outputs_are_finite():
    rng = np.random.default_rng(1); raw = rng.normal(size=300); y = (raw + rng.normal(size=300) > 0).astype(int); cal = fit_platt(raw, y); p = calibrated_probability(cal, raw)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(); c = conformal_binary_quantile(y, p, 0.1); assert 0 <= c["q"] <= 1; assert 0 <= c["singleton_long_threshold"] <= 1


def test_portfolio_caps_and_cost_stress_reduce_return():
    cfg = MLV23Config(max_drawdown=0.90); ts = pd.date_range("2025-01-01", periods=100, freq="4h", tz="UTC")
    rows = pd.DataFrame({"timestamp": np.repeat(ts, 3), "symbol": np.tile(["BTC/USDT", "ETH/USDT", "SOL/USDT"], len(ts)), "gross_return": 0.003, "atr_pct": 0.002}); p = np.full(len(rows), 0.9)
    m12, eq12 = portfolio_backtest(rows, p, 0.6, cfg, 12); m30, _ = portfolio_backtest(rows, p, 0.6, cfg, 30)
    assert eq12.gross_exposure.max() <= cfg.max_portfolio_gross + 1e-12; assert m12["total_return"] > m30["total_return"]


def test_drawdown_kill_switch_is_fail_closed():
    cfg = MLV23Config(max_drawdown=0.05); ts = pd.date_range("2025-01-01", periods=80, freq="4h", tz="UTC")
    rows = pd.DataFrame({"timestamp": ts, "symbol": "BTC/USDT", "gross_return": -0.25, "atr_pct": 0.01}); p = np.full(len(rows), 0.99); m, eq = portfolio_backtest(rows, p, 0.6, cfg, 12)
    assert m["kill_switch_triggered"] is True; k = pd.Timestamp(m["kill_time"]); assert (eq.loc[eq.timestamp > k, "realized_net_return"] == 0).all()
