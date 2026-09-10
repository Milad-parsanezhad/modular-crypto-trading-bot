import numpy as np
import pandas as pd
import pytest

from research_bot.deep_temporal_v22 import TemporalConfig, build_sequences, causal_bar_features, economic_backtest


def synthetic(n=700, seed=22):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0001, 0.006, n)
    close = 100 * np.exp(np.cumsum(r)); open_ = np.r_[close[0], close[:-1]]
    span = close * rng.uniform(0.001, 0.01, n)
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"), "open": open_, "high": np.maximum(open_, close) + span, "low": np.minimum(open_, close) - span, "close": close, "volume": rng.lognormal(8, 0.4, n)})


def test_causal_bar_features_prefix_invariant():
    df = synthetic()
    full = causal_bar_features(df)
    cut = 550
    pre = causal_bar_features(df.iloc[:cut])
    cols = [c for c in full if c != "timestamp"]
    for c in cols:
        assert np.allclose(full[c].iloc[:cut].to_numpy(float), pre[c].to_numpy(float), equal_nan=True)


def test_sequence_builder_never_uses_future_in_input():
    df = synthetic()
    cfg = TemporalConfig(lookback=32, horizon=1)
    a = build_sequences(df, cfg)
    changed = df.copy(); changed.loc[600:, "close"] *= 7; changed.loc[600:, "high"] *= 7; changed.loc[600:, "low"] *= 7; changed.loc[600:, "open"] *= 7
    b = build_sequences(changed, cfg)
    Xa, _, _, ta, _ = a; Xb, _, _, tb, _ = b
    common = min(np.searchsorted(ta.astype("int64"), pd.Timestamp(df.timestamp.iloc[599]).value, side="left"), np.searchsorted(tb.astype("int64"), pd.Timestamp(df.timestamp.iloc[599]).value, side="left"))
    assert common > 50
    assert np.allclose(Xa[:common], Xb[:common], equal_nan=True)


def test_economic_backtest_charges_position_changes():
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9])
    r = np.array([0.01, 0.01, -0.01, -0.01, 0.01])
    no_cost = economic_backtest(p, r, 0.6, cost_bps_each_way=0)
    with_cost = economic_backtest(p, r, 0.6, cost_bps_each_way=12)
    assert with_cost["total_log_return"] < no_cost["total_log_return"]
    assert with_cost["turnover_units"] > 0


def test_lstm_forward_smoke_when_torch_installed():
    torch = pytest.importorskip("torch")
    from research_bot.deep_temporal_v22 import make_model
    model = make_model("lstm", 15, TemporalConfig(hidden_dim=16, layers=1))
    y = model(torch.zeros(4, 32, 15))
    assert tuple(y.shape) == (4,)
