import numpy as np
import pandas as pd
import pytest

from research_bot.deep_rebuild_v23 import (
    DeepRebuildConfig,
    build_causal_sequences,
    chronological_split,
    economic_bar_metrics,
    make_temporal_model,
    fit_temporal_model,
    predict_probability,
)


def make_ohlcv(n=520):
    rng = np.random.default_rng(314)
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    ret = rng.normal(0.0002, 0.008, size=n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1] * np.exp(rng.normal(0, 0.001, size=n-1))]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0002, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0002, 0.004, size=n))
    volume = rng.lognormal(10, 0.35, size=n)
    return pd.DataFrame({
        "timestamp": t,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_sequence_contract_uses_24bps_roundtrip_hurdle():
    cfg = DeepRebuildConfig(fee_bps_each_way=10, slippage_bps_each_way=2)
    assert cfg.one_way_cost == pytest.approx(0.0012)
    assert cfg.roundtrip_cost == pytest.approx(0.0024)


def test_future_mutation_does_not_change_past_windows():
    base = make_ohlcv()
    changed = base.copy()
    cutoff = 360
    changed.loc[cutoff:, ["open", "high", "low", "close", "volume"]] *= np.array([2.0, 2.0, 2.0, 2.0, 10.0])

    cfg = DeepRebuildConfig(lookback=64)
    xa, ya, ra, ta, _ = build_causal_sequences(base, cfg)
    xb, yb, rb, tb, _ = build_causal_sequences(changed, cfg)

    safe_time = base.loc[cutoff - 3, "timestamp"]
    ia = np.where(ta <= safe_time)[0]
    ib = np.where(tb <= safe_time)[0]
    assert len(ia) == len(ib) and len(ia) > 20
    np.testing.assert_allclose(xa[ia], xb[ib], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(ya[ia], yb[ib], atol=0, rtol=0)
    np.testing.assert_allclose(ra[ia], rb[ib], atol=1e-7, rtol=1e-7)


def test_chronological_sequence_split_has_no_overlap():
    x, y, r, ts, _ = build_causal_sequences(make_ohlcv(), DeepRebuildConfig())
    parts = chronological_split(x, y, r, ts)
    d = parts["development"][3]
    v = parts["validation"][3]
    t = parts["test"][3]
    assert d.max() < v.min() < t.min()
    assert v.max() < t.min()


def test_bar_economics_charge_turnover_cost():
    cfg = DeepRebuildConfig()
    p = np.array([0.9, 0.9, 0.1, 0.1], dtype=float)
    r = np.zeros(4, dtype=float)
    m = economic_bar_metrics(p, r, threshold=0.6, config=cfg)
    assert m["turnover_units"] == pytest.approx(3.0)
    assert m["total_return"] < 0.0


def test_all_temporal_model_forward_shapes():
    torch = pytest.importorskip("torch")
    cfg = DeepRebuildConfig(hidden_dim=16, layers=1, dropout=0.0)
    x = torch.randn(5, 32, 14)
    for kind in ("lstm", "gru", "cnn_lstm", "tcn", "transformer"):
        model = make_temporal_model(kind, input_dim=14, config=cfg)
        y = model(x)
        assert tuple(y.shape) == (5,)


def test_lstm_training_smoke():
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(160, 24, 8)).astype(np.float32)
    y = (x[:, -1, 0] + 0.4 * x[:, -2, 1] > 0).astype(np.float32)
    cfg = DeepRebuildConfig(
        lookback=24,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        batch_size=32,
        epochs=2,
        patience=1,
        min_active_validation=10,
    )
    model = make_temporal_model("lstm", input_dim=8, config=cfg)
    model, history = fit_temporal_model(model, x[:120], y[:120], x[120:], y[120:], cfg, seed=314)
    p = predict_probability(model, x[120:])
    assert len(history) >= 1
    assert p.shape == (40,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
