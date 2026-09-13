import numpy as np
import pandas as pd
import pytest

from research_bot.vision_ict_v22 import (
    CHANNEL_NAMES,
    WEAK_LABEL_NAMES,
    VisionConfig,
    build_vision_dataset,
    ict_weak_labels,
    make_vision_model,
    render_multichannel_window,
)


def synthetic(n=520, freq="15min", seed=222):
    rng = np.random.default_rng(seed)
    regime = np.sin(np.linspace(0, 12, n)) * 0.0008
    ret = regime + rng.normal(0, 0.003, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.maximum(close * 0.0008, rng.uniform(0.0004, 0.003, n) * close)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.6, n),
    })


def test_multichannel_renderer_shape_range_and_hash_determinism():
    df = synthetic()
    cfg = VisionConfig(lookback=64, height=64, width=96, min_history=100)
    a, ma = render_multichannel_window(df, 350, cfg)
    b, mb = render_multichannel_window(df, 350, cfg)
    assert a.shape == (len(CHANNEL_NAMES), 64, 96)
    assert a.dtype == np.float32
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    assert np.array_equal(a, b)
    assert ma["source_bar_sha256"] == mb["source_bar_sha256"]
    assert ma["render_sha256"] == mb["render_sha256"]
    assert ma["axes_text_timestamps_drawn"] is False


def test_renderer_is_prefix_invariant_no_future_pixels():
    df = synthetic()
    cfg = VisionConfig(lookback=64, height=64, width=96, min_history=100)
    end = 340
    full, mf = render_multichannel_window(df, end, cfg)
    prefix, mp = render_multichannel_window(df.iloc[:end + 1].copy(), end, cfg)
    assert np.array_equal(full, prefix)
    assert mf["render_sha256"] == mp["render_sha256"]
    assert mf["source_bar_sha256"] == mp["source_bar_sha256"]


def test_rule_derived_ict_labels_are_prefix_invariant():
    df = synthetic()
    full = ict_weak_labels(df)
    cut = 380
    prefix = ict_weak_labels(df.iloc[:cut].copy())
    assert list(full.columns) == ["timestamp", *WEAK_LABEL_NAMES]
    for c in WEAK_LABEL_NAMES:
        assert np.array_equal(full.loc[:cut - 1, c].to_numpy(), prefix[c].to_numpy())


def test_future_outcome_is_target_not_image_channel():
    df = synthetic()
    cfg = VisionConfig(lookback=64, height=64, width=96, min_history=100, label_horizon=2)
    x, weak, y, times, meta = build_vision_dataset(df, cfg, stride=7, max_samples=40)
    assert x.shape[0] == weak.shape[0] == y.shape[0] == len(times) == len(meta)
    assert x.shape[1] == len(CHANNEL_NAMES)
    assert weak.shape[1] == len(WEAK_LABEL_NAMES)
    assert "future_return" not in CHANNEL_NAMES
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_small_cnn_forward_contract_when_torch_available():
    torch = pytest.importorskip("torch")
    cfg = VisionConfig(lookback=64, height=64, width=96)
    model = make_vision_model("small_cnn", config=cfg)
    weak, outcome, emb = model(torch.zeros(2, len(CHANNEL_NAMES), cfg.height, cfg.width))
    assert weak.shape == (2, len(WEAK_LABEL_NAMES))
    assert outcome.shape == (2,)
    assert emb.shape[0] == 2


def test_small_vit_forward_contract_when_torch_available():
    torch = pytest.importorskip("torch")
    cfg = VisionConfig(lookback=64, height=64, width=96)
    model = make_vision_model("small_vit", config=cfg)
    weak, outcome, emb = model(torch.zeros(2, len(CHANNEL_NAMES), cfg.height, cfg.width))
    assert weak.shape == (2, len(WEAK_LABEL_NAMES))
    assert outcome.shape == (2,)
    assert emb.shape[0] == 2
