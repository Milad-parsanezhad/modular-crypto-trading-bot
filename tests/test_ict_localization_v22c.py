import numpy as np
import pandas as pd
import pytest

from research_bot.ict_localization_v22c import (
    LOCALIZATION_LABELS,
    RAW_INPUT_CHANNELS,
    LocalizationConfig,
    build_localization_dataset,
    ict_localization_masks,
    localization_metrics,
    localization_pos_weights,
    make_localizer,
    raw_candle_tensor,
)
from research_bot.vision_ict_v22 import CHANNEL_NAMES


def synthetic(n=520, freq="15min", seed=223):
    rng = np.random.default_rng(seed)
    wave = np.sin(np.linspace(0, 18, n)) * 0.0012
    ret = wave + rng.normal(0, 0.003, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.001, rng.uniform(0.0004, 0.003, n) * close)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    # Inject deterministic gaps/sweeps so spatial targets have support.
    for i in range(80, n, 70):
        close[i] = close[i - 1] * 1.025
        open_[i] = close[i - 1] * 1.018
        high[i] = max(open_[i], close[i]) * 1.003
        low[i] = min(open_[i], close[i]) * 0.998
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.6, n),
    })


def test_raw_localizer_input_is_exactly_candles_wicks_volume():
    df = synthetic()
    cfg = LocalizationConfig(lookback=64, height=64, width=96, min_history=100)
    raw, meta = raw_candle_tensor(df, 350, cfg)
    assert raw.shape == (4, 64, 96)
    assert [CHANNEL_NAMES[i] for i in RAW_INPUT_CHANNELS] == ["bull_body", "bear_body", "wick", "volume"]
    assert meta["engineered_ict_channels_in_input"] is False


def test_spatial_masks_are_prefix_invariant():
    df = synthetic()
    cfg = LocalizationConfig(lookback=64, height=64, width=96, min_history=100)
    end = 350
    full, mf = ict_localization_masks(df, end, cfg)
    prefix, mp = ict_localization_masks(df.iloc[:end + 1].copy(), end, cfg)
    assert full.shape == (len(LOCALIZATION_LABELS), 64, 96)
    assert np.array_equal(full, prefix)
    assert mf["positive_pixels"] == mp["positive_pixels"]


def test_future_mutation_cannot_change_input_or_mask_at_signal_time():
    df = synthetic()
    cfg = LocalizationConfig(lookback=64, height=64, width=96, min_history=100)
    end = 340
    raw1, _ = raw_candle_tensor(df, end, cfg)
    mask1, _ = ict_localization_masks(df, end, cfg)
    changed = df.copy()
    changed.loc[end + 1:, ["open", "high", "low", "close"]] *= 7.0
    raw2, _ = raw_candle_tensor(changed, end, cfg)
    mask2, _ = ict_localization_masks(changed, end, cfg)
    assert np.array_equal(raw1, raw2)
    assert np.array_equal(mask1, mask2)


def test_localization_dataset_alignment():
    df = synthetic()
    cfg = LocalizationConfig(lookback=64, height=64, width=96, min_history=100)
    x, y, t, meta = build_localization_dataset(df, cfg, stride=9, max_samples=30)
    assert x.shape == (30, 4, 64, 96)
    assert y.shape == (30, len(LOCALIZATION_LABELS), 64, 96)
    assert len(t) == len(meta) == 30
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_pixel_positive_weights_are_finite_and_damped():
    rng = np.random.default_rng(5)
    y = np.zeros((12, len(LOCALIZATION_LABELS), 16, 24), dtype=np.float32)
    y[:, 0, 2:8, 3:10] = 1
    y[:2, 1, 4, 5] = 1
    w = localization_pos_weights(y, max_weight=17)
    assert np.isfinite(w).all()
    assert (w >= 1).all()
    assert w.max() <= 17
    assert w[1] > w[0]


def test_spatial_metric_perfect_prediction_is_one_for_supported_labels():
    y = np.zeros((3, len(LOCALIZATION_LABELS), 16, 24), dtype=np.float32)
    y[:, 0, 2:8, 4:10] = 1
    m = localization_metrics(y, y, LocalizationConfig(min_supported_positive_pixels=10))
    assert m["supported_labels"] == 1
    assert m["supported_macro_dice"] == pytest.approx(1.0)
    assert m["supported_macro_iou"] == pytest.approx(1.0)


def test_small_unet_forward_shape_when_torch_available():
    torch = pytest.importorskip("torch")
    model = make_localizer("small_unet", in_channels=4)
    out = model(torch.zeros(2, 4, 64, 96))
    assert out.shape == (2, len(LOCALIZATION_LABELS), 64, 96)
