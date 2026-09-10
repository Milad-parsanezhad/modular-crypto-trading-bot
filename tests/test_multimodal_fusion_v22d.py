import numpy as np
import pandas as pd
import pytest

from research_bot.multimodal_fusion_v22d import (
    MultimodalConfig,
    RobustNumericScaler,
    build_multimodal_dataset,
    decide_multimodal,
    make_model,
)
from research_bot.scientific_liquidity_wyckoff_v22d import (
    COURSE_HYPOTHESIS_FEATURES,
    SUPPORTED_COMPONENT_FEATURES,
)


def synthetic(n=900, seed=221):
    rng = np.random.default_rng(seed)
    ret = 0.0001 + 0.001 * np.sin(np.linspace(0, 20, n)) + rng.normal(0, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * rng.uniform(0.001, 0.006, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.5, n),
    })


def test_supported_and_course_feature_tiers_are_separate():
    assert set(SUPPORTED_COMPONENT_FEATURES).isdisjoint(COURSE_HYPOTHESIS_FEATURES)


def test_dataset_shapes_and_cost_aware_target():
    cfg = MultimodalConfig(max_samples=120, min_history=220, lookback=64, height=64, width=96)
    ds = build_multimodal_dataset(synthetic(), cfg)
    assert ds["image"].shape == (120, 4, 64, 96)
    assert ds["core"].shape[0] == ds["supported"].shape[0] == ds["course"].shape[0] == 120
    assert len(ds["target"]) == len(ds["gross_return"]) == 120
    expected = (ds["gross_return"] > cfg.roundtrip_cost).astype(np.float32)
    np.testing.assert_array_equal(ds["target"], expected)


def test_future_mutation_cannot_change_earlier_sample():
    cfg = MultimodalConfig(max_samples=900, min_history=220, lookback=64, height=32, width=48)
    df = synthetic()
    cut_time = df.loc[600, "timestamp"]
    prefix = df.iloc[:603].copy()  # preserve t+1 and t+2 for the target
    base = build_multimodal_dataset(prefix, cfg)
    changed = df.copy()
    changed.loc[603:, ["open", "high", "low", "close", "volume"]] *= 9.0
    alt = build_multimodal_dataset(changed.iloc[:700].copy(), cfg)

    # Compare the latest sample in prefix with the same timestamp in the longer frame.
    t = base["timestamp"][-1]
    j = int(np.where(alt["timestamp"] == t)[0][0])
    np.testing.assert_allclose(base["image"][-1], alt["image"][j])
    np.testing.assert_allclose(base["core"][-1], alt["core"][j], equal_nan=True)
    np.testing.assert_allclose(base["supported"][-1], alt["supported"][j], equal_nan=True)
    np.testing.assert_allclose(base["course"][-1], alt["course"][j], equal_nan=True)
    assert base["target"][-1] == alt["target"][j]
    assert t <= cut_time + pd.Timedelta(hours=4)


def test_robust_scaler_is_finite_with_missing_values():
    x = np.array([[1.0, np.nan, 4.0], [2.0, np.nan, 6.0], [3.0, np.nan, 8.0]], dtype=float)
    scaler = RobustNumericScaler.fit(x)
    z = scaler.transform(x)
    assert np.isfinite(z).all()
    assert z.shape == x.shape


def _record(v_auc, t_auc, t_bacc=0.52):
    return {
        "validation": {"auc": v_auc, "balanced_accuracy": 0.52},
        "test": {"auc": t_auc, "balanced_accuracy": t_bacc},
    }


def test_multimodal_gate_pass_and_fail_closed():
    cfg = MultimodalConfig()
    passing = {
        "numeric_supported": _record(0.55, 0.54),
        "image_raw": _record(0.56, 0.545),
        "fusion_supported": _record(0.59, 0.55, 0.53),
        "fusion_supported_plus_course": _record(0.60, 0.551),
    }
    d = decide_multimodal(passing, cfg)
    assert d["passed"] is True
    assert d["vision_to_rl_state_connected"] is False
    assert d["live_execution_authorized"] is False

    failing = dict(passing)
    failing["fusion_supported"] = _record(0.565, 0.51, 0.50)
    failing["fusion_supported_plus_course"] = _record(0.566, 0.50)
    d2 = decide_multimodal(failing, cfg)
    assert d2["passed"] is False
    assert d2["decision"] == "NO_MULTIMODAL_ENCODER_PROMOTED"


def test_model_forward_shapes_when_torch_available():
    torch = pytest.importorskip("torch")
    image = torch.zeros(3, 4, 64, 96)
    numeric = torch.zeros(3, 10)
    for kind in ("numeric", "image", "fusion"):
        model = make_model(kind, numeric_dim=10)
        logits, embedding = model(image, numeric)
        assert logits.shape == (3,)
        assert embedding.shape[0] == 3
