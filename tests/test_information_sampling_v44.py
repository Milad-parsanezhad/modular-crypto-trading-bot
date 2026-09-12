from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.information_sampling_v44 import (
    CusumSamplingPolicyV44,
    attach_cusum_activity_v44,
    causal_cusum_activity_v44,
    cusum_candidate_mask_v44,
    preregistration_manifest_v44,
    reserve_complete_label_horizon_v44,
)


def _frame(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    # Deterministic non-constant path with bursts large enough to trigger CUSUM.
    ret = 0.001 * np.sin(np.arange(n) / 5.0)
    ret[80:85] += 0.015
    ret[150:154] -= 0.018
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"timestamp": ts, "close": close})


def test_v44_manifest_is_fail_closed() -> None:
    m = preregistration_manifest_v44()
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["threshold_search"] is False
    assert m["tick_bar_reconstruction"] is False
    assert m["empirical_execution_allowed"] is False


def test_cusum_prefix_is_invariant_to_future_price_perturbation() -> None:
    x = _frame(240)
    a = causal_cusum_activity_v44(x)
    y = x.copy()
    cutoff = 160
    y.loc[cutoff + 1 :, "close"] *= np.linspace(1.0, 2.0, len(y) - cutoff - 1)
    b = causal_cusum_activity_v44(y)
    cols = ["prior_sigma_v44", "cusum_threshold_v44", "cusum_trigger_v44", "cusum_activity_sign_v44"]
    for col in cols:
        av = a.loc[:cutoff, col].to_numpy()
        bv = b.loc[:cutoff, col].to_numpy()
        np.testing.assert_allclose(av, bv, equal_nan=True)


def test_threshold_is_prior_only_not_current_return() -> None:
    x = _frame(240)
    p = CusumSamplingPolicyV44(volatility_span_bars=48, minimum_volatility_bars=48)
    a = causal_cusum_activity_v44(x, p)
    t = 120
    y = x.copy()
    # Change close at t, which changes r_t; the threshold at t must remain based
    # on returns through t-1. Later thresholds may change and are not asserted.
    y.loc[t, "close"] *= 1.25
    b = causal_cusum_activity_v44(y, p)
    assert np.isclose(a.loc[t, "cusum_threshold_v44"], b.loc[t, "cusum_threshold_v44"], equal_nan=True)


def test_cusum_sign_does_not_choose_trade_direction() -> None:
    x = _frame(240)
    f = x.copy()
    f["research_candidate_side_v39"] = -1
    f = attach_cusum_activity_v44(f)
    mask = cusum_candidate_mask_v44(f)
    assert mask.any()
    # At least one positive CUSUM activity trigger can still carry the frozen
    # mother-strategy short side; the activity sign is not the trade signal.
    positive_activity = mask & f["cusum_activity_sign_v44"].eq(1)
    assert positive_activity.any()
    assert (f.loc[positive_activity, "research_candidate_side_v39"] == -1).all()


def test_reserve_horizon_uses_actual_bar_count() -> None:
    x = _frame(100)
    # Drop a candle to prove the helper reserves 30 observed bars, not 120 hours.
    x = x.drop(index=70).reset_index(drop=True)
    y = reserve_complete_label_horizon_v44(x, CusumSamplingPolicyV44(label_horizon_bars=30))
    assert len(y) == len(x) - 30
    assert y["timestamp"].iloc[-1] == x["timestamp"].iloc[-31]
