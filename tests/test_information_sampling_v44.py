from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_bot.information_sampling_v44 import (
    V44_VARIANTS,
    build_sampling_flags_v44,
    cusum_information_events_v44,
    directional_change_events_v44,
    lagged_atr_scale_v44,
    preregistration_manifest_v44,
    variant_column_v44,
)


def bars(n: int = 500) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.0004 * t + 0.012 * np.sin(t / 9.0))
    open_ = np.r_[close[0], close[:-1]]
    span = close * (0.006 + 0.001 * np.cos(t / 13.0))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span,
        "close": close,
        "volume": 1_000_000.0 + 100_000.0 * (1 + np.sin(t / 17.0)),
        "mother_event_v39": (np.arange(n) % 7 == 0).astype(int),
    })
    return frame


def test_manifest_is_fail_closed_and_has_exactly_three_variants():
    m = preregistration_manifest_v44()
    assert tuple(m["variants_in_frozen_order"]) == V44_VARIANTS
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["threshold_relaxation"] is False
    assert m["outcome_based_variant_tuning"] is False
    assert m["true_tick_volume_dollar_bars_claimed"] is False


def test_sampling_flags_do_not_change_before_future_perturbation():
    original = bars()
    changed = original.copy()
    cut = 370
    changed.loc[cut:, "close"] *= np.linspace(1.0, 2.5, len(changed) - cut)
    changed.loc[cut:, "open"] = changed["close"].iloc[cut - 1:-1].to_numpy()
    changed.loc[cut:, "high"] = np.maximum(changed.loc[cut:, "open"], changed.loc[cut:, "close"]) * 1.01
    changed.loc[cut:, "low"] = np.minimum(changed.loc[cut:, "open"], changed.loc[cut:, "close"]) * 0.99

    a = build_sampling_flags_v44(original)
    b = build_sampling_flags_v44(changed)
    pd.testing.assert_frame_equal(a.iloc[:cut], b.iloc[:cut])


def test_current_bar_shock_does_not_change_its_own_lagged_scale():
    a = bars(250)
    b = a.copy()
    i = 200
    b.loc[i, "high"] *= 4.0
    b.loc[i, "close"] *= 2.0
    b.loc[i, "open"] = min(b.loc[i, "open"], b.loc[i, "close"])
    b.loc[i, "low"] = min(b.loc[i, "low"], b.loc[i, "open"])
    # Scale at i is based on ATR/close through i-1 only.
    sa = lagged_atr_scale_v44(a)
    sb = lagged_atr_scale_v44(b)
    assert np.isclose(sa.iloc[i], sb.iloc[i], equal_nan=True)


def test_cusum_and_directional_change_are_discrete_and_causal():
    x = bars(400)
    c = cusum_information_events_v44(x)
    d = directional_change_events_v44(x)
    assert set(c.astype(int).unique()) <= {0, 1}
    assert set(d["v44_dc_event"].astype(int).unique()) <= {0, 1}
    assert set(d["v44_dc_direction"].unique()) <= {-1, 0, 1}
    assert ((d.loc[d["v44_dc_event"], "v44_dc_direction"].abs()) == 1).all()


def test_variant_columns_are_frozen_and_s1_s2_are_subsets_of_mother_events():
    x = bars(450)
    f = build_sampling_flags_v44(x)
    assert variant_column_v44(V44_VARIANTS[0]) == "v44_sample_s0"
    assert variant_column_v44(V44_VARIANTS[1]) == "v44_sample_s1"
    assert variant_column_v44(V44_VARIANTS[2]) == "v44_sample_s2"
    assert (f["v44_sample_s1"] <= f["v44_sample_s0"]).all()
    assert (f["v44_sample_s2"] <= f["v44_sample_s0"]).all()


def test_sampler_rejects_outcome_and_economic_columns():
    x = bars(100)
    x["net_r"] = 0.0
    with pytest.raises(ValueError, match="forbidden outcome/economic"):
        build_sampling_flags_v44(x)


def test_duplicate_timestamp_fails_closed():
    x = bars(100)
    x.loc[10, "timestamp"] = x.loc[9, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        build_sampling_flags_v44(x)
