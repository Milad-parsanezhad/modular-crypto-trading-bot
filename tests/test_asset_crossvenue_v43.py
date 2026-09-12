from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_PERTURBATION_SEEDS,
    brier_skill_v43,
    data_quality_diagnostics_v43,
    empirical_hazard_baseline_v43,
    moving_block_resample_positions_v43,
    preregistration_manifest_v43,
    reserve_unsettled_horizon_v43,
    training_support_v43,
)


def _ohlcv(n: int = 1000) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    base = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame({
        "timestamp": ts,
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.2,
        "volume": np.full(n, 10.0),
    })


def test_v43_manifest_fail_closed() -> None:
    m = preregistration_manifest_v43()
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["threshold_relaxation"] is False
    assert m["outcome_based_asset_pruning"] is False
    assert m["transformer_or_rl_added"] is False
    assert tuple(m["perturbation_seeds"]) == V43_PERTURBATION_SEEDS


def test_clean_ohlcv_passes_and_reserves_label_horizon() -> None:
    x = _ohlcv(1000)
    d = data_quality_diagnostics_v43(x)
    assert d["accepted"] is True
    assert d["reason_codes"] == ["PASS"]
    y = reserve_unsettled_horizon_v43(x)
    assert len(y) == 970
    assert y["timestamp"].max() < x["timestamp"].max()


def test_data_quality_rejects_duplicate_bad_ohlc_and_negative_volume() -> None:
    x = _ohlcv(1000)
    x.loc[10, "timestamp"] = x.loc[9, "timestamp"]
    x.loc[20, "low"] = x.loc[20, "high"] + 5.0
    x.loc[30, "volume"] = -1.0
    d = data_quality_diagnostics_v43(x)
    assert d["accepted"] is False
    assert "DUPLICATE_TIMESTAMP" in d["reason_codes"]
    assert "INVALID_OHLC_RELATION" in d["reason_codes"]
    assert "NEGATIVE_VOLUME" in d["reason_codes"]


def test_data_quality_rejects_excessive_gaps_without_using_returns() -> None:
    p = DataQualityPolicyV43(max_missing_bar_fraction=0.02)
    x = _ohlcv(1000).drop(index=np.arange(100, 150)).reset_index(drop=True)
    d = data_quality_diagnostics_v43(x, p)
    assert d["accepted"] is False
    assert "EXCESSIVE_MISSING_BARS" in d["reason_codes"]


def test_moving_block_perturbations_are_deterministic_and_different_across_seeds() -> None:
    a = moving_block_resample_positions_v43(1000, seed=314)
    b = moving_block_resample_positions_v43(1000, seed=314)
    c = moving_block_resample_positions_v43(1000, seed=1618)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 1000
    assert a.min() >= 0 and a.max() < 1000
    # The first frozen block is contiguous, preserving local dependence.
    assert np.all(np.diff(a[:64]) == 1)


def test_training_support_is_outcome_blind_and_requires_two_venues() -> None:
    p = DataQualityPolicyV43(
        minimum_asset_train_events=600,
        minimum_venue_train_events=100,
        minimum_training_venues=2,
    )
    x = pd.DataFrame({
        "venue": ["coinex"] * 350 + ["okx"] * 250,
        "outcome": ["STOP"] * 600,
    })
    a = training_support_v43(x, p)
    x["outcome"] = ["TARGET"] * 600
    b = training_support_v43(x, p)
    assert a == b
    assert a["supported"] is True

    one_venue = pd.DataFrame({"venue": ["coinex"] * 700})
    assert training_support_v43(one_venue, p)["supported"] is False


def test_empirical_baseline_never_reads_target_outcomes() -> None:
    train = pd.DataFrame({
        "event_family_v41": ["ICT_MSS"] * 120 + ["BROOKS_H2L2"] * 120,
        "side": [1] * 240,
        "regime": ["trend"] * 240,
        "outcome": ["TARGET"] * 80 + ["STOP"] * 40 + ["TARGET"] * 40 + ["STOP"] * 80,
    })
    target_a = pd.DataFrame({
        "event_family_v41": ["ICT_MSS", "BROOKS_H2L2"],
        "side": [1, 1],
        "regime": ["trend", "trend"],
        "outcome": ["TARGET", "STOP"],
    })
    target_b = target_a.copy()
    target_b["outcome"] = ["STOP", "TARGET"]
    a = empirical_hazard_baseline_v43(train, target_a)
    b = empirical_hazard_baseline_v43(train, target_b)
    np.testing.assert_allclose(a["baseline_p_target_v43"], b["baseline_p_target_v43"])
    np.testing.assert_allclose(a["baseline_p_stop_v43"], b["baseline_p_stop_v43"])


def test_brier_skill_positive_only_when_model_beats_baseline() -> None:
    y = [1, 1, 0, 0]
    model = [0.9, 0.8, 0.2, 0.1]
    baseline = [0.5, 0.5, 0.5, 0.5]
    assert brier_skill_v43(y, model, baseline) > 0
    assert brier_skill_v43(y, baseline, model) < 0
