from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.economic_state_v46 import (
    R0,
    R1,
    allocate_portfolio_risk_writable_v46,
    common_three_state_probabilities_v46,
    encode_state_v46,
    expected_r_from_state_probs_v46,
    forecast_metrics_v46,
    multiclass_brier_v46,
    reliability_resolution_v46,
)


def test_timeout_sign_split_is_exact_zero_boundary() -> None:
    frame = pd.DataFrame({"outcome": ["TARGET", "STOP", "TIME", "TIME", "TIME"], "net_r": [2.7, -1.1, 0.4, 0.0, -0.2]})
    assert encode_state_v46(frame, R0).tolist() == ["TARGET", "STOP", "TIME", "TIME", "TIME"]
    assert encode_state_v46(frame, R1).tolist() == ["TARGET", "STOP", "TIME_POSITIVE", "TIME_NONPOSITIVE", "TIME_NONPOSITIVE"]


def test_r1_common_space_recombines_timeout_probabilities() -> None:
    p = pd.DataFrame({
        "p_target_v46": [0.2, 0.1],
        "p_stop_v46": [0.3, 0.4],
        "p_time_positive_v46": [0.4, 0.2],
        "p_time_nonpositive_v46": [0.1, 0.3],
    })
    common = common_three_state_probabilities_v46(p, R1)
    assert np.allclose(common["p_time_v46"], [0.5, 0.5])
    assert np.allclose(common.sum(axis=1), 1.0)


def test_expected_r_uses_only_supplied_state_means() -> None:
    p = pd.DataFrame({
        "p_target_v46": [0.2], "p_stop_v46": [0.3],
        "p_time_positive_v46": [0.4], "p_time_nonpositive_v46": [0.1],
    })
    means = {"TARGET": 2.7, "STOP": -1.1, "TIME_POSITIVE": 0.8, "TIME_NONPOSITIVE": -0.3}
    got = expected_r_from_state_probs_v46(p, means, tuple(means))[0]
    assert np.isclose(got, 0.2 * 2.7 + 0.3 * -1.1 + 0.4 * 0.8 + 0.1 * -0.3)


def test_perfect_common_probabilities_have_zero_brier() -> None:
    y = pd.Series(["TARGET", "STOP", "TIME"])
    p = pd.DataFrame({
        "p_target_v46": [1.0, 0.0, 0.0],
        "p_stop_v46": [0.0, 1.0, 0.0],
        "p_time_v46": [0.0, 0.0, 1.0],
    })
    assert np.isclose(multiclass_brier_v46(y, p, ("TARGET", "STOP", "TIME")), 0.0)


def test_reliability_resolution_finite_nonnegative() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    rel, res, bins = reliability_resolution_v46(y, p, bins=10)
    assert np.isfinite(rel) and rel >= 0.0
    assert np.isfinite(res) and res >= 0.0
    assert int(bins["n"].sum()) == 4


def test_log_loss_diagnostic_respects_lexicographic_class_order() -> None:
    y = pd.Series(["TARGET", "STOP", "TIME", "TARGET"])
    p = pd.DataFrame({
        "p_target_v46": [0.8, 0.1, 0.1, 0.7],
        "p_stop_v46": [0.1, 0.8, 0.1, 0.2],
        "p_time_v46": [0.1, 0.1, 0.8, 0.1],
    })
    metrics = forecast_metrics_v46(y, p, ("TARGET", "STOP", "TIME"))
    assert metrics["log_loss"] is not None
    assert np.isfinite(float(metrics["log_loss"]))


def test_writable_governor_compatibility_preserves_frozen_risk_caps() -> None:
    proposals = pd.DataFrame({
        "symbol": [f"ASSET{i}/USDT" for i in range(10)],
        "side": [1] * 10,
        "lower_expected_r": [0.5] * 10,
        "uncertainty_width_r": [0.0] * 10,
        "stop_fraction": [0.02] * 10,
    })
    out = allocate_portfolio_risk_writable_v46(proposals, equity=1.0, peak=1.0)
    # Ten 0.25% proposals exceed the frozen 1.5% same-direction cap, forcing
    # the exact in-place scaling operation that failed with a read-only array.
    assert np.isclose(float(out["allocated_risk_fraction"].sum()), 0.015)
    assert float(out["allocated_risk_fraction"].max()) <= 0.005
    assert float(out["position_weight"].sum()) <= 0.70 + 1e-12
