from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.conditional_concept_drift_v49 import (
    BOOTSTRAP_REPS_V49,
    BOOTSTRAP_SEED_V49,
    COMMON_STATES_V49,
    MIN_INTERACTION_PAIRS_V49,
    calibration_map_distance_v49,
    common_brier_losses_v49,
    fit_binary_calibration_map_v49,
    probability_logit_v49,
    spearman_feature_residual_v49,
    standardized_cusum_v49,
    state_residual_v49,
)


def test_frozen_v49_constants():
    assert COMMON_STATES_V49 == ("TARGET", "STOP", "TIME")
    assert BOOTSTRAP_REPS_V49 == 3000
    assert BOOTSTRAP_SEED_V49 == 314
    assert MIN_INTERACTION_PAIRS_V49 == 30


def test_probability_logit_is_finite_at_boundaries():
    z = probability_logit_v49([0.0, 0.5, 1.0])
    assert np.isfinite(z).all()
    assert z[0] < 0 < z[2]
    assert abs(z[1]) < 1e-12


def test_calibration_map_distance_zero_for_identical_parameters():
    assert calibration_map_distance_v49((1.0, 2.0), (1.0, 2.0)) == 0.0


def test_binary_calibration_map_requires_both_classes():
    p = np.linspace(0.1, 0.9, 30)
    assert fit_binary_calibration_map_v49(np.zeros(30), p) is None
    y = np.array([0, 1] * 15)
    result = fit_binary_calibration_map_v49(y, p)
    assert result is not None
    assert np.isfinite(result).all()


def test_common_brier_loss_zero_for_perfect_predictions():
    y = ["TARGET", "STOP", "TIME"]
    probs = pd.DataFrame({
        "p_target_v47": [1.0, 0.0, 0.0],
        "p_stop_v47": [0.0, 1.0, 0.0],
        "p_time_v47": [0.0, 0.0, 1.0],
    })
    loss = common_brier_losses_v49(y, probs)
    assert np.allclose(loss, 0.0)


def test_standardized_cusum_detects_loss_level_shift():
    reference = np.tile([0.1, 0.2, 0.15, 0.18], 25)
    unchanged = np.tile([0.1, 0.2, 0.15, 0.18], 25)
    shifted = unchanged + 0.5
    a = standardized_cusum_v49(reference, unchanged)
    b = standardized_cusum_v49(reference, shifted)
    assert a is not None and b is not None
    assert b > a


def test_state_residual_sign_and_spearman():
    n = 40
    y = np.array(["TARGET"] * 20 + ["STOP"] * 20)
    p = np.linspace(0.1, 0.9, n)
    residual = state_residual_v49(y, p, "TARGET")
    assert len(residual) == n
    feature = pd.Series(np.arange(n, dtype=float))
    rho = spearman_feature_residual_v49(feature, residual)
    assert rho is not None
    assert np.isfinite(rho)


def test_spearman_returns_none_for_constant_feature():
    feature = pd.Series(np.ones(40))
    residual = np.linspace(-1.0, 1.0, 40)
    assert spearman_feature_residual_v49(feature, residual) is None
