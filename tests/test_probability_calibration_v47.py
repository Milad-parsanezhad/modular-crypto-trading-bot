from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.probability_calibration_v47 import (
    C0,
    C1,
    C2,
    R1_STATES,
    V47CalibrationPolicy,
    apply_dirichlet_v47,
    calibrate_v47,
    common_three_state_probabilities_v47,
    fit_dirichlet_v47,
    fit_temperature_v47,
    multiclass_nll_v47,
    probability_frame_v47,
    temperature_apply_v47,
)


def _synthetic_calibration_set():
    labels = np.array(list(R1_STATES) * 30, dtype=str)
    mapping = {s: i for i, s in enumerate(R1_STATES)}
    probs = np.full((len(labels), 4), 0.10, dtype=float)
    for i, label in enumerate(labels):
        probs[i, mapping[label]] = 0.70
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


def test_identity_arm_is_exact_copy() -> None:
    cal, y = _synthetic_calibration_set()
    test = cal[:12].copy()
    out, diag = calibrate_v47(C0, cal, y, test)
    assert np.array_equal(out, test)
    assert diag["method"] == "identity"


def test_temperature_probabilities_sum_to_one_and_fit_does_not_worsen_calibration_nll() -> None:
    cal, y = _synthetic_calibration_set()
    # Make the base probabilities artificially overconfident while preserving ranking.
    logits = np.log(cal)
    overconfident = np.exp(logits / 0.35)
    overconfident /= overconfident.sum(axis=1, keepdims=True)
    diagnostics = fit_temperature_v47(overconfident, y)
    scaled = temperature_apply_v47(overconfident, diagnostics["temperature"])
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert diagnostics["calibration_nll_after"] <= diagnostics["calibration_nll_before"] + 1e-10
    assert multiclass_nll_v47(y, scaled) <= multiclass_nll_v47(y, overconfident) + 1e-10


def test_temperature_is_fit_only_from_supplied_calibration_arrays() -> None:
    cal, y = _synthetic_calibration_set()
    p = V47CalibrationPolicy()
    diagnostics = fit_temperature_v47(cal, y, p)
    assert np.exp(p.temperature_log_min) <= diagnostics["temperature"] <= np.exp(p.temperature_log_max)


def test_dirichlet_preserves_all_four_classes_and_simplex() -> None:
    cal, y = _synthetic_calibration_set()
    model = fit_dirichlet_v47(cal, y)
    out = apply_dirichlet_v47(model, cal[:20])
    assert out.shape == (20, 4)
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert set(map(str, model.classes_)) == set(R1_STATES)


def test_calibrate_dirichlet_returns_finite_diagnostics() -> None:
    cal, y = _synthetic_calibration_set()
    out, diag = calibrate_v47(C2, cal, y, cal[:16])
    assert np.isfinite(out).all()
    assert diag["method"] == "dirichlet"
    assert np.isfinite(diag["calibration_nll_before"])
    assert np.isfinite(diag["calibration_nll_after"])


def test_common_space_recombines_timeout_states_exactly() -> None:
    native = probability_frame_v47(
        np.array([
            [0.2, 0.3, 0.4, 0.1],
            [0.1, 0.4, 0.2, 0.3],
        ], dtype=float),
        R1_STATES,
    )
    common = common_three_state_probabilities_v47(native)
    assert np.allclose(common["p_time_v47"].to_numpy(), [0.5, 0.5])
    assert np.allclose(common.sum(axis=1).to_numpy(), 1.0)


def test_temperature_arm_produces_valid_simplex() -> None:
    cal, y = _synthetic_calibration_set()
    out, diag = calibrate_v47(C1, cal, y, cal[:10])
    assert diag["method"] == "temperature"
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_probability_frame_uses_frozen_state_order() -> None:
    matrix = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)
    frame = probability_frame_v47(matrix, R1_STATES)
    assert list(frame.columns) == [
        "p_target_v47",
        "p_stop_v47",
        "p_time_positive_v47",
        "p_time_nonpositive_v47",
    ]
    assert np.isclose(float(frame.iloc[0].sum()), 1.0)
