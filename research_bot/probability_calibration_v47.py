from __future__ import annotations

"""Frozen helpers for the v0.47 post-hoc probability-calibration ablation."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from research_bot.economic_state_v46 import reliability_resolution_v46

C0 = "C0_IDENTITY_R1"
C1 = "C1_TEMPERATURE_R1"
C2 = "C2_DIRICHLET_R1"
V47_ARMS: tuple[str, ...] = (C0, C1, C2)
R1_STATES: tuple[str, ...] = ("TARGET", "STOP", "TIME_POSITIVE", "TIME_NONPOSITIVE")
COMMON_STATES: tuple[str, ...] = ("TARGET", "STOP", "TIME")


@dataclass(frozen=True)
class V47CalibrationPolicy:
    probability_epsilon: float = 1e-6
    temperature_log_min: float = -3.0
    temperature_log_max: float = 3.0
    dirichlet_C: float = 1.0
    dirichlet_solver: str = "lbfgs"
    dirichlet_max_iter: int = 2000
    minimum_fit_events: int = 600
    minimum_calibration_events: int = 50
    minimum_test_events: int = 1


def probability_matrix_v47(frame: pd.DataFrame, states: Iterable[str], suffix: str = "v46") -> np.ndarray:
    states = tuple(states)
    matrix = np.column_stack([
        pd.to_numeric(frame[f"p_{state.lower()}_{suffix}"], errors="raise").to_numpy(dtype=float)
        for state in states
    ])
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite probabilities")
    if (matrix < 0.0).any():
        raise ValueError("negative probability")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probabilities do not sum to one")
    return matrix


def probability_frame_v47(matrix: np.ndarray, states: Iterable[str]) -> pd.DataFrame:
    states = tuple(states)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(states):
        raise ValueError("invalid probability matrix shape")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("invalid probability matrix")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probabilities do not sum to one")
    return pd.DataFrame({f"p_{state.lower()}_v47": matrix[:, i] for i, state in enumerate(states)})


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    denom = exp.sum(axis=1, keepdims=True)
    if (denom <= 0.0).any() or not np.isfinite(denom).all():
        raise ValueError("invalid softmax denominator")
    return exp / denom


def temperature_apply_v47(base_probabilities: np.ndarray, temperature: float, policy: V47CalibrationPolicy | None = None) -> np.ndarray:
    p = policy or V47CalibrationPolicy()
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    base = np.clip(np.asarray(base_probabilities, dtype=float), p.probability_epsilon, 1.0)
    base = base / base.sum(axis=1, keepdims=True)
    return _softmax(np.log(base) / float(temperature))


def _class_indices(y: Iterable[str], states: tuple[str, ...]) -> np.ndarray:
    mapping = {state: i for i, state in enumerate(states)}
    labels = np.asarray(list(y), dtype=str)
    if any(label not in mapping for label in labels):
        raise ValueError("unexpected calibration state")
    return np.asarray([mapping[label] for label in labels], dtype=int)


def multiclass_nll_v47(y: Iterable[str], probabilities: np.ndarray, states: tuple[str, ...] = R1_STATES, epsilon: float = 1e-12) -> float:
    probs = np.asarray(probabilities, dtype=float)
    idx = _class_indices(y, states)
    if len(idx) != len(probs):
        raise ValueError("label/probability length mismatch")
    chosen = np.clip(probs[np.arange(len(idx)), idx], epsilon, 1.0)
    return float(-np.mean(np.log(chosen)))


def fit_temperature_v47(calibration_probabilities: np.ndarray, calibration_y: Iterable[str], policy: V47CalibrationPolicy | None = None) -> dict:
    p = policy or V47CalibrationPolicy()
    base = np.asarray(calibration_probabilities, dtype=float)
    labels = np.asarray(list(calibration_y), dtype=str)
    if len(base) < p.minimum_calibration_events:
        raise ValueError("insufficient calibration events")
    if len(np.unique(labels)) < 2:
        raise ValueError("temperature calibration requires at least two calibration states")

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        scaled = temperature_apply_v47(base, t, p)
        return multiclass_nll_v47(labels, scaled, R1_STATES)

    result = minimize_scalar(
        objective,
        bounds=(p.temperature_log_min, p.temperature_log_max),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not bool(result.success) or not np.isfinite(result.fun):
        raise RuntimeError("temperature optimization failed")
    temperature = float(np.exp(float(result.x)))
    return {
        "temperature": temperature,
        "calibration_nll_before": multiclass_nll_v47(labels, base, R1_STATES),
        "calibration_nll_after": float(result.fun),
    }


def fit_dirichlet_v47(calibration_probabilities: np.ndarray, calibration_y: Iterable[str], policy: V47CalibrationPolicy | None = None):
    p = policy or V47CalibrationPolicy()
    base = np.asarray(calibration_probabilities, dtype=float)
    labels = np.asarray(list(calibration_y), dtype=str)
    if len(base) < p.minimum_calibration_events:
        raise ValueError("insufficient calibration events")
    if set(np.unique(labels)) != set(R1_STATES):
        raise ValueError("Dirichlet calibration requires all four calibration states")
    x = np.log(np.clip(base, p.probability_epsilon, 1.0))
    model = LogisticRegression(
        C=p.dirichlet_C,
        solver=p.dirichlet_solver,
        max_iter=p.dirichlet_max_iter,
    )
    model.fit(x, labels)
    return model


def apply_dirichlet_v47(model: LogisticRegression, probabilities: np.ndarray, policy: V47CalibrationPolicy | None = None) -> np.ndarray:
    p = policy or V47CalibrationPolicy()
    base = np.asarray(probabilities, dtype=float)
    x = np.log(np.clip(base, p.probability_epsilon, 1.0))
    raw = model.predict_proba(x)
    class_to_col = {str(c): i for i, c in enumerate(model.classes_)}
    if set(class_to_col) != set(R1_STATES):
        raise ValueError("Dirichlet model class support mismatch")
    out = np.column_stack([raw[:, class_to_col[state]] for state in R1_STATES])
    if not np.allclose(out.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Dirichlet probabilities do not sum to one")
    return out


def calibrate_v47(
    arm: str,
    calibration_probabilities: np.ndarray,
    calibration_y: Iterable[str],
    test_probabilities: np.ndarray,
    policy: V47CalibrationPolicy | None = None,
):
    p = policy or V47CalibrationPolicy()
    cal = np.asarray(calibration_probabilities, dtype=float)
    test = np.asarray(test_probabilities, dtype=float)
    labels = np.asarray(list(calibration_y), dtype=str)
    if arm == C0:
        return test.copy(), {"method": "identity"}
    if arm == C1:
        diagnostics = fit_temperature_v47(cal, labels, p)
        return temperature_apply_v47(test, diagnostics["temperature"], p), {"method": "temperature", **diagnostics}
    if arm == C2:
        model = fit_dirichlet_v47(cal, labels, p)
        out = apply_dirichlet_v47(model, test, p)
        before = multiclass_nll_v47(labels, cal, R1_STATES)
        cal_after = apply_dirichlet_v47(model, cal, p)
        after = multiclass_nll_v47(labels, cal_after, R1_STATES)
        return out, {
            "method": "dirichlet",
            "calibration_nll_before": before,
            "calibration_nll_after": after,
            "coef_l2": float(np.linalg.norm(model.coef_)),
            "intercept_l2": float(np.linalg.norm(model.intercept_)),
        }
    raise ValueError(f"unknown v0.47 arm: {arm}")


def common_three_state_probabilities_v47(native: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=native.index)
    out["p_target_v47"] = native["p_target_v47"].astype(float)
    out["p_stop_v47"] = native["p_stop_v47"].astype(float)
    out["p_time_v47"] = native["p_time_positive_v47"].astype(float) + native["p_time_nonpositive_v47"].astype(float)
    if not np.allclose(out.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError("common probabilities do not sum to one")
    return out


def expected_r_v47(native: pd.DataFrame, state_means: dict[str, float]) -> pd.Series:
    result = np.zeros(len(native), dtype=float)
    for state in R1_STATES:
        result += native[f"p_{state.lower()}_v47"].to_numpy(dtype=float) * float(state_means[state])
    return pd.Series(result, index=native.index, name="expected_r_v47")


def forecast_metrics_v47(y_true: Iterable[str], probabilities: pd.DataFrame, states: tuple[str, ...]) -> dict:
    y = np.asarray(list(y_true), dtype=str)
    matrix = np.column_stack([probabilities[f"p_{state.lower()}_v47"].to_numpy(dtype=float) for state in states])
    onehot = np.column_stack([(y == state).astype(float) for state in states])
    brier = float(np.mean(np.sum((matrix - onehot) ** 2, axis=1)))
    result: dict[str, float | None] = {"multiclass_brier": brier}
    try:
        ordered = tuple(sorted(states))
        ordered_matrix = np.column_stack([probabilities[f"p_{state.lower()}_v47"].to_numpy(dtype=float) for state in ordered])
        result["log_loss"] = float(log_loss(y, ordered_matrix, labels=list(ordered)))
    except Exception:
        result["log_loss"] = None
    try:
        result["macro_ovr_auc"] = float(roc_auc_score(onehot, matrix, average="macro"))
    except Exception:
        result["macro_ovr_auc"] = None
    rels: list[float] = []
    ress: list[float] = []
    for state in states:
        target = (y == state).astype(float)
        prob = probabilities[f"p_{state.lower()}_v47"].to_numpy(dtype=float)
        rel, res, _ = reliability_resolution_v46(target, prob, bins=10)
        rels.append(rel)
        ress.append(res)
        result[f"brier_{state.lower()}"] = float(np.mean((prob - target) ** 2))
        result[f"reliability_{state.lower()}"] = rel
        result[f"resolution_{state.lower()}"] = res
    result["mean_reliability"] = float(np.mean(rels))
    result["mean_resolution"] = float(np.mean(ress))
    return result
