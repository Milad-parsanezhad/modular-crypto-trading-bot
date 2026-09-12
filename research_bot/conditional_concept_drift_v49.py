from __future__ import annotations

"""Frozen v0.49 conditional concept-drift diagnostic primitives.

Diagnostic only. This module does not alter trading policy, labels, features,
costs, portfolio logic, asset selection, or external-data access.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

COMMON_STATES_V49: tuple[str, ...] = ("TARGET", "STOP", "TIME")
PROBABILITY_EPS_V49 = 1e-6
MIN_INTERACTION_ROWS_V49 = 20
MIN_INTERACTION_PAIRS_V49 = 30
BH_Q_V49 = 0.10
BOOTSTRAP_REPS_V49 = 3000
BOOTSTRAP_SEED_V49 = 314


@dataclass(frozen=True)
class CalibrationMapPolicyV49:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    epsilon: float = PROBABILITY_EPS_V49


def probability_logit_v49(probability: Iterable[float], epsilon: float = PROBABILITY_EPS_V49) -> np.ndarray:
    p = np.asarray(list(probability), dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("non-finite probability")
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def fit_binary_calibration_map_v49(
    y: Iterable[int | float],
    probability: Iterable[float],
    policy: CalibrationMapPolicyV49 | None = None,
) -> tuple[float, float] | None:
    policy = policy or CalibrationMapPolicyV49()
    yy = np.asarray(list(y), dtype=int)
    pp = np.asarray(list(probability), dtype=float)
    if len(yy) != len(pp) or len(yy) < 2:
        return None
    if set(np.unique(yy)) != {0, 1}:
        return None
    x = probability_logit_v49(pp, policy.epsilon).reshape(-1, 1)
    model = LogisticRegression(C=policy.C, solver=policy.solver, max_iter=policy.max_iter)
    model.fit(x, yy)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_map_distance_v49(a: tuple[float, float], b: tuple[float, float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != (2,) or bb.shape != (2,) or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError("invalid calibration-map parameters")
    return float(np.linalg.norm(aa - bb))


def common_brier_losses_v49(y_true: Iterable[str], probabilities: pd.DataFrame) -> np.ndarray:
    y = np.asarray(list(y_true), dtype=str)
    cols = [f"p_{state.lower()}_v47" for state in COMMON_STATES_V49]
    missing = [c for c in cols if c not in probabilities.columns]
    if missing:
        raise ValueError(f"missing common probability columns: {missing}")
    p = probabilities.loc[:, cols].to_numpy(dtype=float)
    if len(y) != len(p):
        raise ValueError("label/probability length mismatch")
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("invalid common probabilities")
    onehot = np.column_stack([(y == state).astype(float) for state in COMMON_STATES_V49])
    return np.sum((p - onehot) ** 2, axis=1)


def standardized_cusum_v49(reference_losses: Iterable[float], evaluation_losses: Iterable[float]) -> float | None:
    a = np.asarray(list(reference_losses), dtype=float)
    b = np.asarray(list(evaluation_losses), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 1:
        return None
    mu = float(np.mean(a))
    sigma = float(np.std(a, ddof=1))
    sigma = max(sigma, 1e-8)
    z = (b - mu) / sigma
    cumulative = np.cumsum(z)
    return float(np.max(np.abs(cumulative)) / np.sqrt(len(b)))


def spearman_feature_residual_v49(feature: pd.Series, residual: Iterable[float]) -> float | None:
    x = pd.to_numeric(feature, errors="coerce").to_numpy(dtype=float)
    r = np.asarray(list(residual), dtype=float)
    if len(x) != len(r):
        raise ValueError("feature/residual length mismatch")
    mask = np.isfinite(x) & np.isfinite(r)
    x = x[mask]
    r = r[mask]
    if len(x) < MIN_INTERACTION_ROWS_V49:
        return None
    if np.unique(x).size < 2 or np.unique(r).size < 2:
        return None
    rho = float(spearmanr(x, r).statistic)
    return rho if np.isfinite(rho) else None


def state_residual_v49(y_true: Iterable[str], probability: Iterable[float], state: str) -> np.ndarray:
    y = np.asarray(list(y_true), dtype=str)
    p = np.asarray(list(probability), dtype=float)
    if len(y) != len(p):
        raise ValueError("label/probability length mismatch")
    return (y == state).astype(float) - p
