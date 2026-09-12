from __future__ import annotations

"""Frozen v0.46 economic-state representation helpers."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

R0 = "R0_FROZEN_THREE_STATE_BASELINE"
R1 = "R1_FOUR_STATE_TIMEOUT_SIGN"
V46_ARMS: tuple[str, ...] = (R0, R1)
R0_STATES: tuple[str, ...] = ("TARGET", "STOP", "TIME")
R1_STATES: tuple[str, ...] = ("TARGET", "STOP", "TIME_POSITIVE", "TIME_NONPOSITIVE")
COMMON_STATES: tuple[str, ...] = R0_STATES


@dataclass(frozen=True)
class V46ModelPolicy:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    minimum_fit_events: int = 600
    minimum_calibration_events: int = 50
    minimum_test_events: int = 1


def states_for_arm_v46(arm: str) -> tuple[str, ...]:
    if arm == R0:
        return R0_STATES
    if arm == R1:
        return R1_STATES
    raise ValueError(f"unknown v0.46 arm: {arm}")


def encode_state_v46(frame: pd.DataFrame, arm: str) -> pd.Series:
    if "outcome" not in frame or "net_r" not in frame:
        raise ValueError("v0.46 requires outcome and net_r")
    outcome = frame["outcome"].astype(str)
    if arm == R0:
        if not outcome.isin(R0_STATES).all():
            raise ValueError("unexpected frozen outcome")
        return outcome.rename("state_v46")
    if arm == R1:
        state = outcome.copy()
        nr = pd.to_numeric(frame["net_r"], errors="coerce")
        timeout = outcome.eq("TIME")
        state.loc[timeout & nr.gt(0.0)] = "TIME_POSITIVE"
        state.loc[timeout & ~nr.gt(0.0)] = "TIME_NONPOSITIVE"
        if not state.isin(R1_STATES).all():
            raise ValueError("unexpected v0.46 state")
        return state.rename("state_v46")
    raise ValueError(f"unknown v0.46 arm: {arm}")


def fit_multinomial_v46(train: pd.DataFrame, feature_cols: Iterable[str], arm: str, policy: V46ModelPolicy | None = None):
    policy = policy or V46ModelPolicy()
    cols = tuple(feature_cols)
    if len(train) < policy.minimum_fit_events:
        raise ValueError("insufficient fit events")
    y = encode_state_v46(train, arm)
    required = states_for_arm_v46(arm)
    if set(y.unique()) != set(required):
        raise ValueError("missing required training state support")
    x = train.loc[:, cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    model = LogisticRegression(C=policy.C, solver=policy.solver, max_iter=policy.max_iter)
    model.fit(x, y.to_numpy())
    means = train.assign(state_v46=y).groupby("state_v46", sort=False)["net_r"].mean().to_dict()
    return model, {str(k): float(v) for k, v in means.items()}


def predict_v46(model: LogisticRegression, frame: pd.DataFrame, feature_cols: Iterable[str], required_states: Iterable[str]) -> pd.DataFrame:
    cols = tuple(feature_cols)
    x = frame.loc[:, cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    raw = model.predict_proba(x)
    class_to_col = {str(c): i for i, c in enumerate(model.classes_)}
    out = pd.DataFrame(index=frame.index)
    for state in required_states:
        if state not in class_to_col:
            raise ValueError(f"model missing required state {state}")
        out[f"p_{state.lower()}_v46"] = raw[:, class_to_col[state]]
    return out


def common_three_state_probabilities_v46(probabilities: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = pd.DataFrame(index=probabilities.index)
    out["p_target_v46"] = probabilities["p_target_v46"].astype(float)
    out["p_stop_v46"] = probabilities["p_stop_v46"].astype(float)
    if arm == R0:
        out["p_time_v46"] = probabilities["p_time_v46"].astype(float)
    elif arm == R1:
        out["p_time_v46"] = (
            probabilities["p_time_positive_v46"].astype(float)
            + probabilities["p_time_nonpositive_v46"].astype(float)
        )
    else:
        raise ValueError(arm)
    total = out.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(total, 1.0, atol=1e-8):
        raise ValueError("common v0.46 probabilities do not sum to one")
    return out


def expected_r_from_state_probs_v46(probabilities: pd.DataFrame, state_means: dict[str, float], states: Iterable[str]) -> pd.Series:
    result = np.zeros(len(probabilities), dtype=float)
    for state in states:
        result += probabilities[f"p_{state.lower()}_v46"].to_numpy(dtype=float) * float(state_means[state])
    return pd.Series(result, index=probabilities.index, name="expected_r_v46")


def multiclass_brier_v46(y_true: pd.Series, probabilities: pd.DataFrame, states: Iterable[str]) -> float:
    states = tuple(states)
    truth = y_true.astype(str).to_numpy()
    score = np.zeros(len(truth), dtype=float)
    for state in states:
        p = probabilities[f"p_{state.lower()}_v46"].to_numpy(dtype=float)
        score += (p - (truth == state).astype(float)) ** 2
    return float(np.mean(score))


def reliability_resolution_v46(y: np.ndarray, p: np.ndarray, bins: int = 10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    base = float(np.mean(y)) if len(y) else np.nan
    reliability = 0.0
    resolution = 0.0
    rows: list[dict] = []
    n = len(y)
    for b in range(bins):
        mask = ids == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": b, "n": 0, "mean_p": None, "mean_y": None})
            continue
        mean_p = float(np.mean(p[mask]))
        mean_y = float(np.mean(y[mask]))
        w = count / n
        reliability += w * (mean_p - mean_y) ** 2
        resolution += w * (mean_y - base) ** 2
        rows.append({"bin": b, "n": count, "mean_p": mean_p, "mean_y": mean_y})
    return float(reliability), float(resolution), pd.DataFrame(rows)


def forecast_metrics_v46(y_true: pd.Series, probabilities: pd.DataFrame, states: Iterable[str]) -> dict:
    states = tuple(states)
    matrix = np.column_stack([probabilities[f"p_{s.lower()}_v46"].to_numpy(dtype=float) for s in states])
    y = y_true.astype(str).to_numpy()
    result: dict[str, float | None] = {"multiclass_brier": multiclass_brier_v46(y_true, probabilities, states)}
    try:
        result["log_loss"] = float(log_loss(y, matrix, labels=list(states)))
    except Exception:
        result["log_loss"] = None
    try:
        onehot = np.column_stack([(y == s).astype(int) for s in states])
        result["macro_ovr_auc"] = float(roc_auc_score(onehot, matrix, average="macro"))
    except Exception:
        result["macro_ovr_auc"] = None
    rels: list[float] = []
    ress: list[float] = []
    for state in states:
        target = (y == state).astype(float)
        p = probabilities[f"p_{state.lower()}_v46"].to_numpy(dtype=float)
        rel, res, _ = reliability_resolution_v46(target, p)
        rels.append(rel)
        ress.append(res)
        result[f"brier_{state.lower()}"] = float(np.mean((p - target) ** 2))
        result[f"reliability_{state.lower()}"] = rel
        result[f"resolution_{state.lower()}"] = res
    result["mean_reliability"] = float(np.mean(rels))
    result["mean_resolution"] = float(np.mean(ress))
    return result
