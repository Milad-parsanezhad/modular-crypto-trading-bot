from __future__ import annotations

"""Frozen v0.46 economic-state representation helpers.

The v0.46-specific portfolio allocator below is a numerical-compatibility copy
of the frozen v0.39 allocator. Its only semantic-neutral change is requesting a
writable NumPy copy before in-place cap scaling, which is required by pandas 3
copy-on-write behaviour. Risk formulas and caps remain unchanged.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from research_bot.financial_system_v39 import (
    FinancialRiskPolicyV39,
    drawdown_risk_scale,
    proposed_trade_risk_fraction,
)

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
        # sklearn requires labels in lexicographic order. Reorder the probability
        # columns to match so this diagnostic does not silently mislabel classes.
        log_states = tuple(sorted(states))
        log_matrix = np.column_stack([
            probabilities[f"p_{s.lower()}_v46"].to_numpy(dtype=float) for s in log_states
        ])
        result["log_loss"] = float(log_loss(y, log_matrix, labels=list(log_states)))
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


def allocate_portfolio_risk_writable_v46(
    proposals: pd.DataFrame,
    *,
    equity: float,
    peak: float,
    open_total_risk_fraction: float = 0.0,
    open_long_risk_fraction: float = 0.0,
    open_short_risk_fraction: float = 0.0,
    policy: FinancialRiskPolicyV39 | None = None,
) -> pd.DataFrame:
    """Numerically identical v0.39 allocation with a writable ndarray copy.

    pandas 3 may expose ``Series.to_numpy()`` as read-only. The frozen v0.39
    allocator scales that array in place. Requesting ``copy=True`` fixes only the
    memory mutability contract; formulas, ordering and all risk caps are unchanged.
    """

    p = policy or FinancialRiskPolicyV39()
    required = {"symbol", "side", "lower_expected_r", "uncertainty_width_r", "stop_fraction"}
    missing = required - set(proposals.columns)
    if missing:
        raise ValueError(f"missing risk proposal columns: {sorted(missing)}")

    x = proposals.copy().reset_index(drop=True)
    if x.empty:
        x["allocated_risk_fraction"] = pd.Series(dtype=float)
        x["position_weight"] = pd.Series(dtype=float)
        return x

    x["side"] = pd.to_numeric(x["side"], errors="raise").astype(int)
    if bool((~x["side"].isin([-1, 1])).any()):
        raise ValueError("side must be -1 or +1")

    x["proposed_risk_fraction"] = [
        proposed_trade_risk_fraction(
            lower_expected_r=float(row.lower_expected_r),
            uncertainty_width_r=float(row.uncertainty_width_r),
            equity=equity,
            peak=peak,
            policy=p,
        )
        for row in x.itertuples(index=False)
    ]

    allocated = x["proposed_risk_fraction"].to_numpy(dtype=float, copy=True)
    for side, open_risk in ((1, open_long_risk_fraction), (-1, open_short_risk_fraction)):
        mask = x["side"].to_numpy() == side
        wanted = float(allocated[mask].sum())
        available = max(0.0, p.directional_open_risk_cap - float(open_risk))
        if wanted > available and wanted > 0:
            allocated[mask] *= available / wanted

    aggregate_available = max(0.0, p.aggregate_open_risk_cap - float(open_total_risk_fraction))
    dd_scale = drawdown_risk_scale(equity, peak, p)
    if dd_scale <= 0:
        aggregate_available = 0.0
    if peak > 0 and equity > 0:
        floor_equity = peak * (1.0 - p.hard_drawdown_cap)
        headroom_fraction = max(0.0, (equity - floor_equity) / equity)
        aggregate_available = min(aggregate_available, headroom_fraction)

    wanted_total = float(allocated.sum())
    if wanted_total > aggregate_available and wanted_total > 0:
        allocated *= aggregate_available / wanted_total

    x["allocated_risk_fraction"] = np.clip(allocated, 0.0, p.max_risk_per_trade)
    stop = pd.to_numeric(x["stop_fraction"], errors="coerce").replace(0.0, np.nan)
    nominal_weight = x["allocated_risk_fraction"] / stop.abs()
    x["position_weight"] = nominal_weight.clip(lower=0.0, upper=p.max_asset_weight).fillna(0.0)

    gross = float(x["position_weight"].sum())
    if gross > p.max_portfolio_gross and gross > 0:
        x["position_weight"] *= p.max_portfolio_gross / gross

    return x
