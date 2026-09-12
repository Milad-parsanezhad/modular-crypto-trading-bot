from __future__ import annotations

"""Frozen primitives for v0.50 expected-R / utility-mapping diagnostics.

Diagnostic only. No trading policy, model, threshold, state definition, cost,
non-overlap rule, or Financial Governor rule is changed here.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research_bot.temporal_regime_diagnostic_v48 import bh_adjust, one_sided_sign_p

C1 = "C1_TEMPERATURE_R1"
R1_STATES: tuple[str, ...] = ("TARGET", "STOP", "TIME_POSITIVE", "TIME_NONPOSITIVE")
BH_Q_V50 = 0.10
MIN_STATE_COUNT_V50 = 5
MIN_RANKING_ROWS_V50 = 30
MIN_SUPPORTED_UNITS_V50 = 108


@dataclass(frozen=True)
class RankingDiagnosticV50:
    n: int
    spearman_rho: float | None
    top_bottom_spread: float | None
    ols_intercept: float | None
    ols_slope: float | None
    mean_bias: float | None


def profit_factor_v50(values: Iterable[float]) -> float | None:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size == 0:
        return None
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else None
    return gains / losses


def state_transport_metrics_v50(
    fit_means: dict[str, float],
    cal_means: dict[str, float],
    test_means: dict[str, float],
    eligible_states: Iterable[str],
) -> dict:
    states = tuple(str(s) for s in eligible_states)
    if len(states) < 3:
        return {"supported": False, "eligible_states": len(states), "cal_state_mae": None, "test_state_mae": None, "state_transport_excess": None}
    cal_err = np.asarray([abs(float(fit_means[s]) - float(cal_means[s])) for s in states], dtype=float)
    test_err = np.asarray([abs(float(fit_means[s]) - float(test_means[s])) for s in states], dtype=float)
    return {
        "supported": True,
        "eligible_states": len(states),
        "cal_state_mae": float(np.mean(cal_err)),
        "test_state_mae": float(np.mean(test_err)),
        "state_transport_excess": float(np.mean(test_err) - np.mean(cal_err)),
    }


def ranking_diagnostic_v50(frame: pd.DataFrame) -> RankingDiagnosticV50:
    d = frame[["expected_r_v47", "net_r", "signal_time", "venue"]].copy()
    d["expected_r_v47"] = pd.to_numeric(d["expected_r_v47"], errors="coerce")
    d["net_r"] = pd.to_numeric(d["net_r"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["expected_r_v47", "net_r"])
    n = int(len(d))
    if n < MIN_RANKING_ROWS_V50 or float(d["expected_r_v47"].std(ddof=0)) <= 1e-12:
        return RankingDiagnosticV50(n, None, None, None, None, None)

    rho = float(spearmanr(d["expected_r_v47"].to_numpy(dtype=float), d["net_r"].to_numpy(dtype=float)).statistic)
    if not np.isfinite(rho):
        rho = None

    ordered = d.sort_values(["expected_r_v47", "signal_time", "venue"], kind="mergesort").reset_index(drop=True)
    chunks = [np.asarray(idx, dtype=int) for idx in np.array_split(np.arange(n), 5)]
    if any(len(c) == 0 for c in chunks):
        spread = None
    else:
        bottom = float(ordered.iloc[chunks[0]]["net_r"].mean())
        top = float(ordered.iloc[chunks[-1]]["net_r"].mean())
        spread = top - bottom

    x = d["expected_r_v47"].to_numpy(dtype=float)
    y = d["net_r"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n, dtype=float), x])
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept, slope = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        intercept, slope = None, None

    return RankingDiagnosticV50(
        n=n,
        spearman_rho=rho,
        top_bottom_spread=None if spread is None or not np.isfinite(spread) else float(spread),
        ols_intercept=intercept,
        ols_slope=slope,
        mean_bias=float(np.mean(x - y)),
    )


def fold_direction_tests_v50(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in range(1, 6):
        vals = pd.to_numeric(frame.loc[frame["fold"].astype(int).eq(fold), metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        k_pos, n_pos, p_pos = one_sided_sign_p(vals.to_numpy(dtype=float))
        k_neg, n_neg, p_neg = one_sided_sign_p((-vals).to_numpy(dtype=float))
        rows.append({
            "fold": fold,
            "metric": metric,
            "n_units": int(vals.size),
            "median": float(vals.median()) if len(vals) else None,
            "positive_units": int(k_pos),
            "negative_units": int(k_neg),
            "nonzero_units": int(max(n_pos, n_neg)),
            "p_positive": float(p_pos),
            "p_negative": float(p_neg),
        })
    q_pos = bh_adjust([r["p_positive"] for r in rows])
    q_neg = bh_adjust([r["p_negative"] for r in rows])
    for r, qp, qn in zip(rows, q_pos, q_neg):
        r["q_positive"] = float(qp)
        r["q_negative"] = float(qn)
        med = r["median"]
        r["positive_fold"] = bool(med is not None and med > 0.0 and qp <= BH_Q_V50)
        r["negative_fold"] = bool(med is not None and med < 0.0 and qn <= BH_Q_V50)
    return pd.DataFrame(rows)


def broad_harmful_stage_v50(deltas: Iterable[float]) -> bool:
    x = np.asarray(list(deltas), dtype=float)
    x = x[np.isfinite(x)]
    if x.size != 5:
        return False
    return bool(np.sum(x < 0.0) >= 3 and np.median(x) < 0.0)


def route_decision_v50(
    *,
    state_failure: bool,
    ranking_failure: bool,
    admission_failure: bool,
    nonoverlap_failure: bool,
    governor_failure: bool,
) -> str:
    conditions = {
        "V50_STATE_UTILITY_TRANSPORT_FAILURE_SUPPORTED": bool(state_failure),
        "V50_EXPECTED_R_RANKING_FAILURE_SUPPORTED": bool(ranking_failure),
        "V50_EXPECTED_R_ADMISSION_FAILURE_SUPPORTED": bool(admission_failure),
        "V50_NONOVERLAP_FAILURE_SUPPORTED": bool(nonoverlap_failure),
        "V50_GOVERNOR_FAILURE_SUPPORTED": bool(governor_failure),
    }
    hits = [name for name, ok in conditions.items() if ok]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "V50_MIXED_UTILITY_PIPELINE_FAILURE_SUPPORTED"
    return "V50_UTILITY_MAPPING_EVIDENCE_INCONCLUSIVE"
