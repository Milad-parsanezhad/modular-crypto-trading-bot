from __future__ import annotations

"""v0.38 temporal/model-consensus event-alpha stability screen.

Motivation: v0.36 HistGB showed strong cross-venue PF/expectancy/breadth but failed
CoinEx moving-block CI. v0.37 showed that adding harder portfolio admission before
repairing event stability can destroy sample size. v0.38 therefore works only on
already-consumed v0.36 event predictions and targets temporal/model stability.

Kraken is sealed. This module never fetches exchange data and cannot authorize
PAPER replacement or LIVE execution.
"""

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd


PRIOR_EFFECTIVE_TRIALS_V38 = 132
NEW_TRIALS_V38 = 5
TOTAL_EFFECTIVE_TRIALS_V38 = PRIOR_EFFECTIVE_TRIALS_V38 + NEW_TRIALS_V38
DEVELOPMENT_VENUES_V38 = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V38 = "kraken"
RISK_PER_EVENT_V38 = 0.0025
SELECTION_FRACTION_V38 = 0.50

MODEL_NAMES_V38 = (
    "V36_EXPECTED_NETR_RIDGE",
    "V36_EXPECTED_NETR_HUBER",
    "V36_EXPECTED_NETR_HISTGB",
)


@dataclass(frozen=True)
class V38Candidate:
    name: str
    rule: str


V38_CANDIDATES: tuple[V38Candidate, ...] = (
    V38Candidate("V38_MEDIAN_UTILITY_GLOBAL", "median utility, global top 50%"),
    V38Candidate("V38_MEDIAN_UTILITY_QUARTER_BALANCED", "median utility, top 50% within calendar quarter"),
    V38Candidate("V38_DISAGREEMENT_PENALIZED", "median utility minus 0.50*cross-model utility std, global top 50%"),
    V38Candidate("V38_TWO_OF_THREE_SELECTION_QUORUM", "at least two of three frozen v0.36 models selected the event"),
    V38Candidate("V38_REGIME_BALANCED_MEDIAN", "median utility, top 50% within frozen v0.36 regime state"),
)


def preregistration_manifest_v38() -> dict[str, Any]:
    return {
        "version": "v0.38",
        "experiment": "TEMPORAL_MODEL_CONSENSUS_STABILITY",
        "motivation": "repair v0.36 temporal instability before any further portfolio optimization",
        "candidate_count": len(V38_CANDIDATES),
        "candidates": [asdict(c) for c in V38_CANDIDATES],
        "source": "frozen v0.36 scored events only; no new market data",
        "development_venues": list(DEVELOPMENT_VENUES_V38),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V38,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V38,
        "new_trials": NEW_TRIALS_V38,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V38,
        "selection_fraction": SELECTION_FRACTION_V38,
        "min_selected_events_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "min_block_ci_low": 0.0,
        "min_positive_quarter_fraction": 0.60,
        "stress_round_trip_bps": 36.0,
        "stress_min_profit_factor": 1.0,
        "portfolio_drawdown_gate_applied_here": False,
        "kraken_touched": False,
        "post_result_threshold_retuning": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def _key_columns() -> list[str]:
    return ["signal_time", "entry_time", "exit_time", "symbol", "base_strategy_v36", "side", "entry", "stop"]


def build_consensus_panel_v38(scored_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    missing = [m for m in MODEL_NAMES_V38 if m not in scored_by_model]
    if missing:
        raise ValueError(f"missing v0.36 model panels: {missing}")

    base_name = "V36_EXPECTED_NETR_HISTGB"
    base = scored_by_model[base_name].copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        base[col] = pd.to_datetime(base[col], utc=True, errors="raise")
    keep = _key_columns() + ["r_multiple", "f_regime_v36"]
    panel = base.loc[:, keep].copy()

    for model in MODEL_NAMES_V38:
        x = scored_by_model[model].copy()
        for col in ("signal_time", "entry_time", "exit_time"):
            x[col] = pd.to_datetime(x[col], utc=True, errors="raise")
        short = model.replace("V36_EXPECTED_NETR_", "").lower()
        cols = _key_columns() + ["utility_v36", "selected_v36", "lower_r_v36", "predicted_r_v36", "conformal_buffer_v36"]
        z = x.loc[:, cols].copy().rename(columns={
            "utility_v36": f"utility_{short}_v38",
            "selected_v36": f"selected_{short}_v38",
            "lower_r_v36": f"lower_r_{short}_v38",
            "predicted_r_v36": f"predicted_r_{short}_v38",
            "conformal_buffer_v36": f"buffer_{short}_v38",
        })
        panel = panel.merge(z, on=_key_columns(), how="inner", validate="one_to_one")

    if panel.empty:
        raise ValueError("v0.38 consensus merge produced no common events")
    return panel.sort_values(["signal_time", "symbol", "base_strategy_v36"], kind="mergesort").reset_index(drop=True)


def _top_fraction_mask(score: pd.Series, group: pd.Series | None = None) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce")
    if group is None:
        valid = s.notna()
        if not valid.any():
            return pd.Series(False, index=s.index)
        q = float(s.loc[valid].quantile(1.0 - SELECTION_FRACTION_V38))
        return valid & (s >= q)
    out = pd.Series(False, index=s.index)
    frame = pd.DataFrame({"score": s, "group": group}, index=s.index)
    for _, g in frame.groupby("group", dropna=False, sort=True):
        valid = g["score"].notna()
        if not valid.any():
            continue
        q = float(g.loc[valid, "score"].quantile(1.0 - SELECTION_FRACTION_V38))
        out.loc[g.index] = valid & (g["score"] >= q)
    return out


def apply_candidate_v38(panel: pd.DataFrame, candidate: V38Candidate) -> pd.DataFrame:
    x = panel.copy()
    util_cols = ["utility_ridge_v38", "utility_huber_v38", "utility_histgb_v38"]
    util = x[util_cols].apply(pd.to_numeric, errors="coerce")
    median = util.median(axis=1, skipna=True)
    disagreement = util.std(axis=1, ddof=0, skipna=True).fillna(0.0)
    x["ensemble_median_utility_v38"] = median
    x["ensemble_disagreement_v38"] = disagreement

    if candidate.name == "V38_MEDIAN_UTILITY_GLOBAL":
        score = median
        selected = _top_fraction_mask(score)
    elif candidate.name == "V38_MEDIAN_UTILITY_QUARTER_BALANCED":
        score = median
        quarter = pd.to_datetime(x["signal_time"], utc=True).dt.to_period("Q").astype(str)
        selected = _top_fraction_mask(score, quarter)
    elif candidate.name == "V38_DISAGREEMENT_PENALIZED":
        score = median - 0.50 * disagreement
        selected = _top_fraction_mask(score)
    elif candidate.name == "V38_TWO_OF_THREE_SELECTION_QUORUM":
        votes = (
            x[["selected_ridge_v38", "selected_huber_v38", "selected_histgb_v38"]]
            .fillna(False).astype(bool).sum(axis=1)
        )
        score = votes.astype(float) + 1e-6 * median.fillna(-1e9)
        selected = votes >= 2
    elif candidate.name == "V38_REGIME_BALANCED_MEDIAN":
        score = median
        regime = pd.to_numeric(x["f_regime_v36"], errors="coerce").fillna(0.0)
        selected = _top_fraction_mask(score, regime)
    else:
        raise ValueError(f"unknown v0.38 candidate: {candidate.name}")

    x["score_v38"] = score
    x["selected_v38"] = selected.astype(bool)
    x["v38_candidate"] = candidate.name
    return x


def _profit_factor(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    wins = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    return wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)


def _moving_block_ci(values: Iterable[float], *, samples: int = 1000, block: int = 20, seed: int = 314) -> tuple[float, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < max(30, block):
        return np.nan, np.nan
    b = min(block, n)
    starts = np.arange(0, n - b + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    need = int(math.ceil(n / b))
    for i in range(samples):
        idx: list[int] = []
        for s in rng.choice(starts, size=need, replace=True):
            idx.extend(range(int(s), int(s) + b))
        means[i] = float(np.mean(a[np.asarray(idx[:n], dtype=int)]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metrics_v38(scored: pd.DataFrame) -> dict[str, Any]:
    z = scored.loc[scored["selected_v38"].astype(bool)].copy()
    r = pd.to_numeric(z["r_multiple"], errors="coerce")
    z = z.loc[r.notna()].copy()
    r = pd.to_numeric(z["r_multiple"], errors="coerce").to_numpy(dtype=float)
    if len(r) == 0:
        return {"selected_events": 0}

    account = RISK_PER_EVENT_V38 * r
    lo, hi = _moving_block_ci(account)
    breadth_by_symbol = z.assign(_r=r).groupby("symbol")["_r"].mean()
    breadth = float((breadth_by_symbol > 0).mean()) if len(breadth_by_symbol) else 0.0

    q = pd.to_datetime(z["signal_time"], utc=True).dt.to_period("Q").astype(str)
    qmean = z.assign(_r=r, _q=q).groupby("_q")["_r"].mean()
    positive_quarter_fraction = float((qmean > 0).mean()) if len(qmean) else 0.0

    entry = pd.to_numeric(z["entry"], errors="coerce")
    stop = pd.to_numeric(z["stop"], errors="coerce")
    stop_fraction = (entry - stop).abs() / entry.replace(0, np.nan)
    extra_cost_r = 0.0012 / stop_fraction.replace(0, np.nan)
    stressed = pd.to_numeric(z["r_multiple"], errors="coerce") - extra_cost_r

    eq = np.cumprod(1.0 + account)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return {
        "selected_events": int(len(z)),
        "profit_factor": float(_profit_factor(r)),
        "expectancy_r": float(np.mean(r)),
        "positive_asset_fraction": breadth,
        "block_ci_low": lo,
        "block_ci_high": hi,
        "positive_quarter_fraction": positive_quarter_fraction,
        "quarter_count": int(len(qmean)),
        "stress_36bps_profit_factor": float(_profit_factor(stressed.to_numpy(dtype=float))),
        "event_level_max_drawdown": float(np.min(dd)) if len(dd) else np.nan,
    }


def screen_three_venues_v38(metrics_by_venue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metrics_by_venue[v] for v in DEVELOPMENT_VENUES_V38]
    n = [int(r.get("selected_events", 0) or 0) for r in rows]
    pf = [float(r.get("profit_factor", np.nan)) for r in rows]
    exp = [float(r.get("expectancy_r", np.nan)) for r in rows]
    br = [float(r.get("positive_asset_fraction", np.nan)) for r in rows]
    ci = [float(r.get("block_ci_low", np.nan)) for r in rows]
    qf = [float(r.get("positive_quarter_fraction", np.nan)) for r in rows]
    spf = [float(r.get("stress_36bps_profit_factor", np.nan)) for r in rows]
    eligible = (
        all(v >= 200 for v in n)
        and all(np.isfinite(v) and v >= 1.05 for v in pf)
        and all(np.isfinite(v) and v > 0 for v in exp)
        and all(np.isfinite(v) and v >= 0.60 for v in br)
        and all(np.isfinite(v) and v > 0 for v in ci)
        and all(np.isfinite(v) and v >= 0.60 for v in qf)
        and all(np.isfinite(v) and v >= 1.0 for v in spf)
    )
    return {
        "development_eligible_v38": bool(eligible),
        "robust_min_events_v38": min(n),
        "robust_floor_profit_factor_v38": min(pf),
        "robust_floor_expectancy_r_v38": min(exp),
        "robust_floor_breadth_v38": min(br),
        "robust_floor_block_ci_low_v38": min(ci),
        "robust_floor_positive_quarter_fraction_v38": min(qf),
        "robust_floor_stress_pf_v38": min(spf),
    }


def select_v38_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v38", False))]
    if not eligible:
        return None
    eligible.sort(key=lambda r: str(r.get("strategy", "")))
    return max(eligible, key=lambda r: (
        float(r["robust_floor_block_ci_low_v38"]),
        float(r["robust_floor_positive_quarter_fraction_v38"]),
        float(r["robust_floor_breadth_v38"]),
        float(r["robust_floor_profit_factor_v38"]),
        float(r["robust_floor_expectancy_r_v38"]),
        int(r["robust_min_events_v38"]),
    ))


def decision_v38(winner: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "version": "v0.38",
        "decision": "V38_TEMPORAL_CONSENSUS_CANDIDATE_REQUIRES_SEPARATE_KRAKEN_HOLDOUT" if winner else "NO_V38_TEMPORALLY_STABLE_DEVELOPMENT_CANDIDATE",
        "winner": None if winner is None else winner.get("strategy"),
        "development_venues": list(DEVELOPMENT_VENUES_V38),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V38,
        "kraken_touched": False,
        "holdout_authorized_to_run_in_this_stage": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
