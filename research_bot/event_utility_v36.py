from __future__ import annotations

"""v0.36 expected-net-R, uncertainty and duration-aware event ranking.

Research-only module. CoinEx, OKX and KuCoin are already-consumed development
venues. Kraken remains sealed and is not fetched by this module.

The purpose of v0.36 is to separate event alpha from portfolio admission. Instead
of classifying win/loss, the model predicts post-cost R-multiple, constructs a
one-sided conformal lower bound, predicts capital-lock duration, and ranks events
by lower-bound economic utility per sqrt(day) of predicted holding time.

Every target development venue is scored by a model fitted on the other two
venues. Source data are split chronologically into 60% fit / 20% calibration;
only target events strictly after the source calibration cutoff are evaluated.
This is still consumed development evidence and cannot authorize PAPER/LIVE.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRIOR_EFFECTIVE_TRIALS_V36 = 120
NEW_TRIALS_V36 = 3
TOTAL_EFFECTIVE_TRIALS_V36 = PRIOR_EFFECTIVE_TRIALS_V36 + NEW_TRIALS_V36
DEVELOPMENT_VENUES_V36 = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V36 = "kraken"
SELECTION_FRACTION_V36 = 0.50
CONFORMAL_ALPHA_V36 = 0.20
RISK_PER_EVENT_V36 = 0.0025

BASE_D1_STRATEGIES_V36 = (
    "V30_D1_REGIME_MOM_20_90",
    "V30_D1_REGIME_MOM_30_120",
    "V30_D1_CUSUM_BREAKOUT_3",
    "V30_D1_CUSUM_BREAKOUT_4",
    "V30_D1_ICHIMOKU_REGIME_100",
    "V30_D1_ICHIMOKU_REGIME_125",
)

MODEL_FEATURES_V36 = (
    "f_side_v36",
    "f_atr_pct_v36",
    "f_ret1_v36",
    "f_ret12_v36",
    "f_ema20_gap_v36",
    "f_ema50_gap_v36",
    "f_ema200_gap_v36",
    "f_regime_v36",
    "f_dispersion_ratio_v36",
    "f_volume_ratio_v36",
    "f_cloud_position_v36",
    "f_month_sin_v36",
    "f_month_cos_v36",
) + tuple(f"f_strategy_{i}_v36" for i in range(len(BASE_D1_STRATEGIES_V36)))


@dataclass(frozen=True)
class V36Candidate:
    name: str
    model_family: str


V36_CANDIDATES: tuple[V36Candidate, ...] = (
    V36Candidate("V36_EXPECTED_NETR_RIDGE", "ridge"),
    V36Candidate("V36_EXPECTED_NETR_HUBER", "huber"),
    V36Candidate("V36_EXPECTED_NETR_HISTGB", "histgb"),
)


def preregistration_manifest_v36() -> dict[str, Any]:
    return {
        "version": "v0.36",
        "experiment": "EXPECTED_NET_R_CONFORMAL_DURATION_RANKING",
        "candidate_count": len(V36_CANDIDATES),
        "candidates": [asdict(c) for c in V36_CANDIDATES],
        "base_event_pool": list(BASE_D1_STRATEGIES_V36),
        "development_venues": list(DEVELOPMENT_VENUES_V36),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V36,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V36,
        "new_trials": NEW_TRIALS_V36,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V36,
        "source_split": "60% chronological fit / 20% conformal calibration / target venue strictly after calibration cutoff",
        "target": "post-cost R-multiple",
        "uncertainty": "one-sided 80% split-conformal lower bound",
        "duration_target": "log1p realized holding days on source venues only",
        "utility": "conformal_lower_R / sqrt(1 + predicted_holding_days)",
        "selection_fraction": SELECTION_FRACTION_V36,
        "symbol_identity_feature": False,
        "min_selected_events_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "min_block_ci_low": 0.0,
        "stress_round_trip_bps": 36.0,
        "stress_min_profit_factor": 1.0,
        "portfolio_drawdown_gate_applied_here": False,
        "reason_no_portfolio_dd_gate": "v0.36 deliberately isolates event alpha; portfolio admission and 5% DD are v0.37",
        "kraken_touched": False,
        "threshold_relaxation": False,
        "post_result_model_retuning": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def _make_regressor(family: str):
    if family == "ridge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
    if family == "huber":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=500)),
        ])
    if family == "histgb":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=314,
            )),
        ])
    raise ValueError(f"unknown v0.36 model family: {family}")


def add_fixed_strategy_features_v36(events: pd.DataFrame) -> pd.DataFrame:
    x = events.copy()
    strategy = x.get("base_strategy_v36", x.get("strategy", pd.Series("", index=x.index))).astype(str)
    for i, name in enumerate(BASE_D1_STRATEGIES_V36):
        x[f"f_strategy_{i}_v36"] = strategy.eq(name).astype(float)
    signal = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    month = signal.dt.month.astype(float)
    x["f_month_sin_v36"] = np.sin(2.0 * math.pi * month / 12.0)
    x["f_month_cos_v36"] = np.cos(2.0 * math.pi * month / 12.0)
    x["f_side_v36"] = pd.to_numeric(x["side"], errors="coerce")
    return x


def _model_matrix(events: pd.DataFrame) -> pd.DataFrame:
    x = add_fixed_strategy_features_v36(events)
    for col in MODEL_FEATURES_V36:
        if col not in x:
            x[col] = np.nan
    return x.loc[:, MODEL_FEATURES_V36].apply(pd.to_numeric, errors="coerce")


def _source_split(source_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    x = source_events.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    times = pd.Index(sorted(x["signal_time"].dropna().unique()))
    if len(times) < 30:
        raise ValueError("v0.36 source panel has too few unique timestamps")
    fit_i = max(1, min(len(times) - 3, int(math.floor(0.60 * len(times))) - 1))
    cal_i = max(fit_i + 1, min(len(times) - 2, int(math.floor(0.80 * len(times))) - 1))
    fit_cut = pd.Timestamp(times[fit_i])
    cal_cut = pd.Timestamp(times[cal_i])
    fit = x.loc[x["signal_time"] <= fit_cut].copy()
    cal = x.loc[(x["signal_time"] > fit_cut) & (x["signal_time"] <= cal_cut)].copy()
    if len(fit) < 300 or len(cal) < 100:
        raise ValueError(f"v0.36 insufficient source split: fit={len(fit)} cal={len(cal)}")
    return fit, cal, cal_cut


def fit_score_target_v36(
    source_events: pd.DataFrame,
    target_events: pd.DataFrame,
    candidate: V36Candidate,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on two source venues and score a third, with temporal fit/calibration separation."""
    fit, cal, cal_cut = _source_split(source_events)
    target = target_events.copy()
    target["signal_time"] = pd.to_datetime(target["signal_time"], utc=True, errors="raise")
    target = target.loc[target["signal_time"] > cal_cut].copy()
    if target.empty:
        return target, {"calibration_cutoff_utc": cal_cut.isoformat(), "target_events": 0}

    y_fit = pd.to_numeric(fit["r_multiple"], errors="coerce")
    y_cal = pd.to_numeric(cal["r_multiple"], errors="coerce")
    duration_fit = (
        (pd.to_datetime(fit["exit_time"], utc=True) - pd.to_datetime(fit["entry_time"], utc=True))
        .dt.total_seconds().div(86400.0).clip(lower=1.0)
    )

    valid_fit = y_fit.notna() & duration_fit.notna()
    fit = fit.loc[valid_fit].copy()
    y_fit = y_fit.loc[valid_fit]
    duration_fit = duration_fit.loc[valid_fit]

    return_model = _make_regressor(candidate.model_family)
    duration_model = _make_regressor(candidate.model_family)
    return_model.fit(_model_matrix(fit), y_fit.to_numpy(dtype=float))
    duration_model.fit(_model_matrix(fit), np.log1p(duration_fit.to_numpy(dtype=float)))

    cal_pred = np.asarray(return_model.predict(_model_matrix(cal)), dtype=float)
    cal_y = y_cal.to_numpy(dtype=float)
    valid_cal = np.isfinite(cal_pred) & np.isfinite(cal_y)
    if int(valid_cal.sum()) < 80:
        raise ValueError("v0.36 insufficient conformal calibration observations")
    nonconformity = cal_pred[valid_cal] - cal_y[valid_cal]
    q = float(np.quantile(nonconformity, 1.0 - CONFORMAL_ALPHA_V36, method="higher"))

    pred = np.asarray(return_model.predict(_model_matrix(target)), dtype=float)
    log_duration_pred = np.asarray(duration_model.predict(_model_matrix(target)), dtype=float)
    duration_pred = np.expm1(log_duration_pred)
    duration_pred = np.clip(duration_pred, 1.0, 35.0)
    lower = pred - q
    utility = lower / np.sqrt(1.0 + duration_pred)

    out = target.copy()
    out["predicted_r_v36"] = pred
    out["conformal_buffer_v36"] = q
    out["lower_r_v36"] = lower
    out["predicted_duration_days_v36"] = duration_pred
    out["utility_v36"] = utility
    finite = np.isfinite(utility)
    threshold = float(np.quantile(utility[finite], 1.0 - SELECTION_FRACTION_V36)) if finite.any() else np.inf
    out["selected_v36"] = finite & (utility >= threshold)
    out["v36_candidate"] = candidate.name
    diag = {
        "model_family": candidate.model_family,
        "fit_events": int(len(fit)),
        "calibration_events": int(len(cal)),
        "target_events": int(len(out)),
        "selected_events": int(out["selected_v36"].sum()),
        "calibration_cutoff_utc": cal_cut.isoformat(),
        "conformal_buffer_r": q,
        "selection_utility_threshold": threshold,
        "median_predicted_r": float(np.nanmedian(pred)),
        "median_lower_r": float(np.nanmedian(lower)),
        "median_predicted_duration_days": float(np.nanmedian(duration_pred)),
    }
    return out, diag


def _profit_factor(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    wins = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    return wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)


def _moving_block_ci(values: Iterable[float], *, samples: int = 750, block: int = 20, seed: int = 314) -> tuple[float, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < max(30, block):
        return np.nan, np.nan
    b = min(block, n)
    starts = np.arange(0, n - b + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    blocks_needed = int(math.ceil(n / b))
    for i in range(samples):
        idx = []
        for s in rng.choice(starts, size=blocks_needed, replace=True):
            idx.extend(range(int(s), int(s) + b))
        means[i] = float(np.mean(a[np.asarray(idx[:n], dtype=int)]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def event_metrics_v36(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty or "selected_v36" not in scored:
        return {
            "selected_events": 0, "profit_factor": np.nan, "expectancy_r": np.nan,
            "positive_asset_fraction": 0.0, "block_ci_low": np.nan, "block_ci_high": np.nan,
            "stress_36bps_profit_factor": np.nan, "event_level_max_drawdown": np.nan,
        }
    z = scored.loc[scored["selected_v36"].astype(bool)].copy()
    r = pd.to_numeric(z["r_multiple"], errors="coerce").to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {
            "selected_events": 0, "profit_factor": np.nan, "expectancy_r": np.nan,
            "positive_asset_fraction": 0.0, "block_ci_low": np.nan, "block_ci_high": np.nan,
            "stress_36bps_profit_factor": np.nan, "event_level_max_drawdown": np.nan,
        }
    account = RISK_PER_EVENT_V36 * r
    eq = np.cumprod(1.0 + account)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    breadth_by_symbol = z.assign(_r=pd.to_numeric(z["r_multiple"], errors="coerce")).groupby("symbol")["_r"].mean()
    breadth = float((breadth_by_symbol > 0).mean()) if len(breadth_by_symbol) else 0.0
    lo, hi = _moving_block_ci(account)

    entry = pd.to_numeric(z["entry"], errors="coerce")
    stop = pd.to_numeric(z["stop"], errors="coerce")
    stop_fraction = (entry - stop).abs() / entry.replace(0, np.nan)
    extra_cost = 0.0012 / stop_fraction.replace(0, np.nan)
    stressed = pd.to_numeric(z["r_multiple"], errors="coerce") - extra_cost
    return {
        "selected_events": int(len(z)),
        "profit_factor": float(_profit_factor(r)),
        "expectancy_r": float(np.nanmean(r)),
        "positive_asset_fraction": breadth,
        "block_ci_low": lo,
        "block_ci_high": hi,
        "stress_36bps_profit_factor": float(_profit_factor(stressed.to_numpy(dtype=float))),
        "event_level_max_drawdown": float(np.nanmin(dd)) if len(dd) else np.nan,
    }


def screen_three_venues_v36(metrics_by_venue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metrics_by_venue[v] for v in DEVELOPMENT_VENUES_V36]
    trades = [int(r.get("selected_events", 0) or 0) for r in rows]
    pf = [float(r.get("profit_factor", np.nan)) for r in rows]
    exp = [float(r.get("expectancy_r", np.nan)) for r in rows]
    breadth = [float(r.get("positive_asset_fraction", np.nan)) for r in rows]
    ci = [float(r.get("block_ci_low", np.nan)) for r in rows]
    stress = [float(r.get("stress_36bps_profit_factor", np.nan)) for r in rows]
    eligible = (
        all(n >= 200 for n in trades)
        and all(np.isfinite(x) and x >= 1.05 for x in pf)
        and all(np.isfinite(x) and x > 0 for x in exp)
        and all(np.isfinite(x) and x >= 0.60 for x in breadth)
        and all(np.isfinite(x) and x > 0 for x in ci)
        and all(np.isfinite(x) and x >= 1.0 for x in stress)
    )
    return {
        "development_eligible_v36": bool(eligible),
        "robust_min_events_v36": min(trades),
        "robust_floor_profit_factor_v36": min(pf),
        "robust_floor_expectancy_r_v36": min(exp),
        "robust_floor_breadth_v36": min(breadth),
        "robust_floor_block_ci_low_v36": min(ci),
        "robust_floor_stress_pf_v36": min(stress),
    }


def select_v36_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v36", False))]
    if not eligible:
        return None
    eligible.sort(key=lambda r: str(r.get("strategy", "")))
    return max(eligible, key=lambda r: (
        float(r["robust_floor_block_ci_low_v36"]),
        float(r["robust_floor_breadth_v36"]),
        float(r["robust_floor_profit_factor_v36"]),
        float(r["robust_floor_expectancy_r_v36"]),
        int(r["robust_min_events_v36"]),
    ))


def v36_decision(winner: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "version": "v0.36",
        "decision": "V36_EVENT_UTILITY_CANDIDATE_REQUIRES_PORTFOLIO_ARBITRATION" if winner else "NO_V36_ROBUST_EVENT_UTILITY_CANDIDATE",
        "winner": None if winner is None else winner.get("strategy"),
        "development_venues": list(DEVELOPMENT_VENUES_V36),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V36,
        "kraken_touched": False,
        "portfolio_gate_complete": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
