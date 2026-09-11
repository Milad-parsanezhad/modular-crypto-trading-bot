from __future__ import annotations

"""v0.33 preregistered breadth-stability hypotheses.

CoinEx, OKX and KuCoin are all consumed development evidence. Kraken is reserved
as the untouched v0.34 holdout. Candidate parameters are frozen before Kraken is
ever fetched.
"""

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import numpy as np
import pandas as pd

PRIOR_EFFECTIVE_TRIALS_V33 = 108
NEW_TRIALS_V33 = 6
TOTAL_EFFECTIVE_TRIALS_V33 = PRIOR_EFFECTIVE_TRIALS_V33 + NEW_TRIALS_V33
DEVELOPMENT_VENUES_V33 = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V33 = "kraken"


@dataclass(frozen=True)
class V33Candidate:
    name: str
    breadth_threshold: float
    persistence_bars: int


V33_CANDIDATES: tuple[V33Candidate, ...] = tuple(
    V33Candidate(f"V33_D1_CUSUM_BREADTH_{int(t*100)}_P{p}", t, p)
    for p in (1, 3)
    for t in (0.55, 0.60, 0.65)
)


def preregistration_manifest_v33() -> dict[str, Any]:
    return {
        "version": "v0.33",
        "experiment": "CROSS_VENUE_BREADTH_STABILITY",
        "base_strategy": "V30_D1_CUSUM_BREAKOUT_3",
        "candidate_count": len(V33_CANDIDATES),
        "candidates": [asdict(x) for x in V33_CANDIDATES],
        "development_venues": list(DEVELOPMENT_VENUES_V33),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V33,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V33,
        "new_trials": NEW_TRIALS_V33,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V33,
        "min_trades_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "max_drawdown": 0.05,
        "min_block_ci_low": 0.0,
        "round_trip_cost_fraction": 0.0024,
        "aggregate_open_risk_cap": 0.02,
        "directional_open_risk_cap": 0.015,
        "threshold_relaxation": False,
        "base_alpha_retuning": False,
        "kraken_touched": False,
        "live_execution_authorized": False,
    }


def cross_sectional_breadth(frames: dict[str, pd.DataFrame], *, ma_window: int = 50) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        x = frame[["timestamp", "close"]].copy().sort_values("timestamp")
        x["ma"] = x["close"].rolling(ma_window, min_periods=ma_window).mean()
        x["above"] = (x["close"] > x["ma"]).astype(float)
        x["symbol"] = symbol
        pieces.append(x[["timestamp", "symbol", "above"]])
    if not pieces:
        return pd.DataFrame(columns=["timestamp", "breadth_v33"])
    long = pd.concat(pieces, ignore_index=True)
    b = long.groupby("timestamp", sort=True)["above"].mean().rename("breadth_v33").reset_index()
    return b


def apply_breadth_gate(
    base_direction: pd.Series,
    features: pd.DataFrame,
    breadth: pd.DataFrame,
    candidate: V33Candidate,
) -> tuple[pd.Series, pd.DataFrame]:
    f = features.copy().sort_values("timestamp").reset_index(drop=True)
    b = breadth.copy().sort_values("timestamp")
    f = pd.merge_asof(f, b, on="timestamp", direction="backward", allow_exact_matches=True)
    smooth = f["breadth_v33"].rolling(candidate.persistence_bars, min_periods=candidate.persistence_bars).mean()
    long_ok = smooth >= candidate.breadth_threshold
    short_ok = smooth <= (1.0 - candidate.breadth_threshold)
    d = pd.Series(base_direction.to_numpy(copy=True), index=f.index, dtype="int8")
    d.loc[(d > 0) & ~long_ok.fillna(False)] = 0
    d.loc[(d < 0) & ~short_ok.fillna(False)] = 0
    f["breadth_smoothed_v33"] = smooth
    f["breadth_long_ok_v33"] = long_ok.fillna(False)
    f["breadth_short_ok_v33"] = short_ok.fillna(False)
    return d, f


def _num(m: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        v = float(m.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def screen_three_venues(metrics_by_venue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metrics_by_venue[v] for v in DEVELOPMENT_VENUES_V33]
    trades = [int(r.get("trades", 0) or 0) for r in rows]
    pf = [_num(r, "profit_factor") for r in rows]
    exp = [_num(r, "expectancy_r") for r in rows]
    breadth = [_num(r, "positive_asset_fraction") for r in rows]
    dd = [abs(_num(r, "max_drawdown")) for r in rows]
    ci = [_num(r, "block_ci_low") for r in rows]
    eligible = (
        all(n >= 200 for n in trades)
        and all(np.isfinite(x) and x >= 1.05 for x in pf)
        and all(np.isfinite(x) and x > 0 for x in exp)
        and all(np.isfinite(x) and x >= 0.60 for x in breadth)
        and all(np.isfinite(x) and x <= 0.05 for x in dd)
        and all(np.isfinite(x) and x > 0 for x in ci)
    )
    return {
        "development_eligible_v33": bool(eligible),
        "robust_min_trades_v33": min(trades),
        "robust_floor_profit_factor_v33": min(pf),
        "robust_floor_expectancy_r_v33": min(exp),
        "robust_floor_breadth_v33": min(breadth),
        "robust_floor_block_ci_low_v33": min(ci),
        "robust_worst_drawdown_v33": max(dd),
    }


def select_v33_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v33", False))]
    if not eligible:
        return None
    eligible = sorted(eligible, key=lambda r: str(r.get("strategy", "")))
    return max(eligible, key=lambda r: (
        float(r["robust_floor_block_ci_low_v33"]),
        float(r["robust_floor_breadth_v33"]),
        float(r["robust_floor_profit_factor_v33"]),
        float(r["robust_floor_expectancy_r_v33"]),
        int(r["robust_min_trades_v33"]),
        -float(r["robust_worst_drawdown_v33"]),
    ))


def v33_decision(winner: dict[str, Any] | None) -> dict[str, Any]:
    state = "V33_BREADTH_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT" if winner else "NO_V33_ROBUST_DEVELOPMENT_CANDIDATE"
    return {
        "version": "v0.33",
        "decision": state,
        "winner": None if winner is None else winner.get("strategy"),
        "timeframe": "1d" if winner is not None else None,
        "development_venues": list(DEVELOPMENT_VENUES_V33),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V33,
        "kraken_touched": False,
        "threshold_relaxation": False,
        "base_alpha_retuning": False,
        "forward_paper_candidate_authorized": False,
        "live_execution_authorized": False,
    }
