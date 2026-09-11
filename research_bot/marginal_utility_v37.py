from __future__ import annotations

"""v0.37 preregistered marginal-utility portfolio admission.

This stage is frozen before v0.36 outcomes are inspected. It cannot rescue a
parent v0.36 model that fails the event-alpha gate: joint eligibility requires
both the parent event gate and the v0.37 portfolio gate.

Kraken remains untouched. CoinEx/OKX/KuCoin are consumed development evidence.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd

from research_bot.event_utility_v36 import DEVELOPMENT_VENUES_V36, V36_CANDIDATES


PRIOR_EFFECTIVE_TRIALS_V37 = 123
NEW_TRIALS_V37 = 9
TOTAL_EFFECTIVE_TRIALS_V37 = PRIOR_EFFECTIVE_TRIALS_V37 + NEW_TRIALS_V37
DEVELOPMENT_VENUES_V37 = DEVELOPMENT_VENUES_V36
RESERVED_HOLDOUT_VENUE_V37 = "kraken"


@dataclass(frozen=True)
class V37Policy:
    name: str
    correlation_weight: float
    cvar_weight: float
    duration_weight: float
    uncertainty_weight: float
    max_new_positions_per_batch: int = 4
    correlation_lookback: int = 60
    cvar_alpha: float = 0.10


V37_POLICIES: tuple[V37Policy, ...] = (
    V37Policy("BALANCED", 0.20, 0.25, 0.10, 0.20),
    V37Policy("TAIL_DEFENSIVE", 0.10, 0.40, 0.10, 0.20),
    V37Policy("DIVERSITY_FIRST", 0.35, 0.15, 0.10, 0.20),
)


def preregistration_manifest_v37() -> dict[str, Any]:
    return {
        "version": "v0.37",
        "experiment": "MARGINAL_UTILITY_PORTFOLIO_ARBITRATION",
        "preregistered_before_v36_result": True,
        "parent_candidates": [c.name for c in V36_CANDIDATES],
        "policies": [asdict(p) for p in V37_POLICIES],
        "candidate_count": len(V36_CANDIDATES) * len(V37_POLICIES),
        "development_venues": list(DEVELOPMENT_VENUES_V37),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V37,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V37,
        "new_trials": NEW_TRIALS_V37,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V37,
        "one_open_position_per_symbol": True,
        "settlement_rule": "prior exit_time < new entry_time; same-time exit remains open",
        "risk_context": "trailing returns ending no later than signal_time",
        "utility_components": ["v36 utility rank", "max open-position correlation", "normalized trailing CVaR", "predicted duration", "conformal uncertainty"],
        "aggregate_open_risk_cap": 0.02,
        "directional_open_risk_cap": 0.015,
        "hard_drawdown_cap": 0.05,
        "min_trades_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "min_block_ci_low": 0.0,
        "parent_v36_must_pass": True,
        "kraken_touched": False,
        "threshold_relaxation": False,
        "post_result_policy_retuning": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def returns_panel_v37(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        x = frame[["timestamp", "close"]].copy().sort_values("timestamp")
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
        x["ret"] = pd.to_numeric(x["close"], errors="coerce").pct_change()
        x["symbol"] = symbol
        pieces.append(x[["timestamp", "symbol", "ret"]])
    if not pieces:
        return pd.DataFrame()
    long = pd.concat(pieces, ignore_index=True)
    return long.pivot_table(index="timestamp", columns="symbol", values="ret", aggfunc="last").sort_index()


def _risk_context(
    returns: pd.DataFrame,
    *,
    signal_time: pd.Timestamp,
    symbol: str,
    open_symbols: list[str],
    policy: V37Policy,
) -> tuple[float, float]:
    if returns.empty or symbol not in returns.columns:
        return 0.0, 0.0
    hist = returns.loc[returns.index <= signal_time].tail(policy.correlation_lookback)
    s = pd.to_numeric(hist[symbol], errors="coerce").dropna()
    if len(s) < 20:
        cvar_norm = 0.0
    else:
        k = max(1, int(math.ceil(policy.cvar_alpha * len(s))))
        tail = np.sort(s.to_numpy(dtype=float))[:k]
        cvar = abs(float(np.mean(tail)))
        vol = float(np.std(s.to_numpy(dtype=float), ddof=1))
        cvar_norm = float(np.clip(cvar / max(3.0 * vol, 1e-9), 0.0, 1.0))

    corr_penalty = 0.0
    peers = [p for p in open_symbols if p in hist.columns and p != symbol]
    if peers and len(hist) >= 20:
        vals = []
        for peer in peers:
            pair = hist[[symbol, peer]].dropna()
            if len(pair) >= 20:
                c = float(pair[symbol].corr(pair[peer]))
                if np.isfinite(c):
                    vals.append(abs(c))
        if vals:
            corr_penalty = float(np.clip(max(vals), 0.0, 1.0))
    return corr_penalty, cvar_norm


def arbitrate_events_v37(
    scored_events: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    policy: V37Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if scored_events.empty:
        return scored_events.copy(), {"input_selected": 0, "admitted": 0}
    x = scored_events.copy()
    if "selected_v36" in x:
        x = x.loc[x["selected_v36"].astype(bool)].copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="raise")
    x = x.sort_values(["entry_time", "symbol", "base_strategy_v36"], kind="mergesort").reset_index(drop=True)
    x["admitted_v37"] = False
    x["reject_reason_v37"] = ""
    x["utility_rank_v37"] = np.nan
    x["correlation_penalty_v37"] = np.nan
    x["cvar_penalty_v37"] = np.nan
    x["duration_penalty_v37"] = np.nan
    x["uncertainty_penalty_v37"] = np.nan
    x["marginal_score_v37"] = np.nan

    returns = returns_panel_v37(frames)
    pending: list[dict[str, Any]] = []
    duplicate_symbol_rejections = 0
    capacity_rejections = 0

    for entry_time, group in x.groupby("entry_time", sort=True):
        pending = [p for p in pending if pd.Timestamp(p["exit_time"]) >= entry_time]
        open_symbols = [str(p["symbol"]) for p in pending]
        candidates: list[tuple[int, float]] = []
        g = group.copy()
        utility = pd.to_numeric(g["utility_v36"], errors="coerce")
        ranks = utility.rank(method="average", pct=True)
        for idx, row in g.iterrows():
            symbol = str(row["symbol"])
            x.at[idx, "utility_rank_v37"] = float(ranks.loc[idx]) if np.isfinite(ranks.loc[idx]) else 0.0
            if symbol in open_symbols:
                x.at[idx, "reject_reason_v37"] = "SYMBOL_ALREADY_OPEN"
                duplicate_symbol_rejections += 1
                continue
            signal_time = pd.Timestamp(row["signal_time"])
            corr, cvar = _risk_context(
                returns,
                signal_time=signal_time,
                symbol=symbol,
                open_symbols=open_symbols,
                policy=policy,
            )
            duration = float(np.clip(float(row.get("predicted_duration_days_v36", 35.0)) / 35.0, 0.0, 1.0))
            pred = abs(float(row.get("predicted_r_v36", 0.0)))
            buffer = abs(float(row.get("conformal_buffer_v36", 0.0)))
            uncertainty = float(np.clip(buffer / max(pred + buffer, 1e-9), 0.0, 1.0))
            score = (
                float(ranks.loc[idx])
                - policy.correlation_weight * corr
                - policy.cvar_weight * cvar
                - policy.duration_weight * duration
                - policy.uncertainty_weight * uncertainty
            )
            x.at[idx, "correlation_penalty_v37"] = corr
            x.at[idx, "cvar_penalty_v37"] = cvar
            x.at[idx, "duration_penalty_v37"] = duration
            x.at[idx, "uncertainty_penalty_v37"] = uncertainty
            x.at[idx, "marginal_score_v37"] = score
            candidates.append((int(idx), float(score)))

        candidates.sort(key=lambda t: (-t[1], str(x.at[t[0], "symbol"]), str(x.at[t[0], "base_strategy_v36"])))
        used_symbols: set[str] = set()
        admitted_count = 0
        for idx, _ in candidates:
            symbol = str(x.at[idx, "symbol"])
            if symbol in used_symbols:
                x.at[idx, "reject_reason_v37"] = "DUPLICATE_SYMBOL_SAME_BATCH"
                duplicate_symbol_rejections += 1
                continue
            if admitted_count >= policy.max_new_positions_per_batch:
                x.at[idx, "reject_reason_v37"] = "BATCH_CAPACITY"
                capacity_rejections += 1
                continue
            x.at[idx, "admitted_v37"] = True
            used_symbols.add(symbol)
            admitted_count += 1
            pending.append({"symbol": symbol, "exit_time": x.at[idx, "exit_time"]})

    diag = {
        "policy": policy.name,
        "input_selected": int(len(x)),
        "admitted": int(x["admitted_v37"].sum()),
        "duplicate_symbol_rejections": int(duplicate_symbol_rejections),
        "capacity_rejections": int(capacity_rejections),
        "mean_marginal_score_admitted": float(pd.to_numeric(x.loc[x["admitted_v37"], "marginal_score_v37"], errors="coerce").mean()) if bool(x["admitted_v37"].any()) else np.nan,
    }
    return x, diag


def _num(m: dict[str, Any], key: str) -> float:
    try:
        v = float(m.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return v


def screen_three_venues_v37(
    metrics_by_venue: dict[str, dict[str, Any]],
    *,
    parent_event_eligible: bool,
) -> dict[str, Any]:
    rows = [metrics_by_venue[v] for v in DEVELOPMENT_VENUES_V37]
    trades = [int(r.get("trades", 0) or 0) for r in rows]
    pf = [_num(r, "profit_factor") for r in rows]
    exp = [_num(r, "expectancy_r") for r in rows]
    breadth = [_num(r, "positive_asset_fraction") for r in rows]
    dd = [abs(_num(r, "max_drawdown")) for r in rows]
    ci = [_num(r, "block_ci_low") for r in rows]
    portfolio_eligible = (
        all(n >= 200 for n in trades)
        and all(np.isfinite(v) and v >= 1.05 for v in pf)
        and all(np.isfinite(v) and v > 0 for v in exp)
        and all(np.isfinite(v) and v >= 0.60 for v in breadth)
        and all(np.isfinite(v) and v <= 0.05 for v in dd)
        and all(np.isfinite(v) and v > 0 for v in ci)
    )
    return {
        "parent_event_eligible_v36": bool(parent_event_eligible),
        "portfolio_eligible_v37": bool(portfolio_eligible),
        "joint_eligible_v37": bool(parent_event_eligible and portfolio_eligible),
        "robust_min_trades_v37": min(trades),
        "robust_floor_profit_factor_v37": min(pf),
        "robust_floor_expectancy_r_v37": min(exp),
        "robust_floor_breadth_v37": min(breadth),
        "robust_floor_block_ci_low_v37": min(ci),
        "robust_worst_drawdown_v37": max(dd),
    }


def select_v37_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("joint_eligible_v37", False))]
    if not eligible:
        return None
    eligible.sort(key=lambda r: str(r.get("strategy", "")))
    return max(eligible, key=lambda r: (
        float(r["robust_floor_block_ci_low_v37"]),
        float(r["robust_floor_breadth_v37"]),
        float(r["robust_floor_profit_factor_v37"]),
        float(r["robust_floor_expectancy_r_v37"]),
        int(r["robust_min_trades_v37"]),
        -float(r["robust_worst_drawdown_v37"]),
    ))


def v37_decision(winner: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "version": "v0.37",
        "decision": "V37_JOINT_CANDIDATE_LOCKED_FOR_UNTOUCHED_KRAKEN" if winner else "NO_V37_JOINT_EVENT_PORTFOLIO_CANDIDATE",
        "winner": None if winner is None else winner.get("strategy"),
        "development_venues": list(DEVELOPMENT_VENUES_V37),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V37,
        "kraken_touched": False,
        "holdout_authorized_to_run": bool(winner is not None),
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
