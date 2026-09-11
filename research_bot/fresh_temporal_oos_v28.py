from __future__ import annotations

"""v0.28 preregistered strictly-fresh temporal OOS qualification.

Scientific contract
-------------------
- v0.28 may run only after v0.27 has locked one candidate and that candidate
  has passed the untouched KuCoin holdout.
- The v0.27 winner, timeframe, v0.25 allocator, transaction-cost assumption and
  qualification thresholds are immutable in v0.28.
- Temporal evidence is strictly limited to signals whose information time is
  later than the v0.27 development-lock timestamp.  No pre-lock signal may be
  recycled into this stage.
- A hard 5% drawdown breach rejects immediately; minimum trade count protects
  inference but never overrides a realized safety-limit breach.
- A pass authorizes only FORWARD_PAPER_CANDIDATE status.  It does not authorize
  paper replacement or live execution.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_V27_DECISION = "ROBUST_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"


@dataclass(frozen=True)
class V28TemporalPolicy:
    min_trades: int = 200
    min_profit_factor: float = 1.05
    min_positive_asset_fraction: float = 0.60
    max_drawdown: float = 0.05


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def v27_prerequisite_failures(v27_decision: dict[str, Any], development_lock: dict[str, Any]) -> list[str]:
    """Verify that v0.28 is permitted to consume any temporal evidence."""
    failures: list[str] = []
    if str(v27_decision.get("decision")) != REQUIRED_V27_DECISION:
        failures.append("V27_DECISION")
    if not bool(v27_decision.get("holdout_used", False)):
        failures.append("V27_HOLDOUT_NOT_USED")
    winner = v27_decision.get("winner")
    timeframe = v27_decision.get("timeframe")
    if not winner or not timeframe:
        failures.append("V27_WINNER_MISSING")
    if winner != development_lock.get("winner") or timeframe != development_lock.get("timeframe"):
        failures.append("V27_LOCK_MISMATCH")
    if bool(v27_decision.get("threshold_relaxation", True)):
        failures.append("THRESHOLD_RELAXATION")
    if bool(v27_decision.get("strategy_parameter_retuning", True)):
        failures.append("STRATEGY_RETUNING")
    if bool(v27_decision.get("holdout_winner_reselection", True)):
        failures.append("HOLDOUT_RESELECTION")
    if not development_lock.get("locked_at_utc"):
        failures.append("LOCK_TIMESTAMP_MISSING")
    return failures


def temporal_anchor(development_lock: dict[str, Any]) -> pd.Timestamp:
    """Return the immutable v0.27 selection-lock timestamp in UTC."""
    raw = development_lock.get("locked_at_utc")
    if not raw:
        raise ValueError("v0.28 requires v0.27 development_lock.locked_at_utc")
    return pd.Timestamp(raw).tz_convert("UTC") if pd.Timestamp(raw).tzinfo else pd.Timestamp(raw, tz="UTC")


def filter_strictly_post_lock(attempts: pd.DataFrame, development_lock: dict[str, Any]) -> pd.DataFrame:
    """Keep only signals whose information time is strictly after v0.27 lock."""
    if attempts.empty:
        return attempts.copy()
    column = "signal_time" if "signal_time" in attempts.columns else "entry_time"
    if column not in attempts.columns:
        raise ValueError("v0.28 requires signal_time or entry_time")
    t = pd.to_datetime(attempts[column], utc=True, errors="raise")
    anchor = temporal_anchor(development_lock)
    return attempts.loc[t > anchor].copy().reset_index(drop=True)


def compact_temporal_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert the frozen v0.25 external evaluator output to v0.28 shape."""
    def finite(key: str) -> float | None:
        value = _finite(raw.get(key))
        return float(value) if np.isfinite(value) else None

    return {
        "temporal_trades": int(raw.get("external_trades", 0) or 0),
        "temporal_total_return": finite("external_total_return"),
        "temporal_profit_factor": finite("external_profit_factor"),
        "temporal_win_rate": finite("external_win_rate"),
        "temporal_expectancy_r": finite("external_expectancy_r"),
        "temporal_max_drawdown": finite("external_max_drawdown"),
        "temporal_positive_asset_fraction": finite("external_positive_asset_fraction"),
        "temporal_block_ci_low": finite("external_block_ci_low"),
        "temporal_block_ci_high": finite("external_block_ci_high"),
        "temporal_frozen_external_pass": bool(raw.get("external_pass_v25", False)),
    }


def temporal_failures(metrics: dict[str, Any], *, policy: V28TemporalPolicy | None = None) -> list[str]:
    p = policy or V28TemporalPolicy()
    n = int(metrics.get("temporal_trades", 0) or 0)
    pf = _finite(metrics.get("temporal_profit_factor"))
    exp = _finite(metrics.get("temporal_expectancy_r"))
    breadth = _finite(metrics.get("temporal_positive_asset_fraction"))
    dd = abs(_finite(metrics.get("temporal_max_drawdown")))
    ci = _finite(metrics.get("temporal_block_ci_low"))

    failures: list[str] = []
    if n < p.min_trades:
        failures.append("TRADES")
    if not np.isfinite(pf) or pf < p.min_profit_factor:
        failures.append("PROFIT_FACTOR")
    if not np.isfinite(exp) or exp <= 0.0:
        failures.append("EXPECTANCY")
    if not np.isfinite(breadth) or breadth < p.min_positive_asset_fraction:
        failures.append("BREADTH")
    if not np.isfinite(dd) or dd > p.max_drawdown:
        failures.append("MAX_DRAWDOWN")
    if not np.isfinite(ci) or ci <= 0.0:
        failures.append("BLOCK_CI")
    return failures


def v28_decision(
    v27_decision_payload: dict[str, Any],
    development_lock: dict[str, Any],
    temporal_metrics: dict[str, Any] | None,
    *,
    policy: V28TemporalPolicy | None = None,
) -> dict[str, Any]:
    """Return the preregistered v0.28 state without authorizing live trading."""
    p = policy or V28TemporalPolicy()
    prerequisite_failures = v27_prerequisite_failures(v27_decision_payload, development_lock)
    metrics = temporal_metrics or {}

    if prerequisite_failures:
        state = "BLOCKED_BY_V27"
        reason = "v0.28 cannot consume temporal OOS evidence because the frozen v0.27 prerequisite is not satisfied."
        failures: list[str] = []
        hard_dd = False
    else:
        failures = temporal_failures(metrics, policy=p)
        dd = abs(_finite(metrics.get("temporal_max_drawdown")))
        hard_dd = bool(np.isfinite(dd) and dd > p.max_drawdown)
        n = int(metrics.get("temporal_trades", 0) or 0)
        if hard_dd:
            state = "REJECTED_FRESH_TEMPORAL_OOS"
            reason = "The locked v0.27 winner breached the frozen 5% hard drawdown limit on strictly post-lock temporal OOS."
        elif n < p.min_trades:
            state = "FRESH_TEMPORAL_OOS_ACCUMULATING"
            reason = "Strictly post-lock temporal OOS has not yet reached the frozen minimum trade count."
        elif failures:
            state = "REJECTED_FRESH_TEMPORAL_OOS"
            reason = "The locked v0.27 winner failed at least one unchanged fresh temporal OOS gate."
        else:
            state = "FORWARD_PAPER_CANDIDATE"
            reason = "The locked v0.27 winner passed untouched KuCoin and the strictly post-lock temporal OOS gates."

    return {
        "version": "v0.28",
        "decision": state,
        "reason": reason,
        "winner": v27_decision_payload.get("winner"),
        "timeframe": v27_decision_payload.get("timeframe"),
        "v27_required_decision": REQUIRED_V27_DECISION,
        "v27_prerequisite_failures": prerequisite_failures,
        "temporal_failures": failures,
        "hard_drawdown_breached": hard_dd,
        "temporal_anchor_utc": development_lock.get("locked_at_utc"),
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "winner_reselection": False,
        "historical_test_recycling": False,
        "forward_paper_candidate_authorized": state == "FORWARD_PAPER_CANDIDATE",
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
