from __future__ import annotations

"""Locked v0.26 replication/forward-OOS qualification helpers.

v0.26 does not tune a strategy.  It monitors the single v0.25 winner with the
frozen v0.25 allocator and the frozen v0.20 external qualification gates.
Live execution is never authorized here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


LOCKED_WINNER_V26 = "H4_D1_S6_VOL_RISK"
LOCKED_TIMEFRAME_V26 = "4h"
V25_SELECTION_LOCK_UTC = pd.Timestamp("2026-09-11T17:24:21Z")
V25_WORKFLOW_RUN_ID = 34626983402
V25_ARTIFACT_ID = 10275235912


@dataclass(frozen=True)
class V26QualificationPolicy:
    min_external_trades: int = 200
    min_temporal_trades: int = 200
    min_profit_factor: float = 1.05
    min_positive_asset_fraction: float = 0.60
    max_drawdown: float = 0.05


def filter_fresh_temporal_attempts(
    attempts: pd.DataFrame,
    *,
    anchor: pd.Timestamp = V25_SELECTION_LOCK_UTC,
) -> pd.DataFrame:
    """Keep only attempts whose *signal* became known strictly after v0.25 lock.

    Filtering on signal time prevents a pre-lock signal with a post-lock fill
    from leaking into the supposedly fresh temporal OOS sample.
    """

    if attempts.empty:
        return attempts.copy()
    x = attempts.copy()
    column = "signal_time" if "signal_time" in x.columns else "entry_time"
    if column not in x.columns:
        raise ValueError("v0.26 temporal filtering requires signal_time or entry_time")
    t = pd.to_datetime(x[column], utc=True, errors="raise")
    return x.loc[t > pd.Timestamp(anchor)].copy().reset_index(drop=True)


def compact_replication_metrics(raw: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Convert evaluate_external_replication_v25 output into a stable v0.26 shape."""

    def finite_number(key: str) -> float | None:
        value = raw.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    return {
        f"{prefix}_trades": int(raw.get("external_trades", 0) or 0),
        f"{prefix}_total_return": finite_number("external_total_return"),
        f"{prefix}_profit_factor": finite_number("external_profit_factor"),
        f"{prefix}_win_rate": finite_number("external_win_rate"),
        f"{prefix}_expectancy_r": finite_number("external_expectancy_r"),
        f"{prefix}_max_drawdown": finite_number("external_max_drawdown"),
        f"{prefix}_positive_asset_fraction": finite_number("external_positive_asset_fraction"),
        f"{prefix}_block_ci_low": finite_number("external_block_ci_low"),
        f"{prefix}_block_ci_high": finite_number("external_block_ci_high"),
        f"{prefix}_pass": bool(raw.get("external_pass_v25", False)),
    }


def gate_failures(metrics: dict[str, Any], *, prefix: str, policy: V26QualificationPolicy | None = None) -> list[str]:
    p = policy or V26QualificationPolicy()

    def value(name: str) -> float:
        raw = metrics.get(f"{prefix}_{name}")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return np.nan

    trades = int(metrics.get(f"{prefix}_trades", 0) or 0)
    pf = value("profit_factor")
    exp = value("expectancy_r")
    dd = abs(value("max_drawdown"))
    breadth = value("positive_asset_fraction")
    ci_low = value("block_ci_low")
    minimum = p.min_external_trades if prefix == "external" else p.min_temporal_trades

    failures: list[str] = []
    if trades < minimum:
        failures.append("TRADES")
    if not np.isfinite(pf) or pf < p.min_profit_factor:
        failures.append("PROFIT_FACTOR")
    if not np.isfinite(exp) or exp <= 0:
        failures.append("EXPECTANCY")
    if not np.isfinite(breadth) or breadth < p.min_positive_asset_fraction:
        failures.append("BREADTH")
    if not np.isfinite(dd) or dd > p.max_drawdown:
        failures.append("MAX_DRAWDOWN")
    if not np.isfinite(ci_low) or ci_low <= 0:
        failures.append("BLOCK_CI")
    return failures


def qualification_decision(
    external: dict[str, Any],
    temporal: dict[str, Any],
    *,
    policy: V26QualificationPolicy | None = None,
) -> dict[str, Any]:
    """Return the only permitted v0.26 promotion state.

    A strategy can become a FORWARD_PAPER_CANDIDATE only when both the newly
    fetched external-venue replication and the strictly post-lock temporal OOS
    clear the unchanged gates.  No live-capital state exists in this function.
    """

    p = policy or V26QualificationPolicy()
    ext_n = int(external.get("external_trades", 0) or 0)
    tmp_n = int(temporal.get("temporal_trades", 0) or 0)
    ext_fail = gate_failures(external, prefix="external", policy=p)
    tmp_fail = gate_failures(temporal, prefix="temporal", policy=p)
    ext_enough = ext_n >= p.min_external_trades
    tmp_enough = tmp_n >= p.min_temporal_trades
    ext_pass = bool(external.get("external_pass", False)) and not ext_fail
    tmp_pass = bool(temporal.get("temporal_pass", False)) and not tmp_fail

    if not ext_enough:
        state = "EXTERNAL_EVIDENCE_INSUFFICIENT"
        reason = "Fresh external venue has not yet produced the frozen minimum trade count."
    elif not ext_pass:
        state = "REJECTED_EXTERNAL_REPLICATION"
        reason = "The locked v0.25 winner failed at least one frozen external replication gate."
    elif not tmp_enough:
        state = "EXTERNAL_PASS_TEMPORAL_ACCUMULATING"
        reason = "External replication passed; strictly post-lock temporal OOS is still accumulating evidence."
    elif not tmp_pass:
        state = "REJECTED_FRESH_TEMPORAL_OOS"
        reason = "External replication passed, but the fresh temporal OOS failed at least one frozen gate."
    else:
        state = "FORWARD_PAPER_CANDIDATE"
        reason = "Both fresh external replication and strictly post-lock temporal OOS passed the unchanged gates."

    return {
        "version": "v0.26",
        "decision": state,
        "reason": reason,
        "winner": LOCKED_WINNER_V26,
        "timeframe": LOCKED_TIMEFRAME_V26,
        "external_failures": ext_fail,
        "temporal_failures": tmp_fail,
        "v25_selection_lock_utc": V25_SELECTION_LOCK_UTC.isoformat(),
        "threshold_relaxation": False,
        "strategy_retuning": False,
        "forward_paper_candidate_authorized": state == "FORWARD_PAPER_CANDIDATE",
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
