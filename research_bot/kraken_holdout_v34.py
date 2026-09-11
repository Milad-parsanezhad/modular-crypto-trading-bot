from __future__ import annotations

"""v0.34 untouched Kraken holdout qualification for the locked v0.33 winner."""

from dataclasses import dataclass
from typing import Any
import numpy as np

REQUIRED_V33_DECISION = "V33_BREADTH_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT"


@dataclass(frozen=True)
class V34Policy:
    min_trades: int = 200
    min_profit_factor: float = 1.05
    min_positive_asset_fraction: float = 0.60
    max_drawdown: float = 0.05


def prerequisite_failures(v33_decision: dict[str, Any], lock: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if v33_decision.get("decision") != REQUIRED_V33_DECISION:
        failures.append("V33_DECISION")
    if not v33_decision.get("winner") or not lock.get("winner"):
        failures.append("V33_WINNER_MISSING")
    if v33_decision.get("winner") != lock.get("winner"):
        failures.append("V33_LOCK_MISMATCH")
    if bool(v33_decision.get("kraken_touched", True)) or bool(lock.get("kraken_touched", True)):
        failures.append("KRAKEN_ALREADY_TOUCHED")
    if bool(v33_decision.get("threshold_relaxation", True)):
        failures.append("THRESHOLD_RELAXATION")
    if bool(v33_decision.get("base_alpha_retuning", True)):
        failures.append("BASE_ALPHA_RETUNING")
    return failures


def _finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def holdout_failures(metrics: dict[str, Any], policy: V34Policy | None = None) -> list[str]:
    p = policy or V34Policy()
    n = int(metrics.get("trades", 0) or 0)
    pf = _finite(metrics.get("profit_factor"))
    exp = _finite(metrics.get("expectancy_r"))
    breadth = _finite(metrics.get("positive_asset_fraction"))
    dd = abs(_finite(metrics.get("max_drawdown")))
    ci = _finite(metrics.get("block_ci_low"))
    out: list[str] = []
    if n < p.min_trades: out.append("TRADES")
    if not np.isfinite(pf) or pf < p.min_profit_factor: out.append("PROFIT_FACTOR")
    if not np.isfinite(exp) or exp <= 0: out.append("EXPECTANCY")
    if not np.isfinite(breadth) or breadth < p.min_positive_asset_fraction: out.append("BREADTH")
    if not np.isfinite(dd) or dd > p.max_drawdown: out.append("MAX_DRAWDOWN")
    if not np.isfinite(ci) or ci <= 0: out.append("BLOCK_CI")
    return out


def v34_decision(v33_decision: dict[str, Any], lock: dict[str, Any], metrics: dict[str, Any] | None) -> dict[str, Any]:
    pre = prerequisite_failures(v33_decision, lock)
    if pre:
        state = "BLOCKED_BY_V33"
        failures: list[str] = []
        used = False
        reason = "Kraken was not touched because the preregistered v0.33 prerequisite was not satisfied."
    else:
        m = metrics or {}
        failures = holdout_failures(m)
        used = True
        dd = abs(_finite(m.get("max_drawdown")))
        n = int(m.get("trades", 0) or 0)
        if np.isfinite(dd) and dd > 0.05:
            state = "V34_CANDIDATE_REJECTED_KRAKEN_HARD_DD"
            reason = "The locked v0.33 winner breached the frozen 5% hard drawdown limit on untouched Kraken."
        elif n < 200:
            state = "V34_KRAKEN_EVIDENCE_INSUFFICIENT"
            reason = "Untouched Kraken has not produced the frozen minimum trade count."
        elif failures:
            state = "V34_CANDIDATE_REJECTED_KRAKEN"
            reason = "The locked v0.33 winner failed at least one unchanged Kraken holdout gate."
        else:
            state = "V34_CANDIDATE_PASSED_KRAKEN_REQUIRES_FRESH_TEMPORAL_OOS"
            reason = "The locked v0.33 winner passed untouched Kraken; strictly fresh temporal OOS remains mandatory."
    return {
        "version":"v0.34","decision":state,"reason":reason,
        "winner":v33_decision.get("winner"),"timeframe":v33_decision.get("timeframe"),
        "v33_prerequisite_failures":pre,"holdout_failures":failures,
        "final_holdout_venue":"kraken","holdout_used":used,
        "threshold_relaxation":False,"base_alpha_retuning":False,"winner_reselection":False,
        "forward_paper_candidate_authorized":False,"paper_replacement_authorized":False,
        "live_execution_authorized":False,
    }
