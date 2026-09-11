from __future__ import annotations

"""Frozen v0.30 qualification state machine.

The development screen uses only consumed CoinEx + OKX evidence. KuCoin is
reserved and may be evaluated only after one development winner is locked.
"""

from typing import Any, Iterable

import numpy as np

from research_bot.cross_venue_robustness_v27 import V27RobustnessPolicy, development_screen, holdout_failures
from research_bot.regime_event_alpha_v30 import DEVELOPMENT_VENUES_V30, FINAL_HOLDOUT_VENUE_V30


def screen_v30_candidate(coinex: dict[str, Any], okx: dict[str, Any]) -> dict[str, Any]:
    return development_screen(coinex, okx, policy=V27RobustnessPolicy())


def select_v30_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v27", False))]
    if not eligible:
        return None

    def number(value: Any, default: float) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return default
        return x if np.isfinite(x) else default

    # Stable lexical ordering is only the final deterministic tie breaker.
    eligible = sorted(eligible, key=lambda r: str(r.get("strategy", "")))
    return max(
        eligible,
        key=lambda r: (
            number(r.get("robust_floor_block_ci_low_v27"), -np.inf),
            number(r.get("robust_floor_breadth_v27"), -np.inf),
            number(r.get("robust_floor_profit_factor_v27"), -np.inf),
            number(r.get("robust_floor_expectancy_r_v27"), -np.inf),
            int(r.get("robust_min_trades_v27", 0) or 0),
            -number(r.get("robust_worst_drawdown_v27"), np.inf),
        ),
    )


def v30_decision(
    winner: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
    *,
    holdout_available: bool = True,
) -> dict[str, Any]:
    policy = V27RobustnessPolicy()
    if winner is None:
        state = "NO_V30_ROBUST_DEVELOPMENT_CANDIDATE"
        reason = "No preregistered v0.30 candidate passed the frozen CoinEx+OKX development robustness screen; KuCoin remained untouched."
        failures: list[str] = []
        holdout_used = False
    elif not holdout_available or holdout is None:
        state = "V30_CANDIDATE_LOCKED_HOLDOUT_UNAVAILABLE"
        reason = "One v0.30 development winner was locked, but the preregistered KuCoin holdout could not be evaluated without changing venue."
        failures = ["HOLDOUT_UNAVAILABLE"]
        holdout_used = False
    else:
        failures = holdout_failures(holdout, policy=policy)
        holdout_used = True
        try:
            dd = abs(float(holdout.get("max_drawdown")))
        except (TypeError, ValueError):
            dd = np.nan
        hard_dd = bool(np.isfinite(dd) and dd > policy.max_drawdown)
        n = int(holdout.get("trades", 0) or 0)
        if hard_dd:
            state = "V30_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked v0.30 winner breached the frozen 5% hard drawdown limit on untouched KuCoin."
        elif n < policy.min_holdout_trades:
            state = "V30_CANDIDATE_HOLDOUT_EVIDENCE_INSUFFICIENT"
            reason = "Untouched KuCoin did not produce the frozen minimum holdout trade count."
        elif failures:
            state = "V30_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked v0.30 winner failed at least one unchanged final KuCoin holdout gate."
        else:
            state = "V30_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
            reason = "The locked v0.30 winner passed untouched KuCoin; a strictly post-lock temporal OOS phase remains mandatory."

    return {
        "version": "v0.30",
        "decision": state,
        "reason": reason,
        "winner": None if winner is None else str(winner.get("strategy")),
        "timeframe": None if winner is None else str(winner.get("timeframe")),
        "development_venues": list(DEVELOPMENT_VENUES_V30),
        "final_holdout_venue": FINAL_HOLDOUT_VENUE_V30,
        "holdout_used": holdout_used,
        "holdout_failures": failures,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "winner_reselection": False,
        "historical_test_recycling": False,
        "forward_paper_candidate_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
