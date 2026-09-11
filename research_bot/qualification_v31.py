from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from research_bot.cross_venue_robustness_v27 import V27RobustnessPolicy, development_screen, holdout_failures


def screen_v31_candidate(coinex: dict[str, Any], okx: dict[str, Any]) -> dict[str, Any]:
    return development_screen(coinex, okx, policy=V27RobustnessPolicy())


def select_v31_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v27", False))]
    if not eligible:
        return None

    def n(value: Any, default: float) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return default
        return x if np.isfinite(x) else default

    eligible = sorted(eligible, key=lambda r: str(r.get("strategy", "")))
    return max(
        eligible,
        key=lambda r: (
            n(r.get("robust_floor_block_ci_low_v27"), -np.inf),
            n(r.get("robust_floor_breadth_v27"), -np.inf),
            n(r.get("robust_floor_profit_factor_v27"), -np.inf),
            n(r.get("robust_floor_expectancy_r_v27"), -np.inf),
            int(r.get("robust_min_trades_v27", 0) or 0),
            -n(r.get("robust_worst_drawdown_v27"), np.inf),
        ),
    )


def v31_decision(
    winner: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
    *,
    holdout_available: bool = True,
) -> dict[str, Any]:
    policy = V27RobustnessPolicy()
    if winner is None:
        state = "NO_V31_FIREWALL_ROBUST_CANDIDATE"
        reason = "No frozen v0.30 alpha candidate passed CoinEx+OKX after the preregistered v0.31 drawdown firewall; KuCoin remained untouched."
        failures: list[str] = []
        holdout_used = False
    elif not holdout_available or holdout is None:
        state = "V31_FIREWALL_CANDIDATE_LOCKED_HOLDOUT_UNAVAILABLE"
        reason = "One v0.31 development winner was locked, but preregistered KuCoin could not be evaluated without changing venue."
        failures = ["HOLDOUT_UNAVAILABLE"]
        holdout_used = False
    else:
        failures = holdout_failures(holdout, policy=policy)
        holdout_used = True
        try:
            dd = abs(float(holdout.get("max_drawdown")))
        except (TypeError, ValueError):
            dd = np.nan
        n_trades = int(holdout.get("trades", 0) or 0)
        if np.isfinite(dd) and dd > policy.max_drawdown:
            state = "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked v0.31 winner breached the unchanged 5% hard drawdown gate on untouched KuCoin."
        elif n_trades < policy.min_holdout_trades:
            state = "V31_FIREWALL_HOLDOUT_EVIDENCE_INSUFFICIENT"
            reason = "Untouched KuCoin did not produce the frozen minimum holdout trade count."
        elif failures:
            state = "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked v0.31 winner failed at least one unchanged KuCoin holdout gate."
        else:
            state = "V31_FIREWALL_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
            reason = "The locked v0.31 winner passed untouched KuCoin; strictly post-lock temporal OOS remains mandatory."

    return {
        "version": "v0.31",
        "decision": state,
        "reason": reason,
        "winner": None if winner is None else str(winner.get("strategy")),
        "timeframe": None if winner is None else str(winner.get("timeframe")),
        "development_venues": ["coinex_consumed", "okx_consumed"],
        "final_holdout_venue": "kucoin",
        "holdout_used": holdout_used,
        "holdout_failures": failures,
        "hard_drawdown_cap": 0.05,
        "alpha_parameter_retuning": False,
        "qualification_threshold_relaxation": False,
        "winner_reselection": False,
        "historical_test_recycling": False,
        "forward_paper_candidate_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
