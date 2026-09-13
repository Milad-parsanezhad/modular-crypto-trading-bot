from __future__ import annotations

"""v0.27 cross-venue robustness selection and untouched-holdout qualification.

Scientific contract
-------------------
- CoinEx historical validation and OKX are already-consumed evidence and may be
  used only for development robustness ranking.
- Strategy definitions, v0.25 allocator parameters and final qualification
  thresholds are frozen; no parameter retuning is performed here.
- A single development winner is selected deterministically before KuCoin is
  inspected. KuCoin is the reserved final venue holdout for v0.27.
- Even a KuCoin pass does not authorize forward paper or live execution; a new
  post-lock temporal OOS phase is still required.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


DEVELOPMENT_VENUES_V27 = ("coinex_frozen", "okx_consumed")
FINAL_HOLDOUT_VENUE_V27 = "kucoin"


@dataclass(frozen=True)
class V27RobustnessPolicy:
    min_development_trades_per_venue: int = 200
    min_holdout_trades: int = 200
    min_profit_factor: float = 1.05
    min_positive_asset_fraction: float = 0.60
    max_drawdown: float = 0.05


def _number(metrics: dict[str, Any], key: str, *, default: float = np.nan) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _trades(metrics: dict[str, Any]) -> int:
    try:
        return int(metrics.get("external_trades", 0) or 0)
    except (TypeError, ValueError):
        return 0


def compact_external_metrics(raw: dict[str, Any], *, venue: str) -> dict[str, Any]:
    """Return stable, venue-prefixed metrics from v0.25 external evaluation."""
    def finite(key: str) -> float | None:
        value = _number(raw, key)
        return value if np.isfinite(value) else None

    return {
        "venue": str(venue),
        "trades": _trades(raw),
        "total_return": finite("external_total_return"),
        "profit_factor": finite("external_profit_factor"),
        "win_rate": finite("external_win_rate"),
        "expectancy_r": finite("external_expectancy_r"),
        "median_r": finite("external_median_r"),
        "max_drawdown": finite("external_max_drawdown"),
        "mean_account_return": finite("external_mean_account_return"),
        "positive_asset_fraction": finite("external_positive_asset_fraction"),
        "block_ci_low": finite("external_block_ci_low"),
        "block_ci_high": finite("external_block_ci_high"),
        "frozen_external_pass": bool(raw.get("external_pass_v25", False)),
    }


def _metric(compact: dict[str, Any], key: str, *, default: float = np.nan) -> float:
    try:
        value = float(compact.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def development_screen(
    coinex: dict[str, Any],
    okx: dict[str, Any],
    *,
    policy: V27RobustnessPolicy | None = None,
) -> dict[str, Any]:
    """Apply the preregistered two-venue development screen.

    Development eligibility deliberately uses only sample adequacy, positive
    economics and the frozen 5% safety boundary. Breadth and Block-CI are not
    relaxed qualification gates; they are used lexicographically to choose one
    candidate from already-consumed evidence, then the untouched KuCoin holdout
    is judged by the full frozen external gate.
    """
    p = policy or V27RobustnessPolicy()
    venues = (coinex, okx)

    trades = [int(v.get("trades", 0) or 0) for v in venues]
    pf = [_metric(v, "profit_factor") for v in venues]
    exp = [_metric(v, "expectancy_r") for v in venues]
    dd = [abs(_metric(v, "max_drawdown")) for v in venues]
    breadth = [_metric(v, "positive_asset_fraction", default=-np.inf) for v in venues]
    ci_low = [_metric(v, "block_ci_low", default=-np.inf) for v in venues]

    sample_ok = all(n >= p.min_development_trades_per_venue for n in trades)
    economics_ok = all(np.isfinite(x) and x >= p.min_profit_factor for x in pf) and all(
        np.isfinite(x) and x > 0.0 for x in exp
    )
    safety_ok = all(np.isfinite(x) and x <= p.max_drawdown for x in dd)

    return {
        "development_eligible_v27": bool(sample_ok and economics_ok and safety_ok),
        "development_sample_ok_v27": bool(sample_ok),
        "development_economics_ok_v27": bool(economics_ok),
        "development_safety_ok_v27": bool(safety_ok),
        "robust_min_trades_v27": int(min(trades)) if trades else 0,
        "robust_floor_profit_factor_v27": float(min(pf)) if pf else np.nan,
        "robust_floor_expectancy_r_v27": float(min(exp)) if exp else np.nan,
        "robust_floor_breadth_v27": float(min(breadth)) if breadth else np.nan,
        "robust_floor_block_ci_low_v27": float(min(ci_low)) if ci_low else np.nan,
        "robust_worst_drawdown_v27": float(max(dd)) if dd else np.nan,
    }


def select_development_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Lock one winner with a deterministic worst-venue lexicographic ranking."""
    eligible = [dict(row) for row in rows if bool(row.get("development_eligible_v27", False))]
    if not eligible:
        return None

    def finite_or(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    # max() is used. The final two keys therefore negate drawdown and strategy
    # lexical rank is handled by a stable sorted fallback below.
    eligible = sorted(eligible, key=lambda r: str(r.get("strategy", "")))
    return max(
        eligible,
        key=lambda r: (
            finite_or(r.get("robust_floor_block_ci_low_v27"), -np.inf),
            finite_or(r.get("robust_floor_breadth_v27"), -np.inf),
            finite_or(r.get("robust_floor_profit_factor_v27"), -np.inf),
            finite_or(r.get("robust_floor_expectancy_r_v27"), -np.inf),
            int(r.get("robust_min_trades_v27", 0) or 0),
            -finite_or(r.get("robust_worst_drawdown_v27"), np.inf),
        ),
    )


def holdout_failures(
    holdout: dict[str, Any],
    *,
    policy: V27RobustnessPolicy | None = None,
) -> list[str]:
    """Full unchanged final-venue gate used only after development lock."""
    p = policy or V27RobustnessPolicy()
    n = int(holdout.get("trades", 0) or 0)
    pf = _metric(holdout, "profit_factor")
    exp = _metric(holdout, "expectancy_r")
    breadth = _metric(holdout, "positive_asset_fraction")
    dd = abs(_metric(holdout, "max_drawdown"))
    ci = _metric(holdout, "block_ci_low")

    failures: list[str] = []
    if n < p.min_holdout_trades:
        failures.append("TRADES")
    if not np.isfinite(pf) or pf < p.min_profit_factor:
        failures.append("PROFIT_FACTOR")
    if not np.isfinite(exp) or exp <= 0:
        failures.append("EXPECTANCY")
    if not np.isfinite(breadth) or breadth < p.min_positive_asset_fraction:
        failures.append("BREADTH")
    if not np.isfinite(dd) or dd > p.max_drawdown:
        failures.append("MAX_DRAWDOWN")
    if not np.isfinite(ci) or ci <= 0:
        failures.append("BLOCK_CI")
    return failures


def v27_decision(
    winner: dict[str, Any] | None,
    holdout: dict[str, Any] | None,
    *,
    policy: V27RobustnessPolicy | None = None,
) -> dict[str, Any]:
    """Return v0.27 research state without ever authorizing live execution."""
    p = policy or V27RobustnessPolicy()
    if winner is None:
        state = "NO_ROBUST_DEVELOPMENT_CANDIDATE"
        reason = "No frozen candidate met the preregistered two-venue development sample/economics/safety screen; untouched KuCoin was not used for selection."
        failures: list[str] = []
        holdout_used = False
    elif holdout is None:
        state = "ROBUST_CANDIDATE_LOCKED_HOLDOUT_UNAVAILABLE"
        reason = "A single development winner was locked, but the reserved KuCoin holdout could not be evaluated without changing venue." 
        failures = ["HOLDOUT_UNAVAILABLE"]
        holdout_used = False
    else:
        failures = holdout_failures(holdout, policy=p)
        holdout_used = True
        try:
            dd = abs(float(holdout.get("max_drawdown")))
        except (TypeError, ValueError):
            dd = np.nan
        hard_dd = bool(np.isfinite(dd) and dd > p.max_drawdown)
        n = int(holdout.get("trades", 0) or 0)
        if hard_dd:
            state = "ROBUST_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked development winner breached the frozen 5% hard drawdown limit on untouched KuCoin."
        elif n < p.min_holdout_trades:
            state = "ROBUST_CANDIDATE_HOLDOUT_EVIDENCE_INSUFFICIENT"
            reason = "Untouched KuCoin has not produced the frozen minimum holdout trade count."
        elif failures:
            state = "ROBUST_CANDIDATE_REJECTED_HOLDOUT"
            reason = "The locked development winner failed at least one unchanged final KuCoin holdout gate."
        else:
            state = "ROBUST_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
            reason = "The locked development winner passed untouched KuCoin; a new strictly post-lock temporal OOS remains mandatory before forward-paper consideration."

    return {
        "version": "v0.27",
        "decision": state,
        "reason": reason,
        "winner": None if winner is None else str(winner.get("strategy")),
        "timeframe": None if winner is None else str(winner.get("timeframe")),
        "development_venues": list(DEVELOPMENT_VENUES_V27),
        "final_holdout_venue": FINAL_HOLDOUT_VENUE_V27,
        "holdout_used": holdout_used,
        "holdout_failures": failures,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "holdout_winner_reselection": False,
        "forward_paper_candidate_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
