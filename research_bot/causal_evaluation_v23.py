from __future__ import annotations

"""Evaluation helpers for the v0.23 causal-risk replay.

The v0.20 metrics compounded account returns in ledger row order.  That row
order is not an economically valid equity path when positions overlap across
symbols.  v0.23 evaluates realized portfolio equity in exit-time order and
batches exits sharing a bar timestamp, while preserving the frozen v0.20 gate
thresholds wherever the statistic remains comparable.
"""

from typing import Iterable, Any

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import moving_block_mean_ci
from research_bot.multitimeframe_strategies_v20 import V20ValidationConfig, RiskPsychologyPolicy
from research_bot.risk_causality_v23 import apply_causal_risk_overlay_v23


def _empty_summary() -> dict[str, float | int]:
    return {
        "trades": 0,
        "total_return": np.nan,
        "profit_factor": np.nan,
        "win_rate": np.nan,
        "expectancy_r": np.nan,
        "median_r": np.nan,
        "max_drawdown": np.nan,
        "mean_account_return": np.nan,
        "settlement_batches": 0,
    }


def settlement_batch_returns_v23(trades: pd.DataFrame) -> pd.Series:
    """Return one realized portfolio factor-return per unique exit timestamp."""
    if trades.empty:
        return pd.Series(dtype=float, name="batch_return")
    required = {"exit_time", "account_return_v23"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"v0.23 settlement series missing columns: {sorted(missing)}")

    x = trades.copy()
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    x["account_return_v23"] = pd.to_numeric(x["account_return_v23"], errors="raise")
    if bool((x["account_return_v23"] <= -1.0).any()):
        raise ValueError("account return <= -100% is invalid for multiplicative settlement")

    grouped = x.sort_values(["exit_time", "symbol"], kind="mergesort").groupby("exit_time", sort=True)
    out = grouped["account_return_v23"].apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
    out.name = "batch_return"
    return out


def summarize_trades_v23(trades: pd.DataFrame) -> dict[str, float | int]:
    """Summarize executed trades using a causal, exit-time portfolio path."""
    if trades.empty:
        return _empty_summary()
    required = {"account_return_v23", "r_multiple", "exit_time"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"v0.23 summary missing columns: {sorted(missing)}")

    r = pd.to_numeric(trades["account_return_v23"], errors="raise").astype(float)
    rm = pd.to_numeric(trades["r_multiple"], errors="raise").astype(float)
    batches = settlement_batch_returns_v23(trades)

    equity = (1.0 + batches).cumprod()
    # Include the initial equity=1.0 in the running peak.  Without this,
    # an immediately losing first settlement would incorrectly report zero DD.
    peaks = np.maximum.accumulate(np.r_[1.0, equity.to_numpy(dtype=float)])
    equity_with_initial = np.r_[1.0, equity.to_numpy(dtype=float)]
    drawdown = equity_with_initial / peaks - 1.0

    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    total_return = float(np.prod(1.0 + r.to_numpy(dtype=float)) - 1.0)

    return {
        "trades": int(len(trades)),
        "total_return": total_return,
        "profit_factor": float(pf),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(rm.mean()),
        "median_r": float(rm.median()),
        "max_drawdown": float(np.min(drawdown)),
        "mean_account_return": float(r.mean()),
        "settlement_batches": int(len(batches)),
    }


def apply_candidate_level_policy_v23(
    attempts: pd.DataFrame,
    spec: Any,
    policy: RiskPsychologyPolicy | None = None,
) -> pd.DataFrame:
    """Apply the causal overlay independently inside frozen evaluation segments."""
    if attempts.empty or "segment" not in attempts:
        return apply_causal_risk_overlay_v23(attempts, spec, policy)

    parts: list[pd.DataFrame] = []
    for segment in ("development", "validation", "test"):
        part = attempts[attempts["segment"] == segment].copy()
        if not part.empty:
            parts.append(apply_causal_risk_overlay_v23(part, spec, policy))
    if not parts:
        return attempts.copy()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def _executed(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    if "executed_v23" not in attempts:
        raise ValueError("v0.23 evaluation requires executed_v23 from causal overlay")
    return attempts[attempts["executed_v23"] == True].copy()  # noqa: E712


def _asset_positive_fraction(executed: pd.DataFrame) -> float:
    if executed.empty:
        return 0.0
    # Preserve the v0.20 breadth definition (sum of account returns by asset)
    # so the only intended methodological change is time-causal risk/equity.
    assets = executed.groupby("symbol")["account_return_v23"].sum()
    return float((assets > 0).mean()) if len(assets) else 0.0


def _causal_bootstrap_series(executed: pd.DataFrame) -> Iterable[float]:
    # Blocks follow realized exit-time batches, not entry-row order.  Portfolio
    # batch returns are the observable time series once positions are settled.
    return settlement_batch_returns_v23(executed).to_numpy(dtype=float)


def evaluate_v23_candidate(
    attempts: pd.DataFrame,
    *,
    total_trials: int,
    validation: V20ValidationConfig | None = None,
) -> dict[str, object]:
    """Evaluate a candidate with frozen v0.20 thresholds and causal accounting."""
    c = validation or V20ValidationConfig()
    executed = _executed(attempts)
    out: dict[str, object] = {
        "attempted_trades": int(len(attempts)),
        "executed_trades": int(len(executed)),
        "risk_rejected_trades": int(len(attempts) - len(executed)),
    }
    for seg in ("development", "validation", "test"):
        part = executed[executed["segment"] == seg] if (not executed.empty and "segment" in executed) else executed
        out.update({f"{seg}_{k}": v for k, v in summarize_trades_v23(part).items()})

    pre_attempts = attempts[attempts["segment"].isin(["development", "validation"])] if (not attempts.empty and "segment" in attempts) else attempts
    val = executed[executed["segment"] == "validation"] if (not executed.empty and "segment" in executed) else executed
    out["pretest_signal_trades"] = int(len(pre_attempts))
    out["pretest_executed_trades"] = int(len(executed[executed["segment"].isin(["development", "validation"])]) if (not executed.empty and "segment" in executed) else len(executed))
    out["pretest_trades"] = int(len(pre_attempts))
    out["validation_positive_asset_fraction"] = _asset_positive_fraction(val)

    block_values = list(_causal_bootstrap_series(val)) if len(val) else []
    lo, hi = moving_block_mean_ci(block_values, samples=c.bootstrap_samples, block=c.bootstrap_block, seed=c.random_seed)
    out["validation_block_ci_low"] = lo
    out["validation_block_ci_high"] = hi

    # Multiplicity statistic remains a trade-level mean test; order does not
    # enter the test itself, so use the causal risk-scaled realized returns.
    values = pd.to_numeric(val["account_return_v23"], errors="coerce").dropna().to_numpy(dtype=float) if len(val) else np.asarray([], dtype=float)
    if len(values) < 30 or np.std(values, ddof=1) <= 0:
        p_adj = 1.0
    else:
        from math import erf, sqrt
        z = float(np.mean(values) / (np.std(values, ddof=1) / np.sqrt(len(values))))
        normal_cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        p_adj = float(min(1.0, max(0.0, 1.0 - normal_cdf) * max(1, total_trials)))
    out["validation_multiplicity_adjusted_p"] = p_adj

    pf = float(out.get("validation_profit_factor", np.nan))
    exp = float(out.get("validation_expectancy_r", np.nan))
    dd = abs(float(out.get("validation_max_drawdown", np.nan)))
    n = int(out.get("validation_trades", 0))
    out["validation_score_v23"] = float(exp * np.sqrt(max(n, 1)) + 0.10 * (min(pf, 5.0) - 1.0) - 0.75 * dd) if np.isfinite(exp) and np.isfinite(pf) and np.isfinite(dd) else -np.inf
    out["internal_eligible_v23"] = bool(
        len(pre_attempts) >= c.min_pretest_trades
        and n >= c.min_validation_trades
        and np.isfinite(pf) and pf >= c.min_profit_factor
        and np.isfinite(exp) and exp > 0
        and float(out["validation_positive_asset_fraction"]) >= c.min_positive_asset_fraction
        and np.isfinite(dd) and dd <= c.max_drawdown
        and np.isfinite(lo) and lo > 0
        and p_adj <= c.max_multiplicity_p
    )
    return out


def evaluate_external_replication_v23(
    attempts: pd.DataFrame,
    validation: V20ValidationConfig | None = None,
) -> dict[str, object]:
    """Apply the frozen external gate to causally settled portfolio evidence."""
    c = validation or V20ValidationConfig()
    executed = _executed(attempts)
    metrics = summarize_trades_v23(executed)
    breadth = _asset_positive_fraction(executed)
    block_values = list(_causal_bootstrap_series(executed)) if len(executed) else []
    lo, hi = moving_block_mean_ci(block_values, samples=c.bootstrap_samples, block=c.bootstrap_block, seed=c.random_seed + 1)
    passed = bool(
        int(metrics["trades"]) >= c.min_external_trades
        and np.isfinite(float(metrics["profit_factor"])) and float(metrics["profit_factor"]) >= c.min_profit_factor
        and np.isfinite(float(metrics["expectancy_r"])) and float(metrics["expectancy_r"]) > 0
        and breadth >= c.min_positive_asset_fraction
        and np.isfinite(float(metrics["max_drawdown"])) and abs(float(metrics["max_drawdown"])) <= c.max_drawdown
        and np.isfinite(lo) and lo > 0
    )
    return {
        **{f"external_{k}": v for k, v in metrics.items()},
        "external_positive_asset_fraction": breadth,
        "external_block_ci_low": lo,
        "external_block_ci_high": hi,
        "external_pass_v23": passed,
    }
