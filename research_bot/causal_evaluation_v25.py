from __future__ import annotations

"""v0.25 evaluation adapter.

The portfolio allocator changes position sizing, not the frozen qualification
statistics. To avoid silently changing the tournament twice, this module maps
v0.25 accounting columns into the already-audited v0.23 causal evaluator, then
renames version-specific outputs.

The v0.25 tournament may deliberately omit the historical ``test`` segment from
selection. The existing test slice has already been inspected in earlier
research generations and is therefore not treated as a pristine holdout.
"""

from typing import Any

import pandas as pd

from research_bot.causal_evaluation_v23 import evaluate_v23_candidate, evaluate_external_replication_v23
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25, apply_portfolio_allocator_v25


V25_TO_V23_COLUMNS = {
    "executed_v25": "executed_v23",
    "account_return_v25": "account_return_v23",
    "settlement_batch_return_v25": "settlement_batch_return_v23",
}


def apply_candidate_level_policy_v25(
    attempts: pd.DataFrame,
    spec: Any,
    *,
    risk_policy: RiskPsychologyPolicy | None = None,
    budget: PortfolioRiskBudgetV25 | None = None,
) -> pd.DataFrame:
    """Reset allocator state independently inside each frozen data segment."""

    if attempts.empty or "segment" not in attempts:
        return apply_portfolio_allocator_v25(attempts, spec, risk_policy=risk_policy, budget=budget)

    parts: list[pd.DataFrame] = []
    known_order = ("development", "validation", "test")
    observed = set(attempts["segment"].astype(str))
    ordered = [segment for segment in known_order if segment in observed]
    ordered += sorted(observed - set(ordered))
    for segment in ordered:
        part = attempts[attempts["segment"].astype(str) == segment].copy()
        if not part.empty:
            parts.append(
                apply_portfolio_allocator_v25(
                    part,
                    spec,
                    risk_policy=risk_policy,
                    budget=budget,
                )
            )
    if not parts:
        return attempts.copy()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def _compat_v23(attempts: pd.DataFrame) -> pd.DataFrame:
    x = attempts.copy()
    missing = set(V25_TO_V23_COLUMNS) - set(x.columns)
    if missing:
        raise ValueError(f"v0.25 evaluation missing accounting columns: {sorted(missing)}")
    for source, target in V25_TO_V23_COLUMNS.items():
        x[target] = x[source]
    return x


def evaluate_v25_candidate(
    attempts: pd.DataFrame,
    *,
    total_trials: int,
    validation: V20ValidationConfig | None = None,
) -> dict[str, object]:
    """Evaluate v0.25 with the exact frozen v0.23/v0.20 gate definitions."""

    raw = evaluate_v23_candidate(_compat_v23(attempts), total_trials=total_trials, validation=validation)
    out = dict(raw)
    out["validation_score_v25"] = out.pop("validation_score_v23")
    out["internal_eligible_v25"] = out.pop("internal_eligible_v23")
    out["allocator_scaled_trades"] = int(
        ((attempts.get("executed_v25", False) == True) & (attempts.get("portfolio_allocation_scale_v25", 1.0) < 1.0 - 1e-12)).sum()  # noqa: E712
    ) if not attempts.empty else 0
    out["allocator_zero_budget_rejections"] = int(
        attempts.get("reject_reason_v25", pd.Series(dtype=str)).astype(str).eq("PORTFOLIO_RISK_BUDGET_EXHAUSTED").sum()
    ) if not attempts.empty else 0
    return out


def evaluate_external_replication_v25(
    attempts: pd.DataFrame,
    validation: V20ValidationConfig | None = None,
) -> dict[str, object]:
    raw = evaluate_external_replication_v23(_compat_v23(attempts), validation=validation)
    out = dict(raw)
    out["external_pass_v25"] = out.pop("external_pass_v23")
    return out
