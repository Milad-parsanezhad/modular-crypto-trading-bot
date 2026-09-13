from __future__ import annotations

"""v0.25 tournament with preregistered pre-entry portfolio risk allocation.

Input contract
--------------
Consumes the exact frozen v0.20 trade-attempt artifact. Signal generation,
transaction-cost-adjusted R outcomes, symbols and validation gates are held
fixed. The historical ``test`` segment is intentionally excluded from model
selection because earlier research generations have already inspected it.

The only new treatment is the preregistered v0.25 allocator:
- 2.0% aggregate cost-adjusted open loss-at-stop budget;
- 1.5% same-direction cost-adjusted budget;
- same-timestamp proposals share budget pro-rata.

No threshold relaxation. No live execution.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v23 import apply_candidate_level_policy_v23, evaluate_v23_candidate
from research_bot.causal_evaluation_v25 import apply_candidate_level_policy_v25, evaluate_v25_candidate
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
    V20ValidationConfig,
)
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25


FROZEN_V20_RUN_ID = 34482338133
FROZEN_V20_ARTIFACT = "v20-global-strategy-risk-lab"
CURRENT_CANDIDATES = len(STRATEGY_REGISTRY_V20)
# Conservative research-family accounting: the 42 original candidate rules and
# their 42 v0.25 portfolio-allocated variants are treated as 84 effective trials.
EFFECTIVE_MULTIPLICITY_TRIALS = CURRENT_CANDIDATES * 2


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_attempts(artifact_dir: Path) -> pd.DataFrame:
    path = artifact_dir / "trade_attempt_ledger.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen v0.20 ledger: {path}")
    frame = pd.read_csv(path)
    for column in ("signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    required = {
        "strategy", "symbol", "entry_time", "exit_time", "r_multiple", "side",
        "entry", "stop", "risk_scale_volatility", "segment",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frozen v0.20 ledger missing v0.25 replay columns: {sorted(missing)}")
    return frame


def _gate_columns(metrics: dict[str, Any], config: V20ValidationConfig) -> dict[str, Any]:
    def finite(key: str) -> bool:
        try:
            return bool(np.isfinite(float(metrics.get(key, np.nan))))
        except (TypeError, ValueError):
            return False

    pretest = int(metrics.get("pretest_trades", 0))
    validation_trades = int(metrics.get("validation_trades", 0))
    pf = float(metrics.get("validation_profit_factor", np.nan))
    exp = float(metrics.get("validation_expectancy_r", np.nan))
    breadth = float(metrics.get("validation_positive_asset_fraction", 0.0))
    dd = abs(float(metrics.get("validation_max_drawdown", np.nan)))
    ci_low = float(metrics.get("validation_block_ci_low", np.nan))
    p_adj = float(metrics.get("validation_multiplicity_adjusted_p", np.nan))

    gates = {
        "gate_pretest_trades": pretest >= config.min_pretest_trades,
        "gate_validation_trades": validation_trades >= config.min_validation_trades,
        "gate_profit_factor": finite("validation_profit_factor") and pf >= config.min_profit_factor,
        "gate_expectancy": finite("validation_expectancy_r") and exp > 0,
        "gate_breadth": finite("validation_positive_asset_fraction") and breadth >= config.min_positive_asset_fraction,
        "gate_max_drawdown": finite("validation_max_drawdown") and dd <= config.max_drawdown,
        "gate_block_ci": finite("validation_block_ci_low") and ci_low > 0,
        "gate_multiplicity": finite("validation_multiplicity_adjusted_p") and p_adj <= config.max_multiplicity_p,
    }
    failures = [name.removeprefix("gate_").upper() for name, passed in gates.items() if not passed]
    return {**gates, "failed_gate_count": len(failures), "failed_gates": "|".join(failures)}


def _allocator_diagnostics(replayed: pd.DataFrame) -> dict[str, Any]:
    if replayed.empty:
        return {
            "max_open_risk_fraction_after_v25": 0.0,
            "max_directional_risk_fraction_after_v25": 0.0,
            "validation_max_open_risk_fraction_after_v25": 0.0,
            "validation_max_directional_risk_fraction_after_v25": 0.0,
            "validation_inherited_aggregate_overage_rows": 0,
            "validation_inherited_directional_overage_rows": 0,
        }

    budget = PortfolioRiskBudgetV25()
    val = replayed[replayed["segment"].astype(str) == "validation"].copy()

    def maximum(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return 0.0
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.max()) if len(values) else 0.0

    return {
        "max_open_risk_fraction_after_v25": maximum(replayed, "open_risk_fraction_after_v25"),
        "max_directional_risk_fraction_after_v25": maximum(replayed, "open_directional_risk_fraction_after_v25"),
        "validation_max_open_risk_fraction_after_v25": maximum(val, "open_risk_fraction_after_v25"),
        "validation_max_directional_risk_fraction_after_v25": maximum(val, "open_directional_risk_fraction_after_v25"),
        "validation_inherited_aggregate_overage_rows": int(
            (pd.to_numeric(val.get("open_risk_fraction_before_v25", pd.Series(dtype=float)), errors="coerce") > budget.aggregate_open_risk_cap + 1e-12).sum()
        ),
        "validation_inherited_directional_overage_rows": int(
            (pd.to_numeric(val.get("open_directional_risk_fraction_before_v25", pd.Series(dtype=float)), errors="coerce") > budget.directional_open_risk_cap + 1e-12).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.25 portfolio allocator tournament")
    parser.add_argument("--v20-artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v25-portfolio-allocator-tournament")
    args = parser.parse_args()

    artifact_dir = Path(args.v20_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_attempts = _load_attempts(artifact_dir)
    # Historical test data is contaminated by prior inspection and is excluded
    # from v0.25 selection. Keep only development + validation for this stage.
    attempts = all_attempts[all_attempts["segment"].isin(["development", "validation"])].copy()
    excluded_test_rows = int((all_attempts["segment"] == "test").sum())

    risk_policy = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    validation = V20ValidationConfig()
    registry = {spec.name: spec for spec in STRATEGY_REGISTRY_V20}

    observed = set(attempts["strategy"].astype(str).unique())
    expected = set(registry)
    if observed != expected:
        raise ValueError(
            f"candidate registry mismatch after dev/validation filtering: missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}"
        )

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    for spec in STRATEGY_REGISTRY_V20:
        raw = attempts[attempts["strategy"].astype(str) == spec.name].copy()
        replayed = apply_candidate_level_policy_v25(raw, spec, risk_policy=risk_policy, budget=budget)
        metrics = evaluate_v25_candidate(
            replayed,
            total_trials=EFFECTIVE_MULTIPLICITY_TRIALS,
            validation=validation,
        )
        gates = _gate_columns(metrics, validation)
        diagnostics = _allocator_diagnostics(replayed)
        rows.append(
            {
                "strategy": spec.name,
                "timeframe": spec.timeframe,
                "family": spec.family,
                "source_basis": spec.source_basis,
                **metrics,
                **gates,
                **diagnostics,
            }
        )
        ledgers[spec.name] = replayed

    summary = pd.DataFrame(rows).sort_values(
        ["internal_eligible_v25", "validation_score_v25"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.to_csv(output / "candidate_summary_v25.csv", index=False)
    summary.head(15).to_csv(output / "top15_candidates_v25.csv", index=False)
    eligible = summary[summary["internal_eligible_v25"] == True].copy()  # noqa: E712
    eligible.to_csv(output / "eligible_candidates_v25.csv", index=False)

    for strategy in summary.head(5)["strategy"].astype(str):
        ledgers[strategy].to_csv(output / f"top5_{strategy}_allocator_ledger.csv", index=False)

    # Controlled H4_S6 comparison: same dev/validation trade attempts, v0.23
    # accounting vs v0.25 allocator. This is diagnostic only, not a second gate.
    h4_spec = registry["H4_S6_BREAKOUT"]
    h4_raw = attempts[attempts["strategy"].astype(str) == "H4_S6_BREAKOUT"].copy()
    h4_v23 = apply_candidate_level_policy_v23(h4_raw, h4_spec, policy=risk_policy)
    h4_v23_metrics = evaluate_v23_candidate(h4_v23, total_trials=CURRENT_CANDIDATES, validation=validation)
    h4_v25 = ledgers["H4_S6_BREAKOUT"]
    h4_v25_metrics = evaluate_v25_candidate(
        h4_v25,
        total_trials=EFFECTIVE_MULTIPLICITY_TRIALS,
        validation=validation,
    )
    h4_comparison = {
        "validation_trades_v23": h4_v23_metrics.get("validation_trades"),
        "validation_trades_v25": h4_v25_metrics.get("validation_trades"),
        "validation_max_drawdown_v23": h4_v23_metrics.get("validation_max_drawdown"),
        "validation_max_drawdown_v25": h4_v25_metrics.get("validation_max_drawdown"),
        "validation_profit_factor_v23": h4_v23_metrics.get("validation_profit_factor"),
        "validation_profit_factor_v25": h4_v25_metrics.get("validation_profit_factor"),
        "validation_block_ci_low_v23": h4_v23_metrics.get("validation_block_ci_low"),
        "validation_block_ci_low_v25": h4_v25_metrics.get("validation_block_ci_low"),
        "validation_multiplicity_p_v23": h4_v23_metrics.get("validation_multiplicity_adjusted_p"),
        "validation_multiplicity_p_v25": h4_v25_metrics.get("validation_multiplicity_adjusted_p"),
        "allocator_scaled_trades_v25": h4_v25_metrics.get("allocator_scaled_trades"),
        "allocator_zero_budget_rejections_v25": h4_v25_metrics.get("allocator_zero_budget_rejections"),
    }
    (output / "h4_s6_v23_vs_v25.json").write_text(
        json.dumps(_safe(h4_comparison), indent=2, sort_keys=True), encoding="utf-8"
    )

    if eligible.empty:
        decision = {
            "version": "v0.25",
            "decision": "NO_PORTFOLIO_ALLOCATED_INTERNAL_CANDIDATE",
            "reason": "None of the 42 candidates clears every frozen internal gate after preregistered pre-entry portfolio allocation.",
            "winner": None,
            "candidate_count": CURRENT_CANDIDATES,
            "effective_multiplicity_trials": EFFECTIVE_MULTIPLICITY_TRIALS,
            "eligible_count": 0,
            "historical_test_segment_used_for_selection": False,
            "external_replication_required": False,
            "fresh_temporal_oos_required": True,
            "threshold_relaxation": False,
            "forward_paper_candidate_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
    else:
        winner = eligible.sort_values("validation_score_v25", ascending=False).iloc[0]
        decision = {
            "version": "v0.25",
            "decision": "PORTFOLIO_ALLOCATED_INTERNAL_WINNER_REQUIRES_FRESH_REPLICATION",
            "reason": "At least one candidate clears frozen internal gates. It must pass a newly fetched external-venue replication and a fresh temporal OOS period before forward-paper consideration.",
            "winner": str(winner["strategy"]),
            "timeframe": str(winner["timeframe"]),
            "candidate_count": CURRENT_CANDIDATES,
            "effective_multiplicity_trials": EFFECTIVE_MULTIPLICITY_TRIALS,
            "eligible_count": int(len(eligible)),
            "historical_test_segment_used_for_selection": False,
            "external_replication_required": True,
            "fresh_temporal_oos_required": True,
            "threshold_relaxation": False,
            "forward_paper_candidate_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }

    failure_counts: dict[str, int] = {}
    for cell in summary["failed_gates"].fillna("").astype(str):
        for reason in [item for item in cell.split("|") if item]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    manifest = {
        "version": "v0.25",
        "experiment": "PREREGISTERED_PRE_ENTRY_PORTFOLIO_ALLOCATOR_TOURNAMENT",
        "source": {
            "workflow_run_id": FROZEN_V20_RUN_ID,
            "artifact_name": FROZEN_V20_ARTIFACT,
            "frozen_input": True,
            "excluded_historical_test_rows": excluded_test_rows,
        },
        "portfolio_budget": {
            "aggregate_open_risk_cap": budget.aggregate_open_risk_cap,
            "directional_open_risk_cap": budget.directional_open_risk_cap,
            "round_trip_cost_fraction": budget.round_trip_cost_fraction,
        },
        "candidate_count": CURRENT_CANDIDATES,
        "effective_multiplicity_trials": EFFECTIVE_MULTIPLICITY_TRIALS,
        "eligible_count": int(len(eligible)),
        "failure_counts": failure_counts,
        "top5": summary.head(5)[
            [
                "strategy", "timeframe", "validation_score_v25", "validation_trades",
                "validation_profit_factor", "validation_expectancy_r", "validation_max_drawdown",
                "validation_positive_asset_fraction", "validation_block_ci_low",
                "validation_multiplicity_adjusted_p", "allocator_scaled_trades",
                "allocator_zero_budget_rejections", "validation_max_open_risk_fraction_after_v25",
                "validation_max_directional_risk_fraction_after_v25", "failed_gates",
            ]
        ].to_dict(orient="records"),
        "h4_s6_v23_vs_v25": h4_comparison,
        "decision": decision,
    }
    (output / "decision_v25.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "tournament_manifest_v25.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
