from __future__ import annotations

"""Re-rank all frozen v0.20 candidates under v0.23 causal accounting.

The input is the exact v0.20 trade-attempt artifact. Signal generation, costs,
segments and candidate registry are held fixed. Every candidate's risk state is
replayed using event-time settlements and entry-equity-fixed cash P&L.

This is the required internal correction after the one-strategy forensic replay.
It intentionally performs no live execution and no threshold relaxation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v23 import apply_candidate_level_policy_v23, evaluate_v23_candidate
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    V20ValidationConfig,
    RiskPsychologyPolicy,
)


FROZEN_V20_RUN_ID = 34482338133
FROZEN_V20_ARTIFACT = "v20-global-strategy-risk-lab"


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
    required = {"strategy", "symbol", "entry_time", "exit_time", "r_multiple", "risk_scale_volatility", "segment"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frozen trade ledger missing replay columns: {sorted(missing)}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.23 full frozen-artifact causal tournament")
    parser.add_argument("--v20-artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v23-full-causal-tournament")
    args = parser.parse_args()

    artifact_dir = Path(args.v20_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    attempts = _load_attempts(artifact_dir)
    config = V20ValidationConfig()
    policy = RiskPsychologyPolicy()
    registry = {spec.name: spec for spec in STRATEGY_REGISTRY_V20}
    total_trials = len(STRATEGY_REGISTRY_V20)

    observed = set(attempts["strategy"].astype(str).unique())
    expected = set(registry)
    missing_candidates = sorted(expected - observed)
    unexpected_candidates = sorted(observed - expected)
    if missing_candidates or unexpected_candidates:
        raise ValueError(
            f"frozen candidate registry mismatch: missing={missing_candidates}, unexpected={unexpected_candidates}"
        )

    summaries: list[dict[str, Any]] = []
    top_ledger_cache: dict[str, pd.DataFrame] = {}
    for strategy in [spec.name for spec in STRATEGY_REGISTRY_V20]:
        spec = registry[strategy]
        raw = attempts[attempts["strategy"].astype(str) == strategy].copy()
        replayed = apply_candidate_level_policy_v23(raw, spec, policy=policy)
        metrics = evaluate_v23_candidate(replayed, total_trials=total_trials, validation=config)
        gates = _gate_columns(metrics, config)
        summaries.append(
            {
                "strategy": strategy,
                "timeframe": spec.timeframe,
                "family": spec.family,
                "source_basis": spec.source_basis,
                **metrics,
                **gates,
            }
        )
        # Cache only viable-ish candidates; final top ledgers are selected after ranking.
        if float(metrics.get("validation_score_v23", -np.inf)) > 0:
            top_ledger_cache[strategy] = replayed

    summary = pd.DataFrame(summaries).sort_values(
        ["internal_eligible_v23", "validation_score_v23"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.to_csv(output / "candidate_summary_v23.csv", index=False)
    summary.head(15).to_csv(output / "top15_candidates_v23.csv", index=False)
    summary[summary["internal_eligible_v23"] == True].to_csv(output / "eligible_candidates_v23.csv", index=False)  # noqa: E712

    top_names = summary.head(5)["strategy"].astype(str).tolist()
    for name in top_names:
        ledger = top_ledger_cache.get(name)
        if ledger is None:
            spec = registry[name]
            raw = attempts[attempts["strategy"].astype(str) == name].copy()
            ledger = apply_candidate_level_policy_v23(raw, spec, policy=policy)
        ledger.to_csv(output / f"top5_{name}_causal_ledger.csv", index=False)

    eligible = summary[summary["internal_eligible_v23"] == True].copy()  # noqa: E712
    if eligible.empty:
        decision = {
            "version": "v0.23",
            "decision": "NO_CAUSAL_INTERNAL_CANDIDATE",
            "reason": "None of the 42 frozen candidates clears every frozen internal gate after event-driven causal risk replay.",
            "winner": None,
            "candidate_count": total_trials,
            "eligible_count": 0,
            "threshold_relaxation": False,
            "external_replication_required": False,
            "forward_paper_candidate_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
    else:
        winner = eligible.sort_values("validation_score_v23", ascending=False).iloc[0]
        decision = {
            "version": "v0.23",
            "decision": "CAUSAL_INTERNAL_WINNER_REQUIRES_EXTERNAL_REPLICATION",
            "reason": "At least one candidate clears frozen internal gates; external-venue replication is mandatory before any forward-paper decision.",
            "winner": str(winner["strategy"]),
            "timeframe": str(winner["timeframe"]),
            "candidate_count": total_trials,
            "eligible_count": int(len(eligible)),
            "threshold_relaxation": False,
            "external_replication_required": True,
            "forward_paper_candidate_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }

    failure_counts: dict[str, int] = {}
    for cell in summary["failed_gates"].fillna("").astype(str):
        for reason in [r for r in cell.split("|") if r]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    manifest = {
        "version": "v0.23",
        "experiment": "FULL_42_CANDIDATE_FROZEN_ARTIFACT_CAUSAL_RERANK",
        "source": {
            "workflow_run_id": FROZEN_V20_RUN_ID,
            "artifact_name": FROZEN_V20_ARTIFACT,
            "input_contract": "frozen v0.20 generated trade attempts; only causal risk/equity accounting is replayed",
        },
        "candidate_count": total_trials,
        "eligible_count": int(len(eligible)),
        "top5": summary.head(5)[
            [
                "strategy", "timeframe", "validation_score_v23", "validation_trades",
                "validation_profit_factor", "validation_expectancy_r", "validation_max_drawdown",
                "validation_positive_asset_fraction", "validation_block_ci_low",
                "validation_multiplicity_adjusted_p", "failed_gates",
            ]
        ].to_dict(orient="records"),
        "failure_counts": failure_counts,
        "decision": decision,
    }
    (output / "decision_v23.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "tournament_manifest_v23.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
