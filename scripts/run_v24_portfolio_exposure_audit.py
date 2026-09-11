from __future__ import annotations

"""Audit concurrent portfolio exposure for all frozen v0.20 candidates.

This diagnostic consumes the exact v0.20 trade-attempt artifact, replays each
candidate with the corrected v0.23 causal risk engine, then measures concurrent
open positions and risk-at-stop. It does not alter signals, gates, or risk
thresholds and cannot authorize paper/live trading.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v23 import apply_candidate_level_policy_v23, evaluate_v23_candidate
from research_bot.multitimeframe_strategies_v20 import STRATEGY_REGISTRY_V20, RiskPsychologyPolicy, V20ValidationConfig
from research_bot.portfolio_exposure_v24 import exposure_path_v24, summarize_exposure_v24


FROZEN_V20_RUN_ID = 34482338133
FROZEN_V20_ARTIFACT = "v20-global-strategy-risk-lab"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.24 concurrent portfolio exposure audit")
    parser.add_argument("--v20-artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v24-portfolio-exposure-audit")
    args = parser.parse_args()

    artifact_dir = Path(args.v20_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    attempts = _load_attempts(artifact_dir)
    policy = RiskPsychologyPolicy()
    validation = V20ValidationConfig()
    total_trials = len(STRATEGY_REGISTRY_V20)
    rows: list[dict[str, Any]] = []
    top_paths: dict[str, pd.DataFrame] = {}

    for spec in STRATEGY_REGISTRY_V20:
        raw = attempts[attempts["strategy"].astype(str) == spec.name].copy()
        if raw.empty:
            raise ValueError(f"frozen artifact has no attempts for {spec.name}")
        replayed = apply_candidate_level_policy_v23(raw, spec, policy=policy)
        metrics = evaluate_v23_candidate(replayed, total_trials=total_trials, validation=validation)
        path = exposure_path_v24(replayed)
        exposure = summarize_exposure_v24(path)

        validation_path = path[path["segment"].astype(str) == "validation"].copy() if not path.empty else path
        validation_exposure = summarize_exposure_v24(validation_path)
        test_path = path[path["segment"].astype(str) == "test"].copy() if not path.empty else path
        test_exposure = summarize_exposure_v24(test_path)

        rows.append(
            {
                "strategy": spec.name,
                "timeframe": spec.timeframe,
                "family": spec.family,
                "internal_eligible_v23": bool(metrics.get("internal_eligible_v23", False)),
                "validation_score_v23": metrics.get("validation_score_v23"),
                "validation_trades_v23": metrics.get("validation_trades"),
                "validation_max_drawdown_v23": metrics.get("validation_max_drawdown"),
                **{f"all_{k}": v for k, v in exposure.items()},
                **{f"validation_{k}": v for k, v in validation_exposure.items()},
                **{f"test_{k}": v for k, v in test_exposure.items()},
            }
        )
        top_paths[spec.name] = path

    summary = pd.DataFrame(rows).sort_values(
        ["validation_max_open_risk_fraction", "validation_max_open_positions"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary.to_csv(output / "portfolio_exposure_summary_v24.csv", index=False)

    # The most relevant view for redesign is the v0.23 ranking, not merely the
    # highest-risk candidates. Preserve both rankings in the audit artifact.
    ranked_v23 = summary.sort_values("validation_score_v23", ascending=False).reset_index(drop=True)
    ranked_v23.head(15).to_csv(output / "top15_v23_candidates_exposure_v24.csv", index=False)
    summary.head(15).to_csv(output / "top15_concurrent_risk_v24.csv", index=False)

    for strategy in ranked_v23.head(5)["strategy"].astype(str):
        top_paths[strategy].to_csv(output / f"exposure_path_{strategy}.csv", index=False)

    h4 = summary[summary["strategy"] == "H4_S6_BREAKOUT"]
    h4_record = h4.iloc[0].to_dict() if len(h4) == 1 else None

    # These are diagnostics, not newly tuned limits. They quantify how often
    # aggregate risk-at-stop already exceeded the frozen 5% hard-DD policy.
    candidates_over_5pct_validation = int((summary["validation_max_open_risk_fraction"] > 0.05).sum())
    candidates_over_2pct_validation = int((summary["validation_max_open_risk_fraction"] > 0.02).sum())

    decision = {
        "version": "v0.24",
        "decision": "PORTFOLIO_EXPOSURE_AUDIT_ONLY",
        "candidate_count": int(len(summary)),
        "candidates_validation_open_risk_gt_2pct": candidates_over_2pct_validation,
        "candidates_validation_open_risk_gt_5pct": candidates_over_5pct_validation,
        "h4_s6_breakout": h4_record,
        "interpretation_contract": (
            "Open-risk metrics diagnose concurrency after v0.23 causality correction. "
            "They do not relax the 5% drawdown gate and are not a promotion rule."
        ),
        "recommended_next_research_step": (
            "Design and test a pre-entry portfolio allocator with aggregate open-risk and correlation constraints, "
            "then rerun the full frozen-gate tournament without recycling the final holdout."
        ),
        "threshold_relaxation": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    manifest = {
        "version": "v0.24",
        "experiment": "CONCURRENT_PORTFOLIO_EXPOSURE_AUDIT",
        "source": {
            "workflow_run_id": FROZEN_V20_RUN_ID,
            "artifact_name": FROZEN_V20_ARTIFACT,
            "candidate_count": total_trials,
        },
        "decision": decision,
        "top5_by_v23_score": ranked_v23.head(5).to_dict(orient="records"),
        "top5_by_validation_open_risk": summary.head(5).to_dict(orient="records"),
    }
    (output / "decision_v24.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "exposure_manifest_v24.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
