from __future__ import annotations

"""Forensic v0.23 replay of the frozen v0.20 H4_S6_BREAKOUT evidence.

This is intentionally *not* a new strategy search. It consumes the exact CSV
ledgers emitted by the successful v0.20 workflow run, strips the old risk-state
outputs conceptually by ignoring them, and replays the same raw trade attempts
through the v0.23 causal event-driven risk engine.

That design isolates one treatment variable: portfolio risk/equity accounting.
Market data, generated signals, transaction-cost-adjusted R multiples, symbols,
segments and the selected strategy are held fixed.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v23 import (
    apply_candidate_level_policy_v23,
    evaluate_external_replication_v23,
    evaluate_v23_candidate,
)
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
    V20ValidationConfig,
)


FROZEN_WINNER = "H4_S6_BREAKOUT"
FROZEN_V20_RUN_ID = 34482338133
FROZEN_V20_ARTIFACT = "v20-global-strategy-risk-lab"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required frozen v0.20 artifact file is missing: {path}")
    frame = pd.read_csv(path)
    for column in ("signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return frame


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required frozen v0.20 artifact file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _winner_spec():
    matches = [spec for spec in STRATEGY_REGISTRY_V20 if spec.name == FROZEN_WINNER]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one frozen winner spec, got {len(matches)}")
    return matches[0]


def _filter_winner(frame: pd.DataFrame) -> pd.DataFrame:
    if "strategy" not in frame:
        raise ValueError("frozen ledger is missing strategy column")
    out = frame[frame["strategy"].astype(str) == FROZEN_WINNER].copy()
    if out.empty:
        raise ValueError(f"frozen ledger contains no rows for {FROZEN_WINNER}")
    required = {"symbol", "entry_time", "exit_time", "r_multiple", "risk_scale_volatility"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"frozen winner ledger missing raw replay columns: {sorted(missing)}")
    return out.reset_index(drop=True)


def _old_internal_reference(artifact_dir: Path) -> dict[str, Any]:
    summary = _load_csv(artifact_dir / "strategy_summary.csv")
    row = summary[summary["strategy"].astype(str) == FROZEN_WINNER]
    if len(row) != 1:
        raise ValueError("frozen strategy_summary does not contain exactly one winner row")
    return row.iloc[0].to_dict()


def _comparison(old_internal: dict[str, Any], old_external: dict[str, Any], new_internal: dict[str, Any], new_external: dict[str, Any]) -> dict[str, Any]:
    def delta(new: Any, old: Any) -> float | None:
        try:
            a, b = float(new), float(old)
        except (TypeError, ValueError):
            return None
        return float(a - b) if np.isfinite(a) and np.isfinite(b) else None

    return {
        "validation": {
            "old_max_drawdown": old_internal.get("validation_max_drawdown"),
            "v23_max_drawdown": new_internal.get("validation_max_drawdown"),
            "delta_max_drawdown": delta(new_internal.get("validation_max_drawdown"), old_internal.get("validation_max_drawdown")),
            "old_total_return": old_internal.get("validation_total_return"),
            "v23_total_return": new_internal.get("validation_total_return"),
            "delta_total_return": delta(new_internal.get("validation_total_return"), old_internal.get("validation_total_return")),
            "old_executed_trades": old_internal.get("executed_trades"),
            "v23_executed_trades": new_internal.get("executed_trades"),
            "old_internal_eligible": old_internal.get("internal_eligible"),
            "v23_internal_eligible": new_internal.get("internal_eligible_v23"),
        },
        "external_okx": {
            "old_max_drawdown": old_external.get("external_max_drawdown"),
            "v23_max_drawdown": new_external.get("external_max_drawdown"),
            "delta_max_drawdown": delta(new_external.get("external_max_drawdown"), old_external.get("external_max_drawdown")),
            "old_total_return": old_external.get("external_total_return"),
            "v23_total_return": new_external.get("external_total_return"),
            "delta_total_return": delta(new_external.get("external_total_return"), old_external.get("external_total_return")),
            "old_trades": old_external.get("external_trades"),
            "v23_trades": new_external.get("external_trades"),
            "old_profit_factor": old_external.get("external_profit_factor"),
            "v23_profit_factor": new_external.get("external_profit_factor"),
            "old_block_ci_low": old_external.get("external_block_ci_low"),
            "v23_block_ci_low": new_external.get("external_block_ci_low"),
            "old_external_pass": old_external.get("external_pass"),
            "v23_external_pass": new_external.get("external_pass_v23"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.23 causal forensic replay of the frozen v0.20 winner")
    parser.add_argument("--v20-artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v23-causal-replay")
    args = parser.parse_args()

    artifact_dir = Path(args.v20_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    policy = RiskPsychologyPolicy()
    validation = V20ValidationConfig()
    spec = _winner_spec()
    total_trials = len(STRATEGY_REGISTRY_V20)

    old_internal = _old_internal_reference(artifact_dir)
    old_external = _load_json(artifact_dir / "external_replication.json")
    old_decision = _load_json(artifact_dir / "decision.json")

    internal_frozen = _filter_winner(_load_csv(artifact_dir / "trade_attempt_ledger.csv"))
    external_frozen = _filter_winner(_load_csv(artifact_dir / "external_replication_ledger.csv"))

    internal_v23 = apply_candidate_level_policy_v23(internal_frozen, spec, policy=policy)
    external_v23 = apply_candidate_level_policy_v23(external_frozen, spec, policy=policy)

    new_internal = evaluate_v23_candidate(internal_v23, total_trials=total_trials, validation=validation)
    new_external = evaluate_external_replication_v23(external_v23, validation=validation)
    comparison = _comparison(old_internal, old_external, new_internal, new_external)

    internal_pass = bool(new_internal.get("internal_eligible_v23"))
    external_pass = bool(new_external.get("external_pass_v23"))
    if internal_pass and external_pass:
        decision_code = "CAUSAL_REPLAY_PASSES_FIXED_WINNER_GATES_FULL_TOURNAMENT_REQUIRED"
        reason = (
            "The fixed v0.20 winner clears the frozen internal/external thresholds after causal replay. "
            "This is a forensic one-strategy diagnostic, not a corrected 42-strategy selection; rerun the full v0.23 tournament before any forward-paper promotion decision."
        )
    elif not internal_pass:
        decision_code = "CAUSAL_REPLAY_INVALIDATES_INTERNAL_ELIGIBILITY"
        reason = "The fixed historical winner no longer clears the frozen internal gate under causal event-driven accounting."
    else:
        decision_code = "CAUSAL_REPLAY_CONFIRMS_NO_EXTERNAL_PROMOTION"
        reason = "The fixed historical winner remains internally eligible but still fails the frozen external replication gate under causal accounting."

    decision = {
        "version": "v0.23",
        "decision": decision_code,
        "reason": reason,
        "strategy": FROZEN_WINNER,
        "timeframe": spec.timeframe,
        "diagnostic_only": True,
        "full_v23_tournament_required": True,
        "forward_paper_candidate_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }

    internal_v23.to_csv(output / "internal_winner_causal_ledger.csv", index=False)
    external_v23.to_csv(output / "external_okx_causal_ledger.csv", index=False)
    (output / "internal_metrics_v23.json").write_text(json.dumps(_json_safe(new_internal), indent=2, sort_keys=True), encoding="utf-8")
    (output / "external_metrics_v23.json").write_text(json.dumps(_json_safe(new_external), indent=2, sort_keys=True), encoding="utf-8")
    (output / "comparison_v20_vs_v23.json").write_text(json.dumps(_json_safe(comparison), indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "version": "v0.23",
        "experiment": "FROZEN_ARTIFACT_CAUSAL_REPLAY",
        "source": {
            "workflow_run_id": FROZEN_V20_RUN_ID,
            "artifact_name": FROZEN_V20_ARTIFACT,
            "strategy": FROZEN_WINNER,
            "input_contract": "same frozen generated trade attempts; only portfolio risk/equity accounting changes",
        },
        "frozen_v20_decision": old_decision.get("decision"),
        "old_internal": old_internal,
        "old_external": old_external,
        "v23_internal": new_internal,
        "v23_external": new_external,
        "comparison": comparison,
        "decision": decision,
    }
    (output / "replay_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(_json_safe({"decision": decision, "comparison": comparison}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
