from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.cross_venue_robustness_v27 import compact_external_metrics
from research_bot.drawdown_firewall_v31 import apply_drawdown_firewall_v31, allocator_diagnostics_v31
from research_bot.event_utility_v36 import DEVELOPMENT_VENUES_V36, TOTAL_EFFECTIVE_TRIALS_V36, V36_CANDIDATES
from research_bot.marginal_utility_v37 import (
    TOTAL_EFFECTIVE_TRIALS_V37,
    V37_POLICIES,
    arbitrate_events_v37,
    preregistration_manifest_v37,
    screen_three_venues_v37,
    select_v37_winner,
    v37_decision,
)
from research_bot.multitimeframe_strategies_v19 import StrategySpec
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(path, compression="gzip")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    return {
        str(symbol): group[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp").reset_index(drop=True)
        for symbol, group in x.groupby("symbol", sort=True)
    }


def read_scored(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, compression="gzip")
    for col in ("signal_time", "entry_time", "exit_time"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="raise")
    if x["selected_v36"].dtype != bool:
        x["selected_v36"] = x["selected_v36"].astype(str).str.lower().eq("true")
    return x


def empty_metrics() -> dict[str, Any]:
    return {
        "external_trades": 0,
        "external_profit_factor": np.nan,
        "external_expectancy_r": np.nan,
        "external_max_drawdown": np.nan,
        "external_positive_asset_fraction": 0.0,
        "external_block_ci_low": np.nan,
        "external_block_ci_high": np.nan,
        "external_pass_v25": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.37 preregistered marginal-utility portfolio arbitration")
    ap.add_argument("--v30-artifact-dir", required=True)
    ap.add_argument("--v31-artifact-dir", required=True)
    ap.add_argument("--v34-artifact-dir", required=True)
    ap.add_argument("--v35-artifact-dir", required=True)
    ap.add_argument("--v36-artifact-dir", required=True)
    ap.add_argument("--v36-run-id", required=True, type=int)
    ap.add_argument("--output-dir", default="artifacts/v37-marginal-utility")
    args = ap.parse_args()

    s30, s31, s34, s35, s36 = map(Path, [
        args.v30_artifact_dir, args.v31_artifact_dir, args.v34_artifact_dir,
        args.v35_artifact_dir, args.v36_artifact_dir,
    ])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prior34 = json.loads((s34 / "decision_v34.json").read_text(encoding="utf-8"))
    prior35 = json.loads((s35 / "decision_v35.json").read_text(encoding="utf-8"))
    prior36 = json.loads((s36 / "decision_v36.json").read_text(encoding="utf-8"))
    if prior34.get("decision") != "BLOCKED_BY_V33" or bool(prior34.get("holdout_used")):
        raise RuntimeError("v0.37 requires Kraken seal evidence")
    if prior35.get("decision") != "NO_V35_ROBUST_DEVELOPMENT_CANDIDATE" or bool(prior35.get("kraken_touched")):
        raise RuntimeError("v0.37 requires v0.35 rejection with Kraken untouched")
    if bool(prior36.get("kraken_touched")):
        raise RuntimeError("v0.37 refuses any v0.36 artifact that touched Kraken")
    if prior36.get("decision") not in {
        "NO_V36_ROBUST_EVENT_UTILITY_CANDIDATE",
        "V36_EVENT_UTILITY_CANDIDATE_REQUIRES_PORTFOLIO_ARBITRATION",
    }:
        raise RuntimeError("unexpected v0.36 decision state")

    frames_by_venue = {
        "coinex_consumed": load_snapshot(s30 / "coinex_1d_ohlcv_v30.csv.gz"),
        "okx_consumed": load_snapshot(s30 / "okx_1d_ohlcv_v30.csv.gz"),
        "kucoin_consumed": load_snapshot(s31 / "kucoin_1d_ohlcv_v31.csv.gz"),
    }
    v36_summary = pd.read_csv(s36 / "development_summary_v36.csv")
    parent_pass = {
        str(row["strategy"]): bool(row["development_eligible_v36"])
        for _, row in v36_summary.iterrows()
    }

    prereg = preregistration_manifest_v37()
    prereg.update({
        "bound_v36_run_id": int(args.v36_run_id),
        "bound_v36_effective_trials": TOTAL_EFFECTIVE_TRIALS_V36,
        "source_v30_run_id": 34632401230,
        "source_v31_run_id": 34633572528,
        "source_v34_run_id": 34635635582,
        "source_v35_run_id": 34637596442,
    })
    write_json(out / "preregistration_v37.json", prereg)

    risk = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    validation = V20ValidationConfig(min_external_trades=200)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for parent in V36_CANDIDATES:
        for policy in V37_POLICIES:
            name = f"{parent.name}__{policy.name}"
            detailed[name] = {}
            compact: dict[str, dict[str, Any]] = {}
            for venue in DEVELOPMENT_VENUES_V36:
                scored = read_scored(s36 / f"scored_{parent.name}_{venue}.csv.gz")
                arbitrated, arb_diag = arbitrate_events_v37(scored, frames_by_venue[venue], policy)
                admitted = arbitrated.loc[arbitrated["admitted_v37"].astype(bool)].copy()
                if admitted.empty:
                    metrics = empty_metrics()
                    allocated = admitted
                else:
                    spec = StrategySpec(
                        name,
                        "1d",
                        "v37_marginal_utility",
                        "v0.36 expected-net-R event ranking + preregistered correlation/CVaR/duration/uncertainty admission",
                        rr=3.0,
                        stop_atr=1.7,
                        max_hold_bars=35,
                        long_short=True,
                    )
                    allocated = apply_drawdown_firewall_v31(admitted, spec, risk_policy=risk, budget=budget)
                    metrics = dict(evaluate_external_replication_v25(allocated, validation=validation))
                    metrics.update(allocator_diagnostics_v31(allocated))
                detailed[name][venue] = {
                    "arbitration": arb_diag,
                    "metrics": metrics,
                    "admitted_before_allocator": int(len(admitted)),
                }
                compact[venue] = compact_external_metrics(metrics, venue=venue)

            screen = screen_three_venues_v37(compact, parent_event_eligible=parent_pass.get(parent.name, False))
            row: dict[str, Any] = {
                "strategy": name,
                "parent_v36": parent.name,
                "policy": policy.name,
                **screen,
            }
            for venue in DEVELOPMENT_VENUES_V36:
                tag = venue.split("_")[0]
                m = compact[venue]
                a = detailed[name][venue]["arbitration"]
                row.update({
                    f"{tag}_trades": m.get("trades"),
                    f"{tag}_profit_factor": m.get("profit_factor"),
                    f"{tag}_expectancy_r": m.get("expectancy_r"),
                    f"{tag}_max_drawdown": m.get("max_drawdown"),
                    f"{tag}_breadth": m.get("positive_asset_fraction"),
                    f"{tag}_block_ci_low": m.get("block_ci_low"),
                    f"{tag}_input_selected": a.get("input_selected"),
                    f"{tag}_admitted": a.get("admitted"),
                    f"{tag}_dup_symbol_rejections": a.get("duplicate_symbol_rejections"),
                    f"{tag}_capacity_rejections": a.get("capacity_rejections"),
                })
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["joint_eligible_v37", "portfolio_eligible_v37", "robust_floor_block_ci_low_v37", "robust_floor_breadth_v37", "robust_floor_profit_factor_v37"],
        ascending=[False, False, False, False, False], kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(out / "development_summary_v37.csv", index=False)
    write_json(out / "development_metrics_v37.json", detailed)

    winner = select_v37_winner(rows)
    lock = {
        "version": "v0.37",
        "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "winner": None if winner is None else winner["strategy"],
        "effective_trials": TOTAL_EFFECTIVE_TRIALS_V37,
        "development_venues": list(DEVELOPMENT_VENUES_V36),
        "reserved_holdout_venue": "kraken",
        "kraken_touched": False,
        "bound_v36_run_id": int(args.v36_run_id),
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    decision = v37_decision(winner)
    write_json(out / "development_lock_v37.json", lock)
    write_json(out / "decision_v37.json", decision)
    print(json.dumps({
        "decision": decision,
        "joint_eligible_count": int(summary["joint_eligible_v37"].sum()),
        "portfolio_eligible_count": int(summary["portfolio_eligible_v37"].sum()),
        "top": summary.head(9).to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
