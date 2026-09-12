from __future__ import annotations

"""Aggregate frozen v0.43 fold outputs into one development decision."""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_DEVELOPMENT_VENUES,
    brier_skill_v43,
    preregistration_manifest_v43,
)
from research_bot.breadth_cluster_v42 import (
    BreadthClusterPolicyV42,
    hierarchical_symbol_block_bootstrap_low_v42,
    v42_gate_from_metrics,
)
from research_bot.financial_system_v39 import FinancialRiskPolicyV39, ValidationPolicyV39

ROOT = Path(__file__).resolve().parents[1]
V42 = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
V39C = ROOT / "scripts" / "run_v39_development_characterization_corrected.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v42 = _load(V42, "v42_for_v43_finalize")
corrected = _load(V39C, "v39c_for_v43_finalize")
base = v42.base
CANDIDATE = "V43_ASSET_CROSSVENUE_HISTGB"


def _venue_metrics(trades: pd.DataFrame, eligible_assets: list[str]) -> dict:
    if trades.empty:
        return {
            "n": 0, "profit_factor": None, "expectancy_r": None,
            "positive_asset_fraction": 0.0, "block_ci_low": None,
            "cluster_ci_low": None, "positive_quarter_fraction": 0.0,
            "stress_profit_factor": None, "max_account_drawdown": None,
        }
    x = trades.sort_values(["exit_time", "symbol"], kind="mergesort").copy()
    asset_exp = x.groupby("symbol")["net_r"].mean()
    breadth = float(np.mean([float(asset_exp.get(s, -np.inf)) > 0.0 for s in eligible_assets])) if eligible_assets else 0.0
    quarter = pd.to_datetime(x["exit_time"], utc=True).dt.to_period("Q")
    q = x.groupby(quarter)["net_r"].sum()
    eq = x["equity_after_exit"].dropna().to_numpy(dtype=float) if "equity_after_exit" in x else np.array([])
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
    max_dd = float(np.min(eq / peak - 1.0)) if len(eq) else None
    return {
        "n": int(len(x)),
        "profit_factor": base._profit_factor(x["net_r"]),
        "expectancy_r": float(x["net_r"].mean()),
        "positive_asset_fraction": breadth,
        "block_ci_low": base._block_ci_low(x["net_r"]),
        "cluster_ci_low": hierarchical_symbol_block_bootstrap_low_v42(x),
        "positive_quarter_fraction": float((q > 0).mean()) if len(q) else 0.0,
        "stress_profit_factor": base._profit_factor(x["stress_net_r"]),
        "max_account_drawdown": max_dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = preregistration_manifest_v43()
    prep = json.loads((indir / "prep_status_v43.json").read_text(encoding="utf-8"))
    if prep.get("status") != "READY":
        raise RuntimeError("v0.43 prep is not READY")

    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    asset_rows: list[pd.DataFrame] = []
    perturb_rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        parts.append(pd.read_pickle(indir / f"oos_fold_{fold}_v43.pkl.gz", compression="gzip"))
        fold_rows.append(json.loads((indir / f"fold_diag_{fold}_v43.json").read_text(encoding="utf-8")))
        asset_rows.append(pd.read_csv(indir / f"asset_diag_fold_{fold}_v43.csv"))
        perturb_rows.append(pd.read_csv(indir / f"perturb_diag_fold_{fold}_v43.csv"))

    oos = pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["signal_time", "venue", "symbol"], kind="mergesort"
    )
    selected = oos[oos["selected_model"].astype(bool)].copy()
    realized = base._realize_nonoverlap(selected)
    financially_realized = corrected._financial_simulation_independent_by_venue(realized)
    executed = financially_realized[financially_realized["executed"]].copy() if not financially_realized.empty else financially_realized
    if not executed.empty:
        executed["candidate"] = CANDIDATE

    quality = pd.read_csv(indir / "data_quality_manifest_v43.csv")
    venue_results: dict[str, dict] = {}
    for venue in V43_DEVELOPMENT_VENUES:
        eligible = sorted(
            quality.loc[(quality["venue"].astype(str) == venue) & quality["accepted"].astype(bool), "symbol"].astype(str).unique()
        )
        vt = executed[executed["venue"].astype(str).eq(venue)].copy() if not executed.empty else executed
        metrics = _venue_metrics(vt, eligible)
        venue_results[venue] = {
            "eligible_asset_count": len(eligible),
            "metrics": metrics,
            "gates": v42_gate_from_metrics(metrics),
        }

    fold_expectancies: list[float] = []
    target_skill: list[float] = []
    stop_skill: list[float] = []
    for fold in range(1, 6):
        x = oos[oos["fold"].astype(int).eq(fold)].copy()
        s = x[x["selected_model"].astype(bool)]
        fold_expectancies.append(float(s["net_r"].mean()) if not s.empty else float("nan"))
        yt = x["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
        ys = x["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
        target_skill.append(brier_skill_v43(yt, x["p_target_v41"], x["baseline_p_target_v43"]))
        stop_skill.append(brier_skill_v43(ys, x["p_stop_v41"], x["baseline_p_stop_v43"]))

    positive_fold_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in fold_expectancies]))
    target_skill_positive_folds = float(np.mean(np.asarray(target_skill) > 0.0))
    stop_skill_positive_folds = float(np.mean(np.asarray(stop_skill) > 0.0))
    skill_gate = bool(
        target_skill_positive_folds >= 0.60
        and stop_skill_positive_folds >= 0.60
        and float(np.median(target_skill)) > 0.0
        and float(np.median(stop_skill)) > 0.0
    )
    fold_gate = positive_fold_fraction >= 0.60

    # Robustness must be measured BEFORE the agreement filter. Measuring it on
    # executed trades is tautological because selected_model already requires
    # >=2/3 perturbation agreement. Use all median-admissible rows instead.
    median_admissible = oos[oos["selected_v41"].astype(bool)].copy() if "selected_v41" in oos else oos.iloc[0:0]
    perturbation_stability = (
        float((pd.to_numeric(median_admissible["perturbation_agreement_v43"], errors="coerce") >= (2.0 / 3.0)).mean())
        if not median_admissible.empty else 0.0
    )
    perturbation_gate = perturbation_stability >= 0.60
    venue_gate = all(bool(venue_results[v]["gates"]["venue_pass"]) for v in V43_DEVELOPMENT_VENUES)
    passed = bool(venue_gate and fold_gate and skill_gate and perturbation_gate)

    oos.to_csv(outdir / "oos_predictions_v43.csv.gz", index=False, compression="gzip")
    executed.to_csv(outdir / "trades_v43.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v43.csv", index=False)
    pd.concat(asset_rows, ignore_index=True, sort=False).to_csv(outdir / "asset_diagnostics_v43.csv", index=False)
    pd.concat(perturb_rows, ignore_index=True, sort=False).to_csv(outdir / "perturbation_diagnostics_v43.csv", index=False)
    quality.to_csv(outdir / "data_quality_manifest_v43.csv", index=False)
    if not executed.empty:
        v42._family_breakdown(executed).to_csv(outdir / "event_family_results_v43.csv", index=False)
        base._worst_groups(executed).to_csv(outdir / "worst_groups_v43.csv", index=False)

    summary = {
        "candidate": CANDIDATE,
        "oos_events": int(len(oos)),
        "median_admissible_events": int(len(median_admissible)),
        "model_selected_events": int(len(selected)),
        "nonoverlap_events": int(len(realized)),
        "financially_executed_events": int(len(executed)),
        "fold_expectancies_r": fold_expectancies,
        "positive_fold_fraction": positive_fold_fraction,
        "target_brier_skill_by_fold": target_skill,
        "stop_brier_skill_by_fold": stop_skill,
        "target_skill_positive_fold_fraction": target_skill_positive_folds,
        "stop_skill_positive_fold_fraction": stop_skill_positive_folds,
        "median_target_brier_skill": float(np.median(target_skill)),
        "median_stop_brier_skill": float(np.median(stop_skill)),
        "perturbation_stable_median_admissible_fraction": perturbation_stability,
        "all_venue_gates_pass": venue_gate,
        "fold_gate_pass": fold_gate,
        "forecast_skill_gate_pass": skill_gate,
        "perturbation_gate_pass": perturbation_gate,
        "candidate_pass": passed,
        "venue_results": venue_results,
    }
    pd.DataFrame([{k: v for k, v in summary.items() if k != "venue_results"}]).to_csv(
        outdir / "candidate_summary_v43.csv", index=False
    )
    decision = {
        "version": "v0.43",
        "experiment": manifest["experiment"],
        "decision": "V43_DEVELOPMENT_WINNER" if passed else "V43_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": CANDIDATE if passed else None,
        "candidate_summary": summary,
        "data_quality_policy": asdict(DataQualityPolicyV43()),
        "breadth_cluster_policy": asdict(BreadthClusterPolicyV42()),
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_asset_pruning": False,
        "perturbation_gate_basis": "median_admissible_pre_agreement_rows",
    }
    (outdir / "decision_v43.json").write_text(json.dumps(base._jsonable(decision), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(base._jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
