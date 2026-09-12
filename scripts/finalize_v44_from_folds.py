from __future__ import annotations

"""Aggregate frozen v0.44 fold outputs and apply all preregistered gates."""

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_DEVELOPMENT_VENUES,
    brier_skill_v43,
)
from research_bot.breadth_cluster_v42 import (
    BreadthClusterPolicyV42,
    hierarchical_symbol_block_bootstrap_low_v42,
    v42_gate_from_metrics,
)
from research_bot.financial_system_v39 import FinancialRiskPolicyV39, ValidationPolicyV39
from research_bot.information_sampling_v44 import V44_VARIANTS, preregistration_manifest_v44

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


v42 = _load(V42, "v42_for_v44_finalize")
corrected = _load(V39C, "v39c_for_v44_finalize")
base = v42.base


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


def _finite_median(values: list[float]) -> float | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _copy_provenance(indir: Path, outdir: Path) -> None:
    names = [
        "availability_v44.csv",
        "data_quality_manifest_v44.csv",
        "eligible_assets_v44.csv",
        "data_manifest_v44.csv",
        "sampling_counts_v44.csv",
        "variant_counts_v44.csv",
        "fold_index_v44.csv",
        "prepared_hashes_v44.json",
        "prep_status_v44.json",
        "execution_provenance.json",
    ]
    provenance = outdir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = indir / name
        if not source.exists():
            raise RuntimeError(f"missing canonical v0.44 provenance file: {name}")
        shutil.copy2(source, provenance / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = preregistration_manifest_v44()
    prep = json.loads((indir / "prep_status_v44.json").read_text(encoding="utf-8"))
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 prep is not READY/fail-closed")

    parts: list[pd.DataFrame] = []
    asset_parts: list[pd.DataFrame] = []
    perturb_parts: list[pd.DataFrame] = []
    fold_variant_parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        parts.append(pd.read_pickle(indir / f"oos_fold_{fold}_v44.pkl.gz", compression="gzip"))
        asset_parts.append(pd.read_csv(indir / f"asset_diag_fold_{fold}_v44.csv"))
        perturb_parts.append(pd.read_csv(indir / f"perturb_diag_fold_{fold}_v44.csv"))
        fold_variant_parts.append(pd.read_csv(indir / f"fold_variant_diag_{fold}_v44.csv"))

    oos = pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["variant", "signal_time", "venue", "symbol"], kind="mergesort"
    )
    assets_all = pd.concat(asset_parts, ignore_index=True, sort=False)
    perturb_all = pd.concat(perturb_parts, ignore_index=True, sort=False)
    fold_variant_all = pd.concat(fold_variant_parts, ignore_index=True, sort=False)
    quality = pd.read_csv(indir / "data_quality_manifest_v44.csv")

    summaries: list[dict] = []
    decisions: dict[str, dict] = {}
    all_trades: list[pd.DataFrame] = []
    family_outputs: list[pd.DataFrame] = []
    worst_outputs: list[pd.DataFrame] = []

    for variant in V44_VARIANTS:
        x = oos.loc[oos["variant"].astype(str).eq(variant)].copy()
        selected = x.loc[x["selected_model"].astype(bool)].copy() if not x.empty else x
        realized = base._realize_nonoverlap(selected) if not selected.empty else selected
        financially_realized = corrected._financial_simulation_independent_by_venue(realized) if not realized.empty else realized
        executed = (
            financially_realized.loc[financially_realized["executed"].astype(bool)].copy()
            if not financially_realized.empty and "executed" in financially_realized else financially_realized
        )
        if not executed.empty:
            executed["variant"] = variant
            executed["candidate"] = f"V44_{variant}"
            all_trades.append(executed)

        venue_results: dict[str, dict] = {}
        for venue in V43_DEVELOPMENT_VENUES:
            eligible = sorted(
                quality.loc[
                    (quality["venue"].astype(str) == venue) & quality["accepted"].astype(bool), "symbol"
                ].astype(str).unique()
            )
            vt = executed.loc[executed["venue"].astype(str).eq(venue)].copy() if not executed.empty else executed
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
            xf = x.loc[x["fold"].astype(int).eq(fold)].copy()
            sf = xf.loc[xf["selected_model"].astype(bool)] if not xf.empty else xf
            fold_expectancies.append(float(sf["net_r"].mean()) if not sf.empty else float("nan"))
            if xf.empty:
                target_skill.append(float("nan"))
                stop_skill.append(float("nan"))
                continue
            yt = xf["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
            ys = xf["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
            target_skill.append(brier_skill_v43(yt, xf["p_target_v41"], xf["baseline_p_target_v44"]))
            stop_skill.append(brier_skill_v43(ys, xf["p_stop_v41"], xf["baseline_p_stop_v44"]))

        positive_fold_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in fold_expectancies]))
        target_positive_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in target_skill]))
        stop_positive_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in stop_skill]))
        med_target = _finite_median(target_skill)
        med_stop = _finite_median(stop_skill)
        skill_gate = bool(
            target_positive_fraction >= 0.60 and stop_positive_fraction >= 0.60
            and med_target is not None and med_target > 0.0
            and med_stop is not None and med_stop > 0.0
        )
        fold_gate = positive_fold_fraction >= 0.60

        median_admissible = x.loc[x["median_admissible_v44"].astype(bool)].copy() if not x.empty else x
        perturbation_stability = (
            float((pd.to_numeric(median_admissible["perturbation_agreement_v44"], errors="coerce") >= (2.0 / 3.0)).mean())
            if not median_admissible.empty else 0.0
        )
        perturbation_gate = perturbation_stability >= 0.60
        venue_gate = all(bool(venue_results[v]["gates"]["venue_pass"]) for v in V43_DEVELOPMENT_VENUES)
        passed = bool(venue_gate and fold_gate and skill_gate and perturbation_gate)

        if not executed.empty:
            fam = v42._family_breakdown(executed)
            fam.insert(0, "variant", variant)
            family_outputs.append(fam)
            worst = base._worst_groups(executed)
            worst.insert(0, "variant", variant)
            worst_outputs.append(worst)

        summary = {
            "variant": variant,
            "oos_events": int(len(x)),
            "median_admissible_events": int(len(median_admissible)),
            "model_selected_events": int(len(selected)),
            "nonoverlap_events": int(len(realized)),
            "financially_executed_events": int(len(executed)),
            "fold_expectancies_r": fold_expectancies,
            "positive_fold_fraction": positive_fold_fraction,
            "target_brier_skill_by_fold": target_skill,
            "stop_brier_skill_by_fold": stop_skill,
            "target_skill_positive_fold_fraction": target_positive_fraction,
            "stop_skill_positive_fold_fraction": stop_positive_fraction,
            "median_target_brier_skill": med_target,
            "median_stop_brier_skill": med_stop,
            "perturbation_stable_median_admissible_fraction": perturbation_stability,
            "all_venue_gates_pass": venue_gate,
            "fold_gate_pass": fold_gate,
            "forecast_skill_gate_pass": skill_gate,
            "perturbation_gate_pass": perturbation_gate,
            "candidate_pass": passed,
            "venue_results": venue_results,
        }
        decisions[variant] = summary
        summaries.append({k: v for k, v in summary.items() if k not in {"venue_results", "fold_expectancies_r", "target_brier_skill_by_fold", "stop_brier_skill_by_fold"}})

    passing = [v for v in V44_VARIANTS if bool(decisions[v]["candidate_pass"])]
    winner = passing[0] if passing else None
    decision_name = "V44_DEVELOPMENT_WINNER" if winner else "V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE"

    oos.to_csv(outdir / "oos_predictions_v44.csv.gz", index=False, compression="gzip")
    pd.concat(all_trades, ignore_index=True, sort=False).to_csv(
        outdir / "trades_v44.csv.gz", index=False, compression="gzip"
    ) if all_trades else pd.DataFrame().to_csv(outdir / "trades_v44.csv.gz", index=False, compression="gzip")
    assets_all.to_csv(outdir / "asset_diagnostics_v44.csv", index=False)
    perturb_all.to_csv(outdir / "perturbation_diagnostics_v44.csv", index=False)
    fold_variant_all.to_csv(outdir / "fold_variant_diagnostics_v44.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "candidate_summary_v44.csv", index=False)
    if family_outputs:
        pd.concat(family_outputs, ignore_index=True, sort=False).to_csv(outdir / "event_family_results_v44.csv", index=False)
    if worst_outputs:
        pd.concat(worst_outputs, ignore_index=True, sort=False).to_csv(outdir / "worst_groups_v44.csv", index=False)

    _copy_provenance(indir, outdir)
    result = {
        "version": "v0.44",
        "experiment": manifest["experiment"],
        "variants_in_frozen_order": list(V44_VARIANTS),
        "decision": decision_name,
        "winner": winner,
        "passing_variants": passing,
        "variant_results": decisions,
        "selection_rule": "first/simplest passing variant in frozen S0->S1->S2 order",
        "data_quality_policy": asdict(DataQualityPolicyV43()),
        "breadth_cluster_policy": asdict(BreadthClusterPolicyV42()),
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "common_fold_source": "S0_CLOCK_MOTHER_BASELINE",
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_variant_tuning": False,
        "post_result_asset_pruning": False,
    }
    (outdir / "decision_v44.json").write_text(json.dumps(base._jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(base._jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
