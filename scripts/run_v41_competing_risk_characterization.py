from __future__ import annotations

"""Frozen v0.41 event-specific competing-risk development characterization."""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.event_competing_risk_v41 import (
    CompetingRiskPolicyV41,
    V41_CANDIDATES,
    V41_FEATURES,
    V41_SEEDS,
    attach_event_family_v41,
    calibrate_expected_r_bounds_v41,
    fit_competing_risk_bundle_v41,
    median_seed_prediction_v41,
    predict_competing_risks_v41,
    preregistration_manifest_v41,
)
from research_bot.financial_system_v39 import FinancialRiskPolicyV39, ValidationPolicyV39

ROOT = Path(__file__).resolve().parents[1]
V39_BASE = ROOT / "scripts" / "run_v39_development_characterization.py"
V39_CORRECTED = ROOT / "scripts" / "run_v39_development_characterization_corrected.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load(V39_BASE, "v39_base_for_v41")
corrected = _load(V39_CORRECTED, "v39_corrected_for_v41")

VENUES = base.VENUES
SYMBOLS = base.SYMBOLS
TIMEFRAME = base.TIMEFRAME
START = base.START
END = base.END
PURGED_FOLDS = base.PURGED_FOLDS
EMBARGO_BARS = base.EMBARGO_BARS
BASE_ROUNDTRIP_BPS = base.BASE_ROUNDTRIP_BPS
STRESS_ROUNDTRIP_BPS = base.STRESS_ROUNDTRIP_BPS


def _jsonable(obj):
    return base._jsonable(obj)


def _prepare_events() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, events, data_manifest = base._prepare_panel()
    events = attach_event_family_v41(panel, events)
    missing = sorted(set(V41_FEATURES) - set(events.columns))
    if missing:
        raise RuntimeError(f"missing frozen v0.41 features: {missing}")
    events = events.replace([np.inf, -np.inf], np.nan)
    events.loc[:, list(V41_FEATURES)] = events.loc[:, list(V41_FEATURES)].fillna(0.0).astype("float32")
    return panel, events, data_manifest


def _predict_fold(candidate: str, fold: dict) -> tuple[pd.DataFrame, dict[int, float], list[dict], dict]:
    seed_outputs: list[pd.DataFrame] = []
    seed_expectancies: dict[int, float] = {}
    seed_diags: list[dict] = []
    policy = CompetingRiskPolicyV41()

    for seed in V41_SEEDS:
        bundle = fit_competing_risk_bundle_v41(
            candidate,
            fold["fit"],
            V41_FEATURES,
            seed,
            policy,
        )
        cal_pred = predict_competing_risks_v41(bundle, fold["cal"])
        test_point = predict_competing_risks_v41(bundle, fold["test"])
        test_pred = calibrate_expected_r_bounds_v41(
            fold["cal"], cal_pred, test_point, policy
        )
        seed_outputs.append(test_pred)
        selected = test_pred[test_pred["selected_v41"]]
        seed_expectancies[seed] = float(selected["net_r"].mean()) if not selected.empty else float("nan")

        y_target = fold["test"]["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
        y_stop = fold["test"]["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
        brier_target = float(np.mean((test_pred["p_target_v41"].to_numpy(dtype=float) - y_target) ** 2))
        brier_stop = float(np.mean((test_pred["p_stop_v41"].to_numpy(dtype=float) - y_stop) ** 2))
        seed_diags.append(
            {
                "candidate": candidate,
                "fold": int(fold["fold"]),
                "seed": int(seed),
                "fit_events": int(len(fold["fit"])),
                "calibration_events": int(len(fold["cal"])),
                "test_events": int(len(fold["test"])),
                "selected_events": int(test_pred["selected_v41"].sum()),
                "expert_fraction": float(test_pred["expert_used_v41"].mean()),
                "brier_target": brier_target,
                "brier_stop": brier_stop,
                "mean_p_target": float(test_pred["p_target_v41"].mean()),
                "mean_p_stop": float(test_pred["p_stop_v41"].mean()),
                "mean_p_timeout": float(test_pred["p_timeout_v41"].mean()),
            }
        )

    combined = median_seed_prediction_v41(seed_outputs, policy)
    combined["fold"] = int(fold["fold"])
    combined["expected_r"] = combined["expected_r_v41"]
    combined["lower_expected_r"] = combined["lower_expected_r_v41"]
    combined["upper_expected_r"] = combined["upper_expected_r_v41"]
    combined["selected_model"] = combined["selected_v41"]
    selected = combined[combined["selected_model"]]
    fold_diag = {
        "candidate": candidate,
        "fold": int(fold["fold"]),
        "fit_events": int(len(fold["fit"])),
        "calibration_events": int(len(fold["cal"])),
        "test_events": int(len(fold["test"])),
        "selected_events": int(len(selected)),
        "selected_expectancy_r": float(selected["net_r"].mean()) if not selected.empty else None,
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "median_target_brier": float(np.median([d["brier_target"] for d in seed_diags])),
        "median_stop_brier": float(np.median([d["brier_stop"] for d in seed_diags])),
        "median_expert_fraction": float(np.median([d["expert_fraction"] for d in seed_diags])),
    }
    return combined, seed_expectancies, seed_diags, fold_diag


def _positive_fraction(values: list[float]) -> float:
    if len(values) != PURGED_FOLDS:
        raise RuntimeError(f"expected exactly {PURGED_FOLDS} folds, got {len(values)}")
    return float(np.mean([bool(np.isfinite(v) and v > 0.0) for v in values]))


def _family_breakdown(executed: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame(columns=["candidate", "event_family_v41", "side", "n", "expectancy_r", "profit_factor"])
    rows: list[dict] = []
    for (family, side), g in executed.groupby(["event_family_v41", "side"], sort=True):
        rows.append(
            {
                "candidate": candidate,
                "event_family_v41": family,
                "side": int(side),
                "n": int(len(g)),
                "expectancy_r": float(g["net_r"].mean()),
                "profit_factor": base._profit_factor(g["net_r"]),
            }
        )
    return pd.DataFrame(rows)


def _worst_groups(candidate_executed: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for x in candidate_executed:
        if x.empty:
            continue
        candidate = str(x["candidate"].iloc[0])
        w = base._worst_groups(x)
        if w.empty:
            continue
        w.insert(0, "candidate", candidate)
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=["candidate", "venue", "quarter", "regime", "n", "expectancy_r", "profit_factor"])
    return pd.concat(rows, ignore_index=True, sort=False).sort_values(
        ["candidate", "expectancy_r", "n"], ascending=[True, True, False]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = preregistration_manifest_v41()
    if manifest["kraken_touched"] is not False or manifest["reserved_holdout"] != "kraken":
        raise RuntimeError("v0.41 holdout governance invalid")

    _, events, data_manifest = _prepare_events()
    if events.empty:
        raise RuntimeError("v0.41 generated no mother events")
    folds = base._folds(events)
    if len(folds) != PURGED_FOLDS:
        raise RuntimeError("v0.41 requires all five frozen folds")

    candidate_summaries: list[dict] = []
    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    all_oos: list[pd.DataFrame] = []
    all_candidate_executed: list[pd.DataFrame] = []
    family_tables: list[pd.DataFrame] = []
    winner: str | None = None
    winner_trades = pd.DataFrame()
    winner_venue_results: dict[str, dict] = {}

    for candidate in V41_CANDIDATES:
        oos_parts: list[pd.DataFrame] = []
        fold_expectancies: list[float] = []
        seed_fold_values: dict[int, list[float]] = {s: [] for s in V41_SEEDS}

        for fold in folds:
            pred, seed_exps, sdiag, fdiag = _predict_fold(candidate, fold)
            oos_parts.append(pred)
            seed_rows.extend(sdiag)
            fold_rows.append(fdiag)
            selected = pred[pred["selected_model"]]
            fold_expectancies.append(float(selected["net_r"].mean()) if not selected.empty else float("nan"))
            for seed, value in seed_exps.items():
                seed_fold_values[seed].append(value)

        oos = pd.concat(oos_parts, ignore_index=True).sort_values(["signal_time", "venue", "symbol"])
        all_oos.append(oos)
        selected = oos[oos["selected_model"]].copy()
        realized = base._realize_nonoverlap(selected)
        financially_realized = corrected._financial_simulation_independent_by_venue(realized)
        executed = financially_realized[financially_realized["executed"]].copy() if not financially_realized.empty else financially_realized
        if not executed.empty:
            executed["candidate"] = candidate
            all_candidate_executed.append(executed)
            family_tables.append(_family_breakdown(executed, candidate))

        venue_results: dict[str, dict] = {}
        for venue in VENUES:
            vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
            metrics = base._venue_metrics(vt)
            gates = base._venue_gate(metrics)
            venue_results[venue] = {"metrics": metrics, "gates": gates}

        positive_fold_fraction = _positive_fraction(fold_expectancies)
        seed_expectancies: list[float] = []
        for seed in V41_SEEDS:
            vals = [v for v in seed_fold_values[seed] if np.isfinite(v)]
            seed_expectancies.append(float(np.mean(vals)) if vals else float("nan"))
        positive_seed_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in seed_expectancies]))
        cross_ok = positive_fold_fraction >= 0.60 and positive_seed_fraction >= (2.0 / 3.0)
        venue_ok = all(bool(venue_results[v]["gates"]["venue_pass"]) for v in VENUES)
        passed = bool(cross_ok and venue_ok)

        summary = {
            "candidate": candidate,
            "oos_events": int(len(oos)),
            "model_selected_events": int(len(selected)),
            "nonoverlap_events": int(len(realized)),
            "financially_executed_events": int(len(executed)),
            "positive_fold_fraction": positive_fold_fraction,
            "seed_expectancies_r": seed_expectancies,
            "positive_seed_fraction": positive_seed_fraction,
            "mean_expert_fraction": float(oos["expert_used_v41"].mean()) if len(oos) else 0.0,
            "all_venue_gates_pass": venue_ok,
            "cross_fold_seed_gate_pass": cross_ok,
            "candidate_pass": passed,
            "venue_results": venue_results,
        }
        candidate_summaries.append(summary)
        if passed:
            winner = candidate
            winner_trades = executed
            winner_venue_results = venue_results
            break

    oos_all = pd.concat(all_oos, ignore_index=True, sort=False) if all_oos else pd.DataFrame()
    executed_all = pd.concat(all_candidate_executed, ignore_index=True, sort=False) if all_candidate_executed else pd.DataFrame()
    family_all = pd.concat(family_tables, ignore_index=True, sort=False) if family_tables else pd.DataFrame()

    oos_all.to_csv(outdir / "oos_predictions_v41.csv.gz", index=False, compression="gzip")
    winner_trades.to_csv(outdir / "trades_v41.csv.gz", index=False, compression="gzip")
    executed_all.to_csv(outdir / "trades_all_candidates_v41.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v41.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(outdir / "seed_hazard_diagnostics_v41.csv", index=False)
    data_manifest.to_csv(outdir / "data_manifest_v41.csv", index=False)
    family_all.to_csv(outdir / "event_family_results_v41.csv", index=False)
    _worst_groups(all_candidate_executed).to_csv(outdir / "worst_groups_v41.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "venue_results"} for s in candidate_summaries]).to_csv(
        outdir / "candidate_summary_v41.csv", index=False
    )

    decision = {
        "version": "v0.41",
        "experiment": "EVENT_SPECIFIC_DISCRETE_TIME_COMPETING_RISK",
        "decision": "V41_DEVELOPMENT_WINNER" if winner else "V41_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": winner,
        "candidate_order": list(V41_CANDIDATES),
        "selection_rule": "first/simplest candidate passing all frozen gates",
        "venues": list(VENUES),
        "symbols": list(SYMBOLS),
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "purged_folds": PURGED_FOLDS,
        "embargo_bars": EMBARGO_BARS,
        "base_roundtrip_bps": BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": STRESS_ROUNDTRIP_BPS,
        "competing_risk_policy": asdict(CompetingRiskPolicyV41()),
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "candidate_summaries": candidate_summaries,
        "winner_venue_results": winner_venue_results,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_feature_selection": False,
    }
    (outdir / "decision_v41.json").write_text(
        json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
