from __future__ import annotations

"""Frozen v0.40 two-stage hurdle development characterization.

Reuses the frozen v0.39 mother-strategy features, event labels, market window,
purged folds, financial simulator and qualification gates.  The only scientific
change is the preregistered v0.40 learning decomposition in
research_bot.two_stage_hurdle_v40.
"""

import argparse
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.financial_system_v39 import FinancialRiskPolicyV39, ValidationPolicyV39
from research_bot.mother_strategy_v39 import NEURAL_FEATURES_V39
from research_bot.two_stage_hurdle_v40 import (
    HurdlePolicyV40,
    V40_CANDIDATES,
    V40_SEEDS,
    fit_predict_seed_v40,
    median_seed_prediction_v40,
    preregistration_manifest_v40,
    split_probability_and_conformal_calibration,
)

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


base = _load(V39_BASE, "v39_base_for_v40")
corrected = _load(V39_CORRECTED, "v39_corrected_for_v40")

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


def _feature_columns() -> list[str]:
    cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        cols.extend([c, f"{c}__missing"])
    return cols


def _predict_fold(candidate: str, fold: dict, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[int, float], list[dict], dict]:
    prob_cal, conf_cal = split_probability_and_conformal_calibration(fold["cal"], HurdlePolicyV40())
    if len(prob_cal) < 30 or len(conf_cal) < 30:
        raise RuntimeError("v0.40 calibration subsegments are too small")

    seed_outputs: list[pd.DataFrame] = []
    seed_diags: list[dict] = []
    seed_expectancies: dict[int, float] = {}
    for seed in V40_SEEDS:
        out, diag = fit_predict_seed_v40(
            candidate,
            fold["fit"],
            prob_cal,
            conf_cal,
            fold["test"],
            feature_cols,
            seed,
            HurdlePolicyV40(),
        )
        seed_outputs.append(out)
        seed_diags.append({**diag, "fold": int(fold["fold"])})
        selected = out[out["selected_v40"]]
        seed_expectancies[seed] = float(selected["net_r"].mean()) if not selected.empty else float("nan")

    combined = median_seed_prediction_v40(seed_outputs, HurdlePolicyV40())
    combined["fold"] = int(fold["fold"])
    combined["expected_r"] = combined["expected_r_v40"]
    combined["lower_expected_r"] = combined["lower_expected_r_v40"]
    combined["upper_expected_r"] = combined["upper_expected_r_v40"]
    combined["selected_model"] = combined["selected_v40"]

    selected = combined[combined["selected_model"]]
    diag = {
        "candidate": candidate,
        "fold": int(fold["fold"]),
        "fit_events": int(len(fold["fit"])),
        "calibration_events": int(len(fold["cal"])),
        "probability_calibration_events": int(len(prob_cal)),
        "conformal_calibration_events": int(len(conf_cal)),
        "test_events": int(len(fold["test"])),
        "selected_events": int(len(selected)),
        "selected_expectancy_r": float(selected["net_r"].mean()) if not selected.empty else None,
        "median_seed_brier": float(np.median([d["brier"] for d in seed_diags])),
        "median_seed_baseline_brier": float(np.median([d["baseline_brier"] for d in seed_diags])),
        "brier_skill_seed_fraction": float(np.mean([d["brier_skill_positive"] for d in seed_diags])),
        "calibration_start": fold["cal_start"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
    }
    return combined, seed_expectancies, seed_diags, diag


def _positive_fold_fraction(values: list[float]) -> float:
    if len(values) != PURGED_FOLDS:
        raise RuntimeError(f"expected {PURGED_FOLDS} folds, got {len(values)}")
    return float(np.mean([bool(np.isfinite(v) and v > 0.0) for v in values]))


def _independent_financial_simulation(realized: pd.DataFrame) -> pd.DataFrame:
    if realized.empty:
        return realized.copy()
    parts: list[pd.DataFrame] = []
    for venue in VENUES:
        z = realized[realized["venue"] == venue].copy()
        if z.empty:
            continue
        parts.append(base._financial_simulation(z))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else realized.iloc[0:0].copy()


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

    manifest = preregistration_manifest_v40()
    if manifest["kraken_touched"] is not False:
        raise RuntimeError("v0.40 holdout governance invalid")

    panel, events, data_manifest = base._prepare_panel()
    if events.empty:
        raise RuntimeError("v0.40 generated no candidate events")
    folds = base._folds(events)
    if len(folds) != PURGED_FOLDS:
        raise RuntimeError("v0.40 requires all five frozen purged folds")
    feature_cols = _feature_columns()

    candidate_summaries: list[dict] = []
    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    all_oos: list[pd.DataFrame] = []
    all_candidate_executed: list[pd.DataFrame] = []
    winner: str | None = None
    winner_trades = pd.DataFrame()
    winner_venue_results: dict[str, dict] = {}

    for candidate in V40_CANDIDATES:
        oos_parts: list[pd.DataFrame] = []
        fold_expectancies: list[float] = []
        seed_selected_r: dict[int, list[np.ndarray]] = {s: [] for s in V40_SEEDS}
        brier_skill_fractions: list[float] = []

        for fold in folds:
            pred, seed_exps, seed_diags, diag = _predict_fold(candidate, fold, feature_cols)
            oos_parts.append(pred)
            fold_rows.append(diag)
            seed_rows.extend(seed_diags)
            brier_skill_fractions.append(float(diag["brier_skill_seed_fraction"]))

            selected = pred[pred["selected_model"]]
            fold_expectancies.append(float(selected["net_r"].mean()) if not selected.empty else float("nan"))
            for seed, value in seed_exps.items():
                seed_rows_for_fold = [d for d in seed_diags if int(d["seed"]) == int(seed)]
                if seed_rows_for_fold:
                    pass
                # The seed expectation is recorded per fold; NaN remains non-positive.
                seed_selected_r[seed].append(np.asarray([value], dtype=float))

        oos = pd.concat(oos_parts, ignore_index=True).sort_values(["signal_time", "venue", "symbol"])
        all_oos.append(oos)
        selected = oos[oos["selected_model"]].copy()
        realized = base._realize_nonoverlap(selected)
        financially_realized = _independent_financial_simulation(realized)
        executed = financially_realized[financially_realized["executed"]].copy() if not financially_realized.empty else financially_realized
        if not executed.empty:
            executed["candidate"] = candidate
            all_candidate_executed.append(executed)

        venue_results: dict[str, dict] = {}
        for venue in VENUES:
            vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
            metrics = base._venue_metrics(vt)
            gates = base._venue_gate(metrics)
            venue_results[venue] = {"metrics": metrics, "gates": gates}

        positive_fold_fraction = _positive_fold_fraction(fold_expectancies)
        seed_expectancies: list[float] = []
        for seed in V40_SEEDS:
            vals = [float(a[0]) for a in seed_selected_r[seed] if len(a) and np.isfinite(float(a[0]))]
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
            "mean_brier_skill_seed_fraction": float(np.mean(brier_skill_fractions)),
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

    oos_all.to_csv(outdir / "oos_predictions_v40.csv.gz", index=False, compression="gzip")
    winner_trades.to_csv(outdir / "trades_v40.csv.gz", index=False, compression="gzip")
    executed_all.to_csv(outdir / "trades_all_candidates_v40.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v40.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(outdir / "seed_calibration_diagnostics_v40.csv", index=False)
    data_manifest.to_csv(outdir / "data_manifest_v40.csv", index=False)
    _worst_groups(all_candidate_executed).to_csv(outdir / "worst_groups_v40.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "venue_results"} for s in candidate_summaries]).to_csv(
        outdir / "candidate_summary_v40.csv", index=False
    )

    decision = {
        "version": "v0.40",
        "experiment": "TWO_STAGE_META_LABEL_CONDITIONAL_NET_R_HURDLE",
        "decision": "V40_DEVELOPMENT_WINNER" if winner else "V40_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": winner,
        "candidate_order": list(V40_CANDIDATES),
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
        "hurdle_policy": asdict(HurdlePolicyV40()),
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
    (outdir / "decision_v40.json").write_text(
        json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
