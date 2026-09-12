from __future__ import annotations

"""Run one frozen v0.43 fold from a screened prepared event table."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_DEVELOPMENT_VENUES,
    V43_PERTURBATION_SEEDS,
    brier_skill_v43,
    empirical_hazard_baseline_v43,
    training_support_v43,
)
from research_bot.event_competing_risk_v41 import (
    CompetingRiskPolicyV41,
    V41_FEATURES,
    calibrate_expected_r_bounds_v41,
    fit_competing_risk_bundle_v41,
    median_seed_prediction_v41,
)
from research_bot.event_competing_risk_vectorized_v41 import predict_competing_risks_vectorized_v41
from research_bot.time_cluster_bootstrap_v43 import timestamp_cluster_block_resample_v43

ROOT = Path(__file__).resolve().parents[1]
V42 = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
spec = importlib.util.spec_from_file_location("v42_for_v43_fold", V42)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 helper")
v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v42)
base = v42.base

V43_CANDIDATE = "V43_ASSET_CROSSVENUE_HISTGB"
V43_FEATURES: tuple[str, ...] = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)


def _fit_one_perturbation(
    fit: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    policy: DataQualityPolicyV43,
) -> tuple[pd.DataFrame, dict]:
    ordered = fit.sort_values(["signal_time", "venue"], kind="mergesort").reset_index(drop=True)
    perturbed, diag = timestamp_cluster_block_resample_v43(
        ordered,
        seed=int(seed),
        target_block_events=policy.perturbation_block_events,
        timestamp_col="signal_time",
        venue_col="venue",
    )
    cr = CompetingRiskPolicyV41()
    bundle = fit_competing_risk_bundle_v41(
        "V41_HISTGB_CAUSE_SPECIFIC", perturbed, V43_FEATURES, int(seed), cr
    )
    cal_point = predict_competing_risks_vectorized_v41(bundle, cal)
    test_point = predict_competing_risks_vectorized_v41(bundle, test)
    pred = calibrate_expected_r_bounds_v41(cal, cal_point, test_point, cr)
    pred["perturbation_seed_v43"] = int(seed)
    return pred, {
        "bootstrap_original_rows": diag.original_rows,
        "bootstrap_original_timestamps": diag.original_timestamps,
        "bootstrap_resampled_rows": diag.resampled_rows,
        "bootstrap_selected_blocks": diag.selected_blocks,
        "bootstrap_target_block_events": diag.target_block_events,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4, 5))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    status = json.loads((indir / "prep_status_v43.json").read_text(encoding="utf-8"))
    if status.get("status") != "READY":
        raise RuntimeError("v0.43 prepared data not READY")

    events = pd.read_pickle(indir / "events_v43.pkl.gz", compression="gzip")
    folds = base._folds(events)
    fold = next(x for x in folds if int(x["fold"]) == int(args.fold))
    policy = DataQualityPolicyV43()

    predictions: list[pd.DataFrame] = []
    asset_rows: list[dict] = []
    perturb_rows: list[dict] = []
    target_skills: list[float] = []
    stop_skills: list[float] = []

    assets = sorted(set(fold["test"]["symbol"].astype(str)))
    for symbol in assets:
        fit = fold["fit"].loc[fold["fit"]["symbol"].astype(str).eq(symbol)].copy()
        cal = fold["cal"].loc[fold["cal"]["symbol"].astype(str).eq(symbol)].copy()
        test = fold["test"].loc[fold["test"]["symbol"].astype(str).eq(symbol)].copy()
        support = training_support_v43(fit, policy)
        if not support["supported"] or len(cal) < 50 or test.empty:
            asset_rows.append({
                "fold": int(args.fold), "symbol": symbol, **support,
                "calibration_events": int(len(cal)), "test_events": int(len(test)),
                "status": "INSUFFICIENT_SUPPORT",
            })
            continue

        baseline = empirical_hazard_baseline_v43(fit, test)
        perturb_outputs: list[pd.DataFrame] = []
        for seed in V43_PERTURBATION_SEEDS:
            pred, bootdiag = _fit_one_perturbation(fit, cal, test, int(seed), policy)
            perturb_outputs.append(pred)
            sel = pred["selected_v41"].astype(bool)
            perturb_rows.append({
                "fold": int(args.fold),
                "symbol": symbol,
                "seed": int(seed),
                "fit_events": int(len(fit)),
                "calibration_events": int(len(cal)),
                "test_events": int(len(test)),
                "selected_events": int(sel.sum()),
                "selected_expectancy_r": float(pred.loc[sel, "net_r"].mean()) if int(sel.sum()) else None,
                **bootdiag,
            })

        combined = median_seed_prediction_v41(perturb_outputs, CompetingRiskPolicyV41())
        stable_count = np.sum(
            np.stack([x["selected_v41"].to_numpy(dtype=bool) for x in perturb_outputs], axis=0),
            axis=0,
        )
        combined["perturbation_selected_count_v43"] = stable_count.astype(int)
        combined["perturbation_agreement_v43"] = stable_count.astype(float) / len(V43_PERTURBATION_SEEDS)
        combined["selected_model"] = combined["selected_v41"].astype(bool) & combined["perturbation_selected_count_v43"].ge(2)
        combined["candidate"] = V43_CANDIDATE
        combined["fold"] = int(args.fold)
        combined["expected_r"] = combined["expected_r_v41"]
        combined["lower_expected_r"] = combined["lower_expected_r_v41"]
        combined["upper_expected_r"] = combined["upper_expected_r_v41"]
        combined["baseline_p_target_v43"] = baseline["baseline_p_target_v43"].to_numpy(dtype=float)
        combined["baseline_p_stop_v43"] = baseline["baseline_p_stop_v43"].to_numpy(dtype=float)
        combined["baseline_source_v43"] = baseline["baseline_source_v43"].astype(str).to_numpy()

        y_target = test["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
        y_stop = test["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
        tskill = brier_skill_v43(
            y_target, combined["p_target_v41"], combined["baseline_p_target_v43"]
        )
        sskill = brier_skill_v43(
            y_stop, combined["p_stop_v41"], combined["baseline_p_stop_v43"]
        )
        target_skills.append(tskill)
        stop_skills.append(sskill)
        selected = combined["selected_model"].astype(bool)
        asset_rows.append({
            "fold": int(args.fold),
            "symbol": symbol,
            **support,
            "calibration_events": int(len(cal)),
            "test_events": int(len(test)),
            "status": "MODELED",
            "target_brier_skill": float(tskill),
            "stop_brier_skill": float(sskill),
            "selected_events": int(selected.sum()),
            "selected_expectancy_r": float(combined.loc[selected, "net_r"].mean()) if int(selected.sum()) else None,
        })
        predictions.append(combined)

    if predictions:
        oos = pd.concat(predictions, ignore_index=True, sort=False).sort_values(
            ["signal_time", "symbol", "venue"], kind="mergesort"
        )
    else:
        oos = events.iloc[0:0].copy()
    oos.to_pickle(outdir / f"oos_fold_{args.fold}_v43.pkl.gz", compression="gzip")
    pd.DataFrame(asset_rows).to_csv(outdir / f"asset_diag_fold_{args.fold}_v43.csv", index=False)
    pd.DataFrame(perturb_rows).to_csv(outdir / f"perturb_diag_fold_{args.fold}_v43.csv", index=False)

    selected = oos.loc[oos.get("selected_model", pd.Series(False, index=oos.index)).astype(bool)] if not oos.empty else oos
    fold_diag = {
        "fold": int(args.fold),
        "modeled_assets": int(sum(r.get("status") == "MODELED" for r in asset_rows)),
        "test_events": int(len(oos)),
        "selected_events": int(len(selected)),
        "selected_expectancy_r": float(selected["net_r"].mean()) if not selected.empty else None,
        "median_asset_target_brier_skill": float(np.median(target_skills)) if target_skills else None,
        "median_asset_stop_brier_skill": float(np.median(stop_skills)) if stop_skills else None,
        "target_skill_positive_asset_fraction": float(np.mean(np.asarray(target_skills) > 0.0)) if target_skills else 0.0,
        "stop_skill_positive_asset_fraction": float(np.mean(np.asarray(stop_skills) > 0.0)) if stop_skills else 0.0,
        "kraken_touched": False,
        "perturbation_unit": "timestamp_cluster",
    }
    (outdir / f"fold_diag_{args.fold}_v43.json").write_text(
        json.dumps(fold_diag, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(fold_diag, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
