from __future__ import annotations

"""Run one frozen v0.44 common fold for all preregistered sampling variants."""

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
from research_bot.information_sampling_v44 import V44_VARIANTS, variant_column_v44
from research_bot.time_cluster_bootstrap_v43 import timestamp_cluster_block_resample_v43

ROOT = Path(__file__).resolve().parents[1]
V42 = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
spec = importlib.util.spec_from_file_location("v42_for_v44_fold", V42)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 helper")
v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v42)
base = v42.base

V44_FEATURES: tuple[str, ...] = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)


def _window_slice(events: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal = pd.to_datetime(events["signal_time"], utc=True)
    exit_ = pd.to_datetime(events["exit_time"], utc=True)
    cal_start = pd.Timestamp(row["cal_start"])
    pretest_cut = pd.Timestamp(row["pretest_cut"])
    test_start = pd.Timestamp(row["test_start"])
    test_end = pd.Timestamp(row["test_end"])
    fit = events[(signal < cal_start) & (exit_ < cal_start)].copy()
    cal = events[(signal >= cal_start) & (signal < pretest_cut) & (exit_ < test_start)].copy()
    test = events[(signal >= test_start) & (signal <= test_end)].copy()
    return fit, cal, test


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
        "V41_HISTGB_CAUSE_SPECIFIC", perturbed, V44_FEATURES, int(seed), cr
    )
    cal_point = predict_competing_risks_vectorized_v41(bundle, cal)
    test_point = predict_competing_risks_vectorized_v41(bundle, test)
    pred = calibrate_expected_r_bounds_v41(cal, cal_point, test_point, cr)
    pred["perturbation_seed_v44"] = int(seed)
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

    status = json.loads((indir / "prep_status_v44.json").read_text(encoding="utf-8"))
    if status.get("status") != "READY" or status.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 prepared data not READY/fail-closed")
    events = pd.read_pickle(indir / "events_v44.pkl.gz", compression="gzip")
    fold_index = pd.read_csv(indir / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    common_row = fold_index.loc[fold_index["fold"].astype(int).eq(int(args.fold))]
    if len(common_row) != 1:
        raise RuntimeError(f"missing common fold boundary {args.fold}")
    common = common_row.iloc[0]
    dq = DataQualityPolicyV43()

    all_predictions: list[pd.DataFrame] = []
    asset_rows: list[dict] = []
    perturb_rows: list[dict] = []
    fold_variant_rows: list[dict] = []

    for variant in V44_VARIANTS:
        sample_col = variant_column_v44(variant)
        variant_events = events.loc[events[sample_col].astype(bool)].copy()
        fit_all, cal_all, test_all = _window_slice(variant_events, common)

        predictions: list[pd.DataFrame] = []
        variant_target_skills: list[float] = []
        variant_stop_skills: list[float] = []
        assets = sorted(set(test_all["symbol"].astype(str)))

        for symbol in assets:
            fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy()
            cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy()
            test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy()
            support = training_support_v43(fit, dq)
            if not support["supported"] or len(cal) < 50 or test.empty:
                asset_rows.append({
                    "variant": variant, "fold": int(args.fold), "symbol": symbol, **support,
                    "fit_events": int(len(fit)), "calibration_events": int(len(cal)),
                    "test_events": int(len(test)), "status": "INSUFFICIENT_SUPPORT",
                })
                continue

            baseline = empirical_hazard_baseline_v43(fit, test)
            perturb_outputs: list[pd.DataFrame] = []
            for seed in V43_PERTURBATION_SEEDS:
                pred, bootdiag = _fit_one_perturbation(fit, cal, test, int(seed), dq)
                perturb_outputs.append(pred)
                sel = pred["selected_v41"].astype(bool)
                perturb_rows.append({
                    "variant": variant, "fold": int(args.fold), "symbol": symbol,
                    "seed": int(seed), "fit_events": int(len(fit)),
                    "calibration_events": int(len(cal)), "test_events": int(len(test)),
                    "median_independent_selected_events": int(sel.sum()),
                    "selected_expectancy_r": float(pred.loc[sel, "net_r"].mean()) if int(sel.sum()) else None,
                    **bootdiag,
                })

            combined = median_seed_prediction_v41(perturb_outputs, CompetingRiskPolicyV41())
            stable_count = np.sum(
                np.stack([x["selected_v41"].to_numpy(dtype=bool) for x in perturb_outputs], axis=0), axis=0
            )
            combined["perturbation_selected_count_v44"] = stable_count.astype(int)
            combined["perturbation_agreement_v44"] = stable_count.astype(float) / len(V43_PERTURBATION_SEEDS)
            combined["median_admissible_v44"] = combined["selected_v41"].astype(bool)
            combined["selected_model"] = combined["median_admissible_v44"] & combined["perturbation_selected_count_v44"].ge(2)
            combined["variant"] = variant
            combined["candidate"] = f"V44_{variant}"
            combined["fold"] = int(args.fold)
            combined["expected_r"] = combined["expected_r_v41"]
            combined["lower_expected_r"] = combined["lower_expected_r_v41"]
            combined["upper_expected_r"] = combined["upper_expected_r_v41"]
            combined["baseline_p_target_v44"] = baseline["baseline_p_target_v43"].to_numpy(dtype=float)
            combined["baseline_p_stop_v44"] = baseline["baseline_p_stop_v43"].to_numpy(dtype=float)
            combined["baseline_source_v44"] = baseline["baseline_source_v43"].astype(str).to_numpy()

            y_target = test["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
            y_stop = test["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
            tskill = brier_skill_v43(y_target, combined["p_target_v41"], combined["baseline_p_target_v44"])
            sskill = brier_skill_v43(y_stop, combined["p_stop_v41"], combined["baseline_p_stop_v44"])
            variant_target_skills.append(tskill)
            variant_stop_skills.append(sskill)
            selected = combined["selected_model"].astype(bool)
            asset_rows.append({
                "variant": variant, "fold": int(args.fold), "symbol": symbol, **support,
                "fit_events": int(len(fit)), "calibration_events": int(len(cal)),
                "test_events": int(len(test)), "status": "MODELED",
                "target_brier_skill": float(tskill), "stop_brier_skill": float(sskill),
                "median_admissible_events": int(combined["median_admissible_v44"].sum()),
                "selected_events": int(selected.sum()),
                "selected_expectancy_r": float(combined.loc[selected, "net_r"].mean()) if int(selected.sum()) else None,
            })
            predictions.append(combined)

        if predictions:
            variant_oos = pd.concat(predictions, ignore_index=True, sort=False).sort_values(
                ["signal_time", "symbol", "venue"], kind="mergesort"
            )
            all_predictions.append(variant_oos)
            selected = variant_oos.loc[variant_oos["selected_model"].astype(bool)]
            median_admissible = variant_oos.loc[variant_oos["median_admissible_v44"].astype(bool)]
        else:
            variant_oos = events.iloc[0:0].copy()
            selected = variant_oos
            median_admissible = variant_oos

        fold_variant_rows.append({
            "variant": variant,
            "fold": int(args.fold),
            "common_test_start": common["test_start"],
            "common_test_end": common["test_end"],
            "fit_events_after_sampling": int(len(fit_all)),
            "calibration_events_after_sampling": int(len(cal_all)),
            "test_events_after_sampling": int(len(test_all)),
            "modeled_assets": int(sum(r.get("variant") == variant and r.get("fold") == int(args.fold) and r.get("status") == "MODELED" for r in asset_rows)),
            "oos_predictions": int(len(variant_oos)),
            "median_admissible_events": int(len(median_admissible)),
            "selected_events": int(len(selected)),
            "selected_expectancy_r": float(selected["net_r"].mean()) if not selected.empty else None,
            "median_asset_target_brier_skill": float(np.median(variant_target_skills)) if variant_target_skills else None,
            "median_asset_stop_brier_skill": float(np.median(variant_stop_skills)) if variant_stop_skills else None,
            "target_skill_positive_asset_fraction": float(np.mean(np.asarray(variant_target_skills) > 0.0)) if variant_target_skills else 0.0,
            "stop_skill_positive_asset_fraction": float(np.mean(np.asarray(variant_stop_skills) > 0.0)) if variant_stop_skills else 0.0,
            "kraken_touched": False,
        })

    if all_predictions:
        oos_all = pd.concat(all_predictions, ignore_index=True, sort=False).sort_values(
            ["variant", "signal_time", "symbol", "venue"], kind="mergesort"
        )
    else:
        oos_all = events.iloc[0:0].copy()
    oos_all.to_pickle(outdir / f"oos_fold_{args.fold}_v44.pkl.gz", compression="gzip")
    pd.DataFrame(asset_rows).to_csv(outdir / f"asset_diag_fold_{args.fold}_v44.csv", index=False)
    pd.DataFrame(perturb_rows).to_csv(outdir / f"perturb_diag_fold_{args.fold}_v44.csv", index=False)
    pd.DataFrame(fold_variant_rows).to_csv(outdir / f"fold_variant_diag_{args.fold}_v44.csv", index=False)
    (outdir / f"fold_diag_{args.fold}_v44.json").write_text(
        json.dumps({"fold": int(args.fold), "variants": fold_variant_rows, "kraken_touched": False}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"fold": int(args.fold), "variants": fold_variant_rows, "kraken_touched": False}, indent=2, default=str))


if __name__ == "__main__":
    main()
