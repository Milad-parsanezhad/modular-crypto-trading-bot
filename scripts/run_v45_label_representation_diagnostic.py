from __future__ import annotations

"""Execute the frozen, non-promotional v0.45 Route-C diagnostic."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.label_representation_diagnostic_v45 import (
    assign_common_test_folds_v45,
    brier_decomposition_v45,
    cause_age_hazard_strata_v45,
    feature_information_v45,
    label_distribution_v45,
    timeout_diagnostics_v45,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PREREG_GIT_BLOB = "3cf704d363e99976fdec9a9d9708e67970bf5737"
V44_RUN_ID = 34700944062
V44_HEAD_SHA = "5efb843d385d194c549a69dc4b2fd38d56ab906c"
V44_FINAL_ARTIFACT_ID = 10300676250
V44_FINAL_ARTIFACT_SHA256 = "f3834a6dda2758648f5e5ad3dc176ad89699d559f0d8076482ef8cf673002ff2"
V44_PREPARED_ARTIFACT_ID = 10299564398
V44_PREPARED_ARTIFACT_SHA256 = "a229a36f98cc1c84de1ae303f3e747be44138780937e5a327e3e6876eb737384"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_preregistration() -> str:
    prereg = ROOT / "docs" / "V45_LABEL_REPRESENTATION_DIAGNOSTIC_PREREGISTRATION.md"
    actual = subprocess.check_output(["git", "hash-object", str(prereg)], text=True).strip()
    if actual != EXPECTED_PREREG_GIT_BLOB:
        raise RuntimeError(f"v0.45 preregistration changed after freeze: {actual}")
    return actual


def _verify_source(prepared: Path, final: Path) -> dict:
    decision_path = final / "decision_v44.json"
    if not decision_path.exists():
        raise RuntimeError("missing canonical v0.44 decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE":
        raise RuntimeError("v0.45 Route C not activated by canonical v0.44 decision")
    if decision.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 holdout integrity violated")
    if decision.get("paper_execution") is not False or decision.get("live_execution") is not False:
        raise RuntimeError("v0.45 requires PAPER/LIVE disabled")
    if decision.get("winner") is not None or decision.get("passing_variants") not in ([], None):
        raise RuntimeError("v0.45 Route C source unexpectedly contains a winner")

    events_path = prepared / "events_v44.pkl.gz"
    fold_path = prepared / "fold_index_v44.csv"
    if not events_path.exists() or not fold_path.exists():
        raise RuntimeError("missing v0.44 prepared event/fold evidence")

    hash_path = final / "provenance" / "prepared_hashes_v44.json"
    if not hash_path.exists():
        hash_path = prepared / "prepared_hashes_v44.json"
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    actual_event_hash = _sha256(events_path)
    expected_event_hash = hashes.get("events_v44_pickle_sha256")
    if actual_event_hash != expected_event_hash:
        raise RuntimeError("prepared v0.44 event table hash mismatch")

    folds = pd.read_csv(fold_path)
    if len(folds) != 5 or folds["fold"].astype(int).nunique() != 5:
        raise RuntimeError("expected exactly five common v0.44 folds")

    return {
        "source_run_id": V44_RUN_ID,
        "source_head_sha": V44_HEAD_SHA,
        "source_final_artifact_id": V44_FINAL_ARTIFACT_ID,
        "source_final_artifact_sha256": V44_FINAL_ARTIFACT_SHA256,
        "source_prepared_artifact_id": V44_PREPARED_ARTIFACT_ID,
        "source_prepared_artifact_sha256": V44_PREPARED_ARTIFACT_SHA256,
        "events_v44_pickle_sha256": actual_event_hash,
        "expected_events_v44_pickle_sha256": expected_event_hash,
        "common_fold_count": 5,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
    }


def _calibration_tables(oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    bin_rows: list[pd.DataFrame] = []
    configs = (
        ("TARGET", "p_target_v41", "baseline_p_target_v44"),
        ("STOP", "p_stop_v41", "baseline_p_stop_v44"),
    )
    variants = ["S0_CLOCK_MOTHER_BASELINE", "S1_CUSUM_LAGGED_VOL", "S2_DIRECTIONAL_CHANGE_LAGGED_ATR"]

    for variant in variants:
        vx = oos.loc[oos["variant"].astype(str).eq(variant)].copy() if "variant" in oos else oos.iloc[0:0].copy()
        for fold in range(1, 6):
            fx = vx.loc[vx["fold"].astype(int).eq(fold)].copy() if not vx.empty else vx
            for cause, model_col, baseline_col in configs:
                y = fx["outcome"].astype(str).eq(cause).to_numpy(dtype=float) if not fx.empty else np.array([], dtype=float)
                for source, pcol in (("MODEL", model_col), ("TRAINING_EMPIRICAL_BASELINE", baseline_col)):
                    if fx.empty or pcol not in fx:
                        metrics, bins = brier_decomposition_v45([], [])
                    else:
                        metrics, bins = brier_decomposition_v45(y, pd.to_numeric(fx[pcol], errors="coerce"))
                    row = {"variant": variant, "fold": fold, "cause": cause, "forecast_source": source, **metrics}
                    summary_rows.append(row)
                    if not bins.empty:
                        bins = bins.copy()
                        bins.insert(0, "forecast_source", source)
                        bins.insert(0, "cause", cause)
                        bins.insert(0, "fold", fold)
                        bins.insert(0, "variant", variant)
                        bin_rows.append(bins)

    return pd.DataFrame(summary_rows), (
        pd.concat(bin_rows, ignore_index=True, sort=False)
        if bin_rows
        else pd.DataFrame(columns=["variant", "fold", "cause", "forecast_source", "bin", "lower", "upper", "n", "mean_prediction", "empirical_frequency"])
    )


def _diagnostic_summary(
    test_events: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_info: pd.DataFrame,
) -> dict:
    outcomes = test_events["outcome"].astype(str)
    label_summary = {
        "n": int(len(test_events)),
        "target_fraction": float(outcomes.eq("TARGET").mean()) if len(test_events) else None,
        "stop_fraction": float(outcomes.eq("STOP").mean()) if len(test_events) else None,
        "time_fraction": float(outcomes.eq("TIME").mean()) if len(test_events) else None,
    }

    cal_rows: list[dict] = []
    for (variant, cause, source), g in calibration.groupby(["variant", "cause", "forecast_source"], sort=True):
        valid = g.loc[pd.to_numeric(g["brier"], errors="coerce").notna()].copy()
        cal_rows.append({
            "variant": str(variant),
            "cause": str(cause),
            "forecast_source": str(source),
            "folds_with_data": int(len(valid)),
            "median_brier": float(valid["brier"].median()) if len(valid) else None,
            "median_reliability": float(valid["reliability"].median()) if len(valid) else None,
            "median_resolution": float(valid["resolution"].median()) if len(valid) else None,
        })

    supported = feature_info.loc[feature_info["target_supported"].astype(bool) | feature_info["stop_supported"].astype(bool)].copy()
    target_edges = pd.to_numeric(supported["target_auc_abs_edge"], errors="coerce").dropna()
    stop_edges = pd.to_numeric(supported["stop_auc_abs_edge"], errors="coerce").dropna()
    feature_summary = {
        "diagnostic_rows": int(len(feature_info)),
        "supported_rows": int(len(supported)),
        "median_target_auc_abs_edge": float(target_edges.median()) if len(target_edges) else None,
        "p90_target_auc_abs_edge": float(target_edges.quantile(0.90)) if len(target_edges) else None,
        "median_stop_auc_abs_edge": float(stop_edges.median()) if len(stop_edges) else None,
        "p90_stop_auc_abs_edge": float(stop_edges.quantile(0.90)) if len(stop_edges) else None,
    }

    return {
        "version": "v0.45",
        "experiment": "LABEL_REPRESENTATION_DIAGNOSTIC",
        "state": "DIAGNOSTIC_COMPLETE_NO_PROMOTION",
        "route": "C",
        "source_v44_decision": "V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "s0_common_test_label_summary": label_summary,
        "calibration_summary": cal_rows,
        "feature_information_summary": feature_summary,
        "promotion_authorized": False,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "next_stage_requires_new_preregistration": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    final = Path(args.final_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prereg_blob = _verify_preregistration()
    integrity = _verify_source(prepared, final)
    integrity["v45_preregistration_git_blob"] = prereg_blob
    integrity["v45_feature_list"] = list(V41_FEATURES)
    integrity["v45_feature_count"] = len(V41_FEATURES)

    events = pd.read_pickle(prepared / "events_v44.pkl.gz", compression="gzip")
    folds = pd.read_csv(prepared / "fold_index_v44.csv")
    test_events = assign_common_test_folds_v45(events, folds)

    labels = label_distribution_v45(test_events)
    hazards = cause_age_hazard_strata_v45(test_events)
    timeouts = timeout_diagnostics_v45(test_events)
    features = feature_information_v45(test_events, V41_FEATURES)

    oos = pd.read_csv(final / "oos_predictions_v44.csv.gz", compression="gzip", low_memory=False)
    calibration, calibration_bins = _calibration_tables(oos)

    summary = _diagnostic_summary(test_events, calibration, features)

    labels.to_csv(outdir / "label_distribution_v45.csv", index=False)
    hazards.to_csv(outdir / "cause_age_hazard_v45.csv", index=False)
    calibration.to_csv(outdir / "calibration_decomposition_v45.csv", index=False)
    calibration_bins.to_csv(outdir / "calibration_bins_v45.csv", index=False)
    features.to_csv(outdir / "feature_information_v45.csv", index=False)
    timeouts.to_csv(outdir / "timeout_diagnostics_v45.csv", index=False)
    (outdir / "source_integrity_v45.json").write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "diagnostic_summary_v45.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
