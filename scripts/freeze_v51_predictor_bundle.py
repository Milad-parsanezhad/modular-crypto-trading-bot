from __future__ import annotations

"""Reconstruct, verify and serialize the unique v0.51 production predictor bundle.

This script is reproducibility infrastructure only. It consumes already-spent
canonical v0.44/v0.47 evidence and may not produce a v0.51 economic decision.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import V43_DEVELOPMENT_VENUES
from research_bot.economic_state_v46 import (
    R1,
    R1_STATES,
    V46ModelPolicy,
    fit_multinomial_v46,
    predict_v46,
)
from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.predictor_identity_v51 import (
    PredictorVerificationV51,
    V51_CANONICAL_V44_RUN,
    V51_CANONICAL_V47_HEAD,
    V51_CANONICAL_V47_RUN,
    V51_PREDICTOR_BUNDLE_VERSION,
    V51_PRODUCTION_FOLD,
    V51_PRODUCTION_SYMBOLS,
    V51_REPAIRED_PROSPECTIVE_START,
    canonical_json_sha256,
    sha256_file,
    verification_manifest,
)
from research_bot.probability_calibration_v47 import (
    C1,
    V47CalibrationPolicy,
    expected_r_v47,
    probability_frame_v47,
    probability_matrix_v47,
    temperature_apply_v47,
)

FEATURES = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)
IDENTITY = ["venue", "symbol", "series_id", "signal_time", "entry_time"]


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _window_slice(events: pd.DataFrame, row: pd.Series):
    signal = pd.to_datetime(events["signal_time"], utc=True)
    exit_ = pd.to_datetime(events["exit_time"], utc=True)
    cal_start = _utc(row["cal_start"])
    pretest_cut = _utc(row["pretest_cut"])
    test_start = _utc(row["test_start"])
    test_end = _utc(row["test_end"])
    fit = events[(signal < cal_start) & (exit_ < cal_start)].copy()
    cal = events[(signal >= cal_start) & (signal < pretest_cut) & (exit_ < test_start)].copy()
    test = events[(signal >= test_start) & (signal <= test_end)].copy()
    return fit, cal, test


def _verify_source(v44: Path, v47: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_v44 = ["events_v44.pkl.gz", "fold_index_v44.csv", "prepared_hashes_v44.json", "prep_status_v44.json"]
    for name in required_v44:
        if not (v44 / name).is_file():
            raise RuntimeError(f"missing canonical v0.44 file: {name}")
    prep = json.loads((v44 / "prep_status_v44.json").read_text(encoding="utf-8"))
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("canonical v0.44 source is not READY/fail-closed")
    if int(prep.get("workflow_run_id", V51_CANONICAL_V44_RUN)) != V51_CANONICAL_V44_RUN:
        raise RuntimeError("unexpected canonical v0.44 run")
    hashes = json.loads((v44 / "prepared_hashes_v44.json").read_text(encoding="utf-8"))
    if sha256_file(v44 / "events_v44.pkl.gz") != hashes.get("events_v44_pickle_sha256"):
        raise RuntimeError("canonical v0.44 event-table hash mismatch")

    required_v47 = [
        "oos_predictions_v47.csv.gz", "calibration_parameters_v47.csv",
        "support_manifest_v47.csv", "output_hashes_v47.json", "execution_provenance_v47.json",
    ]
    for name in required_v47:
        if not (v47 / name).is_file():
            raise RuntimeError(f"missing canonical v0.47 file: {name}")
    out_hashes = json.loads((v47 / "output_hashes_v47.json").read_text(encoding="utf-8"))
    for name in ("oos_predictions_v47.csv.gz", "calibration_parameters_v47.csv", "support_manifest_v47.csv"):
        if out_hashes.get(name) != sha256_file(v47 / name):
            raise RuntimeError(f"canonical v0.47 artifact hash mismatch: {name}")
    provenance = json.loads((v47 / "execution_provenance_v47.json").read_text(encoding="utf-8"))
    if int(provenance.get("workflow_run_id")) != V51_CANONICAL_V47_RUN:
        raise RuntimeError("unexpected canonical v0.47 run")
    if provenance.get("head_sha") != V51_CANONICAL_V47_HEAD:
        raise RuntimeError("unexpected canonical v0.47 head")
    if provenance.get("kraken_touched") is not False or provenance.get("paper_execution") is not False or provenance.get("live_execution") is not False:
        raise RuntimeError("canonical v0.47 safety provenance invalid")

    events = pd.read_pickle(v44 / "events_v44.pkl.gz", compression="gzip")
    folds = pd.read_csv(v44 / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    calibration = pd.read_csv(v47 / "calibration_parameters_v47.csv")
    predictions = pd.read_csv(v47 / "oos_predictions_v47.csv.gz")
    return events, folds, calibration, predictions


def _canonical_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for col in ("signal_time", "entry_time"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="raise")
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v44-dir", required=True)
    parser.add_argument("--v47-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    v44 = Path(args.v44_dir)
    v47 = Path(args.v47_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    events, folds, calibration, canonical_predictions = _verify_source(v44, v47)
    if sorted(folds["fold"].astype(int).unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("v0.51 identity freeze requires exact five canonical v0.47 folds")
    fold_row = folds.loc[folds["fold"].astype(int).eq(V51_PRODUCTION_FOLD)].iloc[0]
    fit_all, cal_all, test_all = _window_slice(events, fold_row)

    missing_features = sorted(set(FEATURES).difference(events.columns))
    if missing_features:
        raise RuntimeError(f"canonical v0.44 events missing v0.47 features: {missing_features}")

    policy = V46ModelPolicy()
    cal_policy = V47CalibrationPolicy()
    bundles: dict[str, dict] = {}
    verification_rows: list[PredictorVerificationV51] = []
    verification_table: list[dict] = []

    canonical_predictions = _canonical_key_frame(canonical_predictions)

    for symbol in V51_PRODUCTION_SYMBOLS:
        fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy()
        cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy()
        test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy()
        if fit.empty or cal.empty or test.empty:
            raise RuntimeError(f"missing fold-5 fit/cal/test support for {symbol}")

        model, state_means = fit_multinomial_v46(fit, FEATURES, R1, policy)
        base_test = predict_v46(model, test, FEATURES, R1_STATES).reset_index(drop=True)
        base_matrix = probability_matrix_v47(base_test, R1_STATES, suffix="v46")

        temp_row = calibration.loc[
            calibration["arm"].astype(str).eq(C1)
            & calibration["fold"].astype(int).eq(V51_PRODUCTION_FOLD)
            & calibration["symbol"].astype(str).eq(symbol)
        ]
        if len(temp_row) != 1:
            raise RuntimeError(f"canonical C1 temperature is not unique for {symbol}")
        temperature = float(temp_row.iloc[0]["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise RuntimeError(f"invalid canonical C1 temperature for {symbol}")

        calibrated = temperature_apply_v47(base_matrix, temperature, cal_policy)
        native_p = probability_frame_v47(calibrated, R1_STATES)
        expected = expected_r_v47(native_p, state_means).to_numpy(dtype=float)
        reconstructed = test.reset_index(drop=True).loc[:, IDENTITY].copy()
        reconstructed["reconstructed_expected_r"] = expected
        reconstructed["reconstructed_selected"] = np.isfinite(expected) & (expected > 0.0)
        reconstructed = _canonical_key_frame(reconstructed)

        canonical = canonical_predictions.loc[
            canonical_predictions["arm"].astype(str).eq(C1)
            & canonical_predictions["fold"].astype(int).eq(V51_PRODUCTION_FOLD)
            & canonical_predictions["symbol"].astype(str).eq(symbol),
            IDENTITY + ["expected_r_v47", "selected_model"],
        ].copy()
        if canonical.empty:
            raise RuntimeError(f"canonical v0.47 C1 fold-5 predictions missing for {symbol}")
        canonical = _canonical_key_frame(canonical)

        merged = canonical.merge(reconstructed, on=IDENTITY, how="outer", indicator=True, validate="one_to_one")
        identity_match = bool((merged["_merge"] == "both").all() and len(merged) == len(canonical) == len(reconstructed))
        common = merged.loc[merged["_merge"].eq("both")].copy()
        if common.empty:
            max_error = mean_error = float("inf")
            selected_match = 0.0
        else:
            diff = np.abs(pd.to_numeric(common["expected_r_v47"], errors="coerce").to_numpy(dtype=float) - common["reconstructed_expected_r"].to_numpy(dtype=float))
            max_error = float(np.max(diff)) if len(diff) else float("inf")
            mean_error = float(np.mean(diff)) if len(diff) else float("inf")
            canonical_selected = common["selected_model"].astype(bool).to_numpy()
            reconstructed_selected = common["reconstructed_selected"].astype(bool).to_numpy()
            selected_match = float(np.mean(canonical_selected == reconstructed_selected)) if len(common) else 0.0

        verification = PredictorVerificationV51(
            symbol=symbol,
            fold=V51_PRODUCTION_FOLD,
            rows=int(len(canonical)),
            canonical_temperature=temperature,
            max_abs_expected_r_error=max_error,
            mean_abs_expected_r_error=mean_error,
            selected_model_match_fraction=selected_match,
            identity_match=identity_match,
        )
        verification_rows.append(verification)
        verification_table.append({
            "symbol": symbol,
            "fold": V51_PRODUCTION_FOLD,
            "canonical_rows": int(len(canonical)),
            "reconstructed_rows": int(len(reconstructed)),
            "identity_match": identity_match,
            "canonical_temperature": temperature,
            "max_abs_expected_r_error": max_error,
            "mean_abs_expected_r_error": mean_error,
            "selected_model_match_fraction": selected_match,
            "passed": verification.passed,
        })

        bundles[symbol] = {
            "symbol": symbol,
            "fold": V51_PRODUCTION_FOLD,
            "model": model,
            "state_means": {str(k): float(v) for k, v in state_means.items()},
            "temperature": temperature,
            "features": FEATURES,
            "states": tuple(R1_STATES),
            "fit_event_count": int(len(fit)),
            "calibration_event_count": int(len(cal)),
            "verification_test_event_count": int(len(test)),
            "cal_start": _utc(fold_row["cal_start"]).isoformat(),
            "pretest_cut": _utc(fold_row["pretest_cut"]).isoformat(),
            "test_start": _utc(fold_row["test_start"]).isoformat(),
            "test_end": _utc(fold_row["test_end"]).isoformat(),
        }

    manifest = verification_manifest(verification_rows)
    if manifest["all_verified"] is not True:
        pd.DataFrame(verification_table).to_csv(out / "predictor_verification_v51.csv", index=False)
        (out / "predictor_manifest_v51.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError("v0.51 production predictor reconstruction did not verify against canonical v0.47")

    bundle_payload = {
        "bundle_version": V51_PREDICTOR_BUNDLE_VERSION,
        "production_fold": V51_PRODUCTION_FOLD,
        "production_symbols": V51_PRODUCTION_SYMBOLS,
        "repaired_prospective_start": V51_REPAIRED_PROSPECTIVE_START,
        "features": FEATURES,
        "states": tuple(R1_STATES),
        "canonical_v44_events_sha256": sha256_file(v44 / "events_v44.pkl.gz"),
        "canonical_v47_predictions_sha256": sha256_file(v47 / "oos_predictions_v47.csv.gz"),
        "canonical_v47_calibration_sha256": sha256_file(v47 / "calibration_parameters_v47.csv"),
        "models": bundles,
    }
    bundle_path = out / "v51_production_predictor.joblib"
    joblib.dump(bundle_payload, bundle_path, compress=3)

    manifest["bundle_file"] = bundle_path.name
    manifest["bundle_sha256"] = sha256_file(bundle_path)
    manifest["canonical_v44_events_sha256"] = bundle_payload["canonical_v44_events_sha256"]
    manifest["canonical_v47_predictions_sha256"] = bundle_payload["canonical_v47_predictions_sha256"]
    manifest["canonical_v47_calibration_sha256"] = bundle_payload["canonical_v47_calibration_sha256"]
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)

    pd.DataFrame(verification_table).to_csv(out / "predictor_verification_v51.csv", index=False)
    (out / "predictor_manifest_v51.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    hashes = {
        "v51_production_predictor.joblib": sha256_file(bundle_path),
        "predictor_verification_v51.csv": sha256_file(out / "predictor_verification_v51.csv"),
        "predictor_manifest_v51.json": sha256_file(out / "predictor_manifest_v51.json"),
    }
    (out / "output_hashes_v51_identity.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
