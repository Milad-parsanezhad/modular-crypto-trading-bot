from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import V43_DEVELOPMENT_VENUES
from research_bot.conditional_concept_drift_v49 import (
    BH_Q_V49,
    BOOTSTRAP_REPS_V49,
    BOOTSTRAP_SEED_V49,
    COMMON_STATES_V49,
    MIN_INTERACTION_PAIRS_V49,
    calibration_map_distance_v49,
    common_brier_losses_v49,
    fit_binary_calibration_map_v49,
    spearman_feature_residual_v49,
    standardized_cusum_v49,
    state_residual_v49,
)
from research_bot.economic_state_v46 import R1, R1_STATES, V46ModelPolicy, encode_state_v46, fit_multinomial_v46, predict_v46
from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.probability_calibration_v47 import (
    C1,
    V47CalibrationPolicy,
    common_three_state_probabilities_v47,
    fit_temperature_v47,
    probability_frame_v47,
    probability_matrix_v47,
    temperature_apply_v47,
)
from research_bot.temporal_regime_diagnostic_v48 import bh_adjust, fold_cluster_bootstrap_spearman, one_sided_sign_p

EXPECTED_V44_RUN = 34700944062
EXPECTED_V47_RUN = 34705324352
EXPECTED_V47_SCIENTIFIC_HEAD = "802b0cde38549617589b6a4a313949361dd9425c"
EXPECTED_V47_ARTIFACT_DIGEST = "sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118"
EXPECTED_V48_RUN = 34706819566
EXPECTED_V48_SCIENTIFIC_HEAD = "eb2e02f464db3ca58889a0474c453b9d378b66f4"
FEATURES = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def window_slice(events: pd.DataFrame, row: pd.Series):
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


def chronological_row_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 2:
        return frame.iloc[0:0].copy(), frame.copy()
    x = frame.copy()
    x["_v49_order"] = np.arange(len(x), dtype=int)
    x["_v49_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    x = x.sort_values(["_v49_time", "_v49_order"], kind="mergesort").drop(columns=["_v49_time", "_v49_order"]).reset_index(drop=True)
    cut = len(x) // 2
    return x.iloc[:cut].copy(), x.iloc[cut:].copy()


def calibrated_common_probabilities(model, temperature: float, frame: pd.DataFrame) -> pd.DataFrame:
    base = predict_v46(model, frame, FEATURES, R1_STATES).reset_index(drop=True)
    matrix = probability_matrix_v47(base, R1_STATES, suffix="v46")
    calibrated = temperature_apply_v47(matrix, temperature)
    native = probability_frame_v47(calibrated, R1_STATES)
    return common_three_state_probabilities_v47(native).reset_index(drop=True)


def calibration_excess(early: pd.DataFrame, late: pd.DataFrame, test: pd.DataFrame, pe: pd.DataFrame, pl: pd.DataFrame, pt: pd.DataFrame) -> float | None:
    state_excess: list[float] = []
    for state in COMMON_STATES_V49:
        col = f"p_{state.lower()}_v47"
        maps = []
        for part, probs in ((early, pe), (late, pl), (test, pt)):
            y = (part["outcome"].astype(str).to_numpy() == state).astype(int)
            maps.append(fit_binary_calibration_map_v49(y, probs[col].to_numpy(dtype=float)))
        if any(m is None for m in maps):
            return None
        internal = calibration_map_distance_v49(maps[0], maps[1])
        forward = calibration_map_distance_v49(maps[1], maps[2])
        state_excess.append(float(forward - internal))
    return float(np.median(state_excess))


def interaction_excess(early: pd.DataFrame, late: pd.DataFrame, test: pd.DataFrame, pe: pd.DataFrame, pl: pd.DataFrame, pt: pd.DataFrame) -> tuple[float | None, int]:
    pair_excess: list[float] = []
    for state in COMMON_STATES_V49:
        col = f"p_{state.lower()}_v47"
        re = state_residual_v49(early["outcome"].astype(str), pe[col], state)
        rl = state_residual_v49(late["outcome"].astype(str), pl[col], state)
        rt = state_residual_v49(test["outcome"].astype(str), pt[col], state)
        for feature in V41_FEATURES:
            a = spearman_feature_residual_v49(early[feature], re)
            b = spearman_feature_residual_v49(late[feature], rl)
            c = spearman_feature_residual_v49(test[feature], rt)
            if a is None or b is None or c is None:
                continue
            pair_excess.append(float(abs(b - c) - abs(a - b)))
    if len(pair_excess) < MIN_INTERACTION_PAIRS_V49:
        return None, len(pair_excess)
    return float(np.median(pair_excess)), len(pair_excess)


def loss_cusum_excess(early: pd.DataFrame, late: pd.DataFrame, test: pd.DataFrame, pe: pd.DataFrame, pl: pd.DataFrame, pt: pd.DataFrame) -> float | None:
    le = common_brier_losses_v49(early["outcome"].astype(str), pe)
    ll = common_brier_losses_v49(late["outcome"].astype(str), pl)
    lt = common_brier_losses_v49(test["outcome"].astype(str), pt)
    internal = standardized_cusum_v49(le, ll)
    forward = standardized_cusum_v49(ll, lt)
    if internal is None or forward is None:
        return None
    return float(forward - internal)


def context_rows(units: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for fold, g in units.groupby("fold", sort=True):
        rows.append({
            "dimension": "diagnostic_fold",
            "key": str(fold),
            "n": int(len(g)),
            "conditional_calibration_excess": float(pd.to_numeric(g["conditional_calibration_excess"], errors="coerce").median()),
            "interaction_excess": float(pd.to_numeric(g["interaction_excess"], errors="coerce").median()),
            "loss_cusum_excess": float(pd.to_numeric(g["loss_cusum_excess"], errors="coerce").median()),
            "expectancy_r": None,
        })
    if not trades.empty:
        for keys, group_cols in [("fold_x_venue", ["fold", "venue"]), ("fold_x_side", ["fold", "side"])]:
            for key, g in trades.groupby(group_cols, dropna=False, sort=True):
                vals = key if isinstance(key, tuple) else (key,)
                rows.append({
                    "dimension": keys,
                    "key": " | ".join(map(str, vals)),
                    "n": int(len(g)),
                    "conditional_calibration_excess": None,
                    "interaction_excess": None,
                    "loss_cusum_excess": None,
                    "expectancy_r": float(pd.to_numeric(g["net_r"], errors="coerce").mean()),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v44-dir", required=True)
    ap.add_argument("--v47-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    v44 = Path(args.v44_dir)
    v47 = Path(args.v47_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Exact source guards.
    prep = json.loads((v44 / "prep_status_v44.json").read_text())
    ph = json.loads((v44 / "prepared_hashes_v44.json").read_text())
    events_path = v44 / "events_v44.pkl.gz"
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 source not READY or Kraken firewall violated")
    if sha256(events_path) != ph.get("events_v44_pickle_sha256"):
        raise RuntimeError("v0.44 event hash mismatch")

    required_v47 = [
        "decision_v47.json", "execution_provenance_v47.json", "support_manifest_v47.csv",
        "forecast_metrics_v47.csv", "post_governor_trades_v47.csv.gz", "output_hashes_v47.json",
    ]
    for name in required_v47:
        if not (v47 / name).is_file():
            raise RuntimeError(f"missing canonical v0.47 file: {name}")
    hashes = json.loads((v47 / "output_hashes_v47.json").read_text())
    for name in ["decision_v47.json", "support_manifest_v47.csv", "forecast_metrics_v47.csv", "post_governor_trades_v47.csv.gz"]:
        if hashes.get(name) != sha256(v47 / name):
            raise RuntimeError(f"v0.47 hash mismatch: {name}")
    prov = json.loads((v47 / "execution_provenance_v47.json").read_text())
    if int(prov.get("workflow_run_id")) != EXPECTED_V47_RUN or prov.get("head_sha") != EXPECTED_V47_SCIENTIFIC_HEAD:
        raise RuntimeError("unexpected canonical v0.47 provenance")
    if prov.get("kraken_touched") is not False:
        raise RuntimeError("canonical v0.47 touched Kraken")

    events = pd.read_pickle(events_path, compression="gzip")
    folds = pd.read_csv(v44 / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    support = pd.read_csv(v47 / "support_manifest_v47.csv")
    metrics = pd.read_csv(v47 / "forecast_metrics_v47.csv")
    trades = pd.read_csv(v47 / "post_governor_trades_v47.csv.gz", low_memory=False)

    supported = parse_bool(support["supported"])
    canonical = support.loc[support["arm"].astype(str).eq(C1) & supported, ["fold", "symbol"]].drop_duplicates()
    canonical = canonical.sort_values(["fold", "symbol"]).reset_index(drop=True)
    if len(canonical) != 120:
        raise RuntimeError(f"unexpected canonical C1 unit count: {len(canonical)}")

    missing = sorted(set(FEATURES) - set(events.columns))
    if missing:
        raise RuntimeError(f"missing frozen predictor features: {missing}")

    base_policy = V46ModelPolicy()
    cal_policy = V47CalibrationPolicy()
    unit_rows: list[dict] = []

    for fold_no in range(1, 6):
        fr = folds.loc[folds["fold"].astype(int).eq(fold_no)].iloc[0]
        fit_all, cal_all, test_all = window_slice(events, fr)
        symbols = canonical.loc[canonical["fold"].astype(int).eq(fold_no), "symbol"].astype(str).tolist()
        for symbol in symbols:
            fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy().reset_index(drop=True)
            cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy().reset_index(drop=True)
            test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy().reset_index(drop=True)
            early, late = chronological_row_halves(cal)

            row = {
                "fold": fold_no,
                "symbol": symbol,
                "fit_n": int(len(fit)),
                "cal_n": int(len(cal)),
                "cal_early_n": int(len(early)),
                "cal_late_n": int(len(late)),
                "test_n": int(len(test)),
                "conditional_calibration_excess": np.nan,
                "interaction_excess": np.nan,
                "loss_cusum_excess": np.nan,
                "interaction_finite_pairs": 0,
                "fully_supported": False,
                "reason": "UNSET",
            }
            try:
                if min(len(fit), len(cal), len(early), len(late), len(test)) < 1:
                    raise ValueError("empty frozen partition")
                model, _ = fit_multinomial_v46(fit, FEATURES, R1, base_policy)
                base_cal = predict_v46(model, cal, FEATURES, R1_STATES).reset_index(drop=True)
                base_cal_matrix = probability_matrix_v47(base_cal, R1_STATES, suffix="v46")
                cal_state = encode_state_v46(cal, R1).astype(str).to_numpy()
                temp_info = fit_temperature_v47(base_cal_matrix, cal_state, cal_policy)
                temperature = float(temp_info["temperature"])

                pe = calibrated_common_probabilities(model, temperature, early)
                pl = calibrated_common_probabilities(model, temperature, late)
                pt = calibrated_common_probabilities(model, temperature, test)

                a = calibration_excess(early, late, test, pe, pl, pt)
                b, pairs = interaction_excess(early, late, test, pe, pl, pt)
                c = loss_cusum_excess(early, late, test, pe, pl, pt)
                row["conditional_calibration_excess"] = a if a is not None else np.nan
                row["interaction_excess"] = b if b is not None else np.nan
                row["loss_cusum_excess"] = c if c is not None else np.nan
                row["interaction_finite_pairs"] = int(pairs)
                row["temperature"] = temperature
                row["fully_supported"] = bool(a is not None and b is not None and c is not None)
                row["reason"] = "SUPPORTED" if row["fully_supported"] else "DIAGNOSTIC_COMPONENT_UNSUPPORTED"
            except Exception as exc:
                row["reason"] = f"RECONSTRUCTION_ERROR:{type(exc).__name__}:{exc}"
            unit_rows.append(row)

    units = pd.DataFrame(unit_rows).sort_values(["fold", "symbol"]).reset_index(drop=True)
    fully_supported = parse_bool(units["fully_supported"])
    support_n = int(fully_supported.sum())

    families = {
        "conditional_calibration": "conditional_calibration_excess",
        "interaction": "interaction_excess",
        "loss_cusum": "loss_cusum_excess",
    }
    test_rows: list[dict] = []
    for family, col in families.items():
        tmp: list[dict] = []
        for fold_no in range(1, 6):
            vals = pd.to_numeric(units.loc[units["fold"].astype(int).eq(fold_no), col], errors="coerce")
            k, n, p = one_sided_sign_p(vals)
            tmp.append({
                "family": family,
                "metric": col,
                "fold": fold_no,
                "median_excess": float(vals.median()) if vals.notna().any() else np.nan,
                "positive_units": k,
                "nonzero_units": n,
                "p_value": p,
            })
        qvals = bh_adjust([r["p_value"] for r in tmp])
        for r, q in zip(tmp, qvals):
            r["q_value"] = float(q)
            r["fold_positive"] = bool(np.isfinite(r["median_excess"]) and r["median_excess"] > 0 and q <= BH_Q_V49)
            test_rows.append(r)
    fold_tests = pd.DataFrame(test_rows)
    positive_counts = fold_tests.groupby("family")["fold_positive"].sum().astype(int).to_dict()

    # Link primary conditional-calibration drift to canonical forecast quality.
    c1_metrics = metrics.loc[metrics["arm"].astype(str).eq(C1)].copy()
    linked = units.merge(c1_metrics, on=["fold", "symbol"], how="inner", validate="one_to_one")
    linkage_rows: list[dict] = []
    for metric, harmful in [("common_multiclass_brier", "positive"), ("common_macro_ovr_auc", "negative")]:
        d = fold_cluster_bootstrap_spearman(
            linked, "conditional_calibration_excess", metric,
            reps=BOOTSTRAP_REPS_V49, seed=BOOTSTRAP_SEED_V49,
        )
        harmful_ci = bool(
            (harmful == "positive" and d["ci90_low"] is not None and d["ci90_low"] > 0)
            or (harmful == "negative" and d["ci90_high"] is not None and d["ci90_high"] < 0)
        )
        linkage_rows.append({"link": f"conditional_calibration_to_{metric}", "metric": metric, "harmful_direction": harmful, **d, "harmful_ci_excludes_zero": harmful_ci})

    executed = trades.loc[trades["arm"].astype(str).eq(C1) & parse_bool(trades["executed"])].copy()
    econ_units = executed.groupby(["fold", "symbol"], as_index=False).agg(trade_n=("net_r", "size"), unit_expectancy_r=("net_r", "mean"))
    econ_units = econ_units.loc[econ_units["trade_n"].astype(int).ge(5)].copy()
    econ_linked = units.merge(econ_units, on=["fold", "symbol"], how="inner")
    d = fold_cluster_bootstrap_spearman(
        econ_linked, "conditional_calibration_excess", "unit_expectancy_r",
        reps=BOOTSTRAP_REPS_V49, seed=BOOTSTRAP_SEED_V49,
    )
    harmful_ci = bool(d["ci90_high"] is not None and d["ci90_high"] < 0)
    linkage_rows.append({"link": "conditional_calibration_to_unit_expectancy", "metric": "unit_expectancy_r", "harmful_direction": "negative", **d, "harmful_ci_excludes_zero": harmful_ci})
    linkage = pd.DataFrame(linkage_rows)

    A = int(positive_counts.get("conditional_calibration", 0)) >= 3
    B = int(positive_counts.get("interaction", 0)) >= 3
    C = int(positive_counts.get("loss_cusum", 0)) >= 3
    L = bool(linkage["harmful_ci_excludes_zero"].any()) if not linkage.empty else False

    if support_n < 108:
        decision = "V49_INVALID_INSUFFICIENT_DIAGNOSTIC_SUPPORT"
    elif A and (B or C) and L:
        decision = "V49_CONDITIONAL_INSTABILITY_SUPPORTED"
    elif A and (B or C) and not L:
        decision = "V49_CONDITIONAL_SHIFT_PRESENT_LINK_INCONCLUSIVE"
    elif (not A) and B and C:
        decision = "V49_RESIDUAL_PROCESS_SHIFT_ONLY"
    elif (not A) and (not B) and (not C):
        decision = "V49_CONDITIONAL_INSTABILITY_NOT_SUPPORTED"
    else:
        decision = "V49_CONDITIONAL_INSTABILITY_EVIDENCE_INCONCLUSIVE"

    decision_obj = {
        "version": "v0.49",
        "experiment": "CONDITIONAL_CONCEPT_DRIFT_DIAGNOSTIC",
        "decision": decision,
        "candidate_promotion_allowed": False,
        "primary_arm": C1,
        "canonical_units": 120,
        "fully_supported_units": support_n,
        "positive_fold_counts": {k: int(v) for k, v in positive_counts.items()},
        "family_A_conditional_calibration": A,
        "family_B_interaction": B,
        "family_C_loss_cusum": C,
        "harmful_linkage": L,
        "source_v44_run": EXPECTED_V44_RUN,
        "source_v47_run": EXPECTED_V47_RUN,
        "source_v47_scientific_head": EXPECTED_V47_SCIENTIFIC_HEAD,
        "source_v47_artifact_digest": EXPECTED_V47_ARTIFACT_DIGEST,
        "source_v48_run": EXPECTED_V48_RUN,
        "source_v48_scientific_head": EXPECTED_V48_SCIENTIFIC_HEAD,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_pruning": False,
        "test_labels_used_for_model_or_calibrator_fit": False,
    }

    units.to_csv(out / "unit_conditional_drift_v49.csv", index=False)
    fold_tests.to_csv(out / "fold_conditional_tests_v49.csv", index=False)
    linkage.to_csv(out / "linkage_v49.csv", index=False)
    pd.DataFrame(context_rows(units, executed)).to_csv(out / "context_decomposition_v49.csv", index=False)
    (out / "decision_v49.json").write_text(json.dumps(decision_obj, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision_obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
