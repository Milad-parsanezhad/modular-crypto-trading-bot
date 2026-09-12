from __future__ import annotations

"""Execute the frozen v0.47 post-hoc probability-calibration ablation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import V43_DEVELOPMENT_VENUES
from research_bot.economic_state_v46 import (
    R1,
    R1_STATES,
    V46ModelPolicy,
    allocate_portfolio_risk_writable_v46,
    encode_state_v46,
    fit_multinomial_v46,
    predict_v46,
)
from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.probability_calibration_v47 import (
    C0,
    C1,
    C2,
    COMMON_STATES,
    V47_ARMS,
    V47CalibrationPolicy,
    calibrate_v47,
    common_three_state_probabilities_v47,
    expected_r_v47,
    forecast_metrics_v47,
    probability_frame_v47,
    probability_matrix_v47,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_RUN = 34700944062
EXPECTED_SOURCE_HEAD = "5efb843d385d194c549a69dc4b2fd38d56ab906c"
EXPECTED_V46_RUN = 34704267477
FEATURES = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)

V46_R1_CANONICAL = {
    "supported_asset_folds": 120,
    "median_common_multiclass_brier": 0.6165395472330891,
    "median_common_mean_reliability": 0.022109087746161108,
    "positive_fold_fraction": 0.4,
    "n": 1458,
    "expectancy_r": 0.013414068988907483,
    "profit_factor": 1.0233601003945918,
    "stress_profit_factor": 0.9659025413599204,
    "max_drawdown": -0.053680363381163665,
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load(ROOT / "scripts" / "run_v39_development_characterization.py", "v39_base_for_v47")
corrected = _load(ROOT / "scripts" / "run_v39_development_characterization_corrected.py", "v39_corrected_for_v47")
base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46
corrected.base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _window_slice(events: pd.DataFrame, row: pd.Series):
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


def _profit_factor(values: pd.Series) -> float | None:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return None
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else None
    return gains / losses


def _worst_governor_drawdown(trades: pd.DataFrame) -> float | None:
    if trades.empty or "equity_after_exit" not in trades:
        return None
    groups = trades.groupby(["fold", "venue"], sort=False) if {"fold", "venue"}.issubset(trades.columns) else [("all", trades)]
    worst: float | None = None
    for _, g in groups:
        eq = pd.to_numeric(g.sort_values("exit_time")["equity_after_exit"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(eq) == 0:
            continue
        peak = np.maximum.accumulate(eq)
        dd = float(np.min(eq / peak - 1.0))
        worst = dd if worst is None else min(worst, dd)
    return worst


def _economic_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "expectancy_r": None, "profit_factor": None, "stress_profit_factor": None, "max_drawdown": None}
    return {
        "n": int(len(trades)),
        "expectancy_r": float(pd.to_numeric(trades["net_r"], errors="coerce").mean()),
        "profit_factor": _profit_factor(trades["net_r"]),
        "stress_profit_factor": _profit_factor(trades["stress_net_r"]),
        "max_drawdown": _worst_governor_drawdown(trades),
    }


def _assert_c0_reproduces_v46(aggregate: dict) -> None:
    for key, expected in V46_R1_CANONICAL.items():
        got = aggregate.get(key)
        if isinstance(expected, int):
            if int(got) != expected:
                raise RuntimeError(f"C0 does not reproduce v0.46 {key}: {got} != {expected}")
        else:
            if got is None or not np.isclose(float(got), float(expected), rtol=1e-9, atol=1e-12):
                raise RuntimeError(f"C0 does not reproduce v0.46 {key}: {got} != {expected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep = json.loads((indir / "prep_status_v44.json").read_text(encoding="utf-8"))
    hashes = json.loads((indir / "prepared_hashes_v44.json").read_text(encoding="utf-8"))
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("source v0.44 prepared evidence is not READY/fail-closed")
    events_path = indir / "events_v44.pkl.gz"
    if _sha256(events_path) != hashes.get("events_v44_pickle_sha256"):
        raise RuntimeError("source v0.44 event table hash mismatch")

    events = pd.read_pickle(events_path, compression="gzip")
    folds = pd.read_csv(indir / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    if len(folds) != 5 or sorted(folds["fold"].astype(int)) != [1, 2, 3, 4, 5]:
        raise RuntimeError("v0.47 requires exact five frozen common folds")
    missing = sorted(set(FEATURES) - set(events.columns))
    if missing:
        raise RuntimeError(f"missing frozen features: {missing}")

    base_policy = V46ModelPolicy()
    cal_policy = V47CalibrationPolicy()
    support_rows: list[dict] = []
    metric_rows: list[dict] = []
    calibration_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []

    for fold_no in range(1, 6):
        row = folds.loc[folds["fold"].astype(int).eq(fold_no)].iloc[0]
        fit_all, cal_all, test_all = _window_slice(events, row)
        for symbol in sorted(test_all["symbol"].astype(str).unique()):
            fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy()
            cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy()
            test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy()
            fit_state = encode_state_v46(fit, R1) if not fit.empty else pd.Series(dtype=str)
            cal_state = encode_state_v46(cal, R1) if not cal.empty else pd.Series(dtype=str)
            fit_counts = fit_state.value_counts().to_dict()
            cal_counts = cal_state.value_counts().to_dict()
            base_supported = bool(
                len(fit) >= cal_policy.minimum_fit_events
                and len(cal) >= cal_policy.minimum_calibration_events
                and len(test) >= cal_policy.minimum_test_events
                and set(fit_counts) == set(R1_STATES)
            )

            if not base_supported:
                for arm in V47_ARMS:
                    support_rows.append({
                        "arm": arm, "fold": fold_no, "symbol": symbol,
                        "fit_events": int(len(fit)), "calibration_events": int(len(cal)), "test_events": int(len(test)),
                        "supported": False,
                        "fit_state_counts": json.dumps({str(k): int(v) for k, v in fit_counts.items()}, sort_keys=True),
                        "cal_state_counts": json.dumps({str(k): int(v) for k, v in cal_counts.items()}, sort_keys=True),
                        "reason": "BASE_R1_UNSUPPORTED",
                    })
                continue

            model, state_means = fit_multinomial_v46(fit, FEATURES, R1, base_policy)
            base_cal = predict_v46(model, cal, FEATURES, R1_STATES).reset_index(drop=True)
            base_test = predict_v46(model, test, FEATURES, R1_STATES).reset_index(drop=True)
            cal_matrix = probability_matrix_v47(base_cal, R1_STATES, suffix="v46")
            test_matrix = probability_matrix_v47(base_test, R1_STATES, suffix="v46")
            test_native_y = encode_state_v46(test, R1).reset_index(drop=True)
            common_y = test["outcome"].astype(str).reset_index(drop=True)
            unique_cal = set(cal_state.astype(str).unique())

            arm_support = {
                C0: True,
                C1: len(unique_cal) >= 2,
                C2: unique_cal == set(R1_STATES),
            }

            for arm in V47_ARMS:
                supported = bool(arm_support[arm])
                support_rows.append({
                    "arm": arm, "fold": fold_no, "symbol": symbol,
                    "fit_events": int(len(fit)), "calibration_events": int(len(cal)), "test_events": int(len(test)),
                    "supported": supported,
                    "fit_state_counts": json.dumps({str(k): int(v) for k, v in fit_counts.items()}, sort_keys=True),
                    "cal_state_counts": json.dumps({str(k): int(v) for k, v in cal_counts.items()}, sort_keys=True),
                    "reason": "SUPPORTED" if supported else "CALIBRATION_STATE_SUPPORT_INSUFFICIENT",
                })
                if not supported:
                    continue

                calibrated_matrix, diagnostics = calibrate_v47(arm, cal_matrix, cal_state.astype(str).to_numpy(), test_matrix, cal_policy)
                native_p = probability_frame_v47(calibrated_matrix, R1_STATES).reset_index(drop=True)
                common_p = common_three_state_probabilities_v47(native_p).reset_index(drop=True)
                native_metrics = forecast_metrics_v47(test_native_y.astype(str).to_numpy(), native_p, R1_STATES)
                common_metrics = forecast_metrics_v47(common_y.astype(str).to_numpy(), common_p, COMMON_STATES)
                metric_rows.append({
                    "arm": arm, "fold": fold_no, "symbol": symbol,
                    "native_multiclass_brier": native_metrics["multiclass_brier"],
                    "native_mean_reliability": native_metrics["mean_reliability"],
                    "native_log_loss": native_metrics["log_loss"],
                    "native_macro_ovr_auc": native_metrics["macro_ovr_auc"],
                    "common_multiclass_brier": common_metrics["multiclass_brier"],
                    "common_mean_reliability": common_metrics["mean_reliability"],
                    "common_log_loss": common_metrics["log_loss"],
                    "common_macro_ovr_auc": common_metrics["macro_ovr_auc"],
                })
                calibration_rows.append({
                    "arm": arm, "fold": fold_no, "symbol": symbol,
                    **{str(k): v for k, v in diagnostics.items()},
                })

                expected = expected_r_v47(native_p, state_means).reset_index(drop=True)
                pred = test.reset_index(drop=True).copy()
                for c in native_p.columns:
                    pred[c] = native_p[c]
                for c in common_p.columns:
                    pred[f"common_{c}"] = common_p[c]
                pred["arm"] = arm
                pred["fold"] = fold_no
                pred["state_v47"] = test_native_y
                pred["expected_r_v47"] = expected
                pred["selected_model"] = np.isfinite(expected) & expected.gt(0.0)
                pred["expected_r"] = expected
                pred["lower_expected_r"] = expected
                pred["upper_expected_r"] = expected
                prediction_parts.append(pred)

    support = pd.DataFrame(support_rows)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False) if prediction_parts else pd.DataFrame()
    support.to_csv(outdir / "support_manifest_v47.csv", index=False)
    metrics.to_csv(outdir / "forecast_metrics_v47.csv", index=False)
    calibration.to_csv(outdir / "calibration_parameters_v47.csv", index=False)
    predictions.to_csv(outdir / "oos_predictions_v47.csv.gz", index=False, compression="gzip")

    trade_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for arm in V47_ARMS:
        arm_pred = predictions.loc[predictions["arm"].astype(str).eq(arm)].copy() if not predictions.empty else predictions
        for fold_no in range(1, 6):
            fold_pred = arm_pred.loc[arm_pred["fold"].astype(int).eq(fold_no)].copy() if not arm_pred.empty else arm_pred
            selected = fold_pred.loc[fold_pred["selected_model"].astype(bool)].copy() if not fold_pred.empty else fold_pred
            realized = base._realize_nonoverlap(selected) if not selected.empty else selected
            governed = corrected._financial_simulation_independent_by_venue(realized) if not realized.empty else realized
            executed = governed.loc[governed["executed"].astype(bool)].copy() if not governed.empty and "executed" in governed else governed
            if not executed.empty:
                executed["arm"] = arm
                executed["fold"] = fold_no
                trade_parts.append(executed)
            fold_rows.append({"arm": arm, "fold": fold_no, **_economic_summary(executed)})

    trades = pd.concat(trade_parts, ignore_index=True, sort=False) if trade_parts else pd.DataFrame()
    fold_econ = pd.DataFrame(fold_rows)
    trades.to_csv(outdir / "post_governor_trades_v47.csv.gz", index=False, compression="gzip")
    fold_econ.to_csv(outdir / "fold_economics_v47.csv", index=False)

    aggregate: dict[str, dict] = {}
    for arm in V47_ARMS:
        m = metrics.loc[metrics["arm"].astype(str).eq(arm)] if not metrics.empty else metrics
        t = trades.loc[trades["arm"].astype(str).eq(arm)] if not trades.empty else trades
        f = fold_econ.loc[fold_econ["arm"].astype(str).eq(arm)]
        exps = pd.to_numeric(f["expectancy_r"], errors="coerce")
        aggregate[arm] = {
            "supported_asset_folds": int(len(m)),
            "median_native_multiclass_brier": float(m["native_multiclass_brier"].median()) if len(m) else None,
            "median_common_multiclass_brier": float(m["common_multiclass_brier"].median()) if len(m) else None,
            "median_common_mean_reliability": float(m["common_mean_reliability"].median()) if len(m) else None,
            "median_common_macro_ovr_auc": float(m["common_macro_ovr_auc"].dropna().median()) if len(m) and m["common_macro_ovr_auc"].notna().any() else None,
            "positive_fold_fraction": float(np.mean(exps.fillna(-np.inf).to_numpy(dtype=float) > 0.0)) if len(f) else 0.0,
            **_economic_summary(t),
        }

    _assert_c0_reproduces_v46(aggregate[C0])
    c0 = aggregate[C0]
    gates: dict[str, dict] = {}
    winner: str | None = None
    for arm in (C1, C2):
        a = aggregate[arm]
        arm_gates = {
            "support_equals_c0": a["supported_asset_folds"] == c0["supported_asset_folds"],
            "common_brier_better_than_c0": a["median_common_multiclass_brier"] is not None and a["median_common_multiclass_brier"] < c0["median_common_multiclass_brier"],
            "common_reliability_better_than_c0": a["median_common_mean_reliability"] is not None and a["median_common_mean_reliability"] < c0["median_common_mean_reliability"],
            "common_auc_noninferior": a["median_common_macro_ovr_auc"] is not None and c0["median_common_macro_ovr_auc"] is not None and a["median_common_macro_ovr_auc"] >= c0["median_common_macro_ovr_auc"] - 0.01,
            "positive_fold_fraction_ge_0_60": a["positive_fold_fraction"] >= 0.60,
            "aggregate_expectancy_gt_0": a["expectancy_r"] is not None and a["expectancy_r"] > 0.0,
            "aggregate_pf_ge_1_05": a["profit_factor"] is not None and a["profit_factor"] >= 1.05,
            "stress_pf_ge_1_00": a["stress_profit_factor"] is not None and a["stress_profit_factor"] >= 1.00,
            "max_drawdown_le_0_05": a["max_drawdown"] is not None and a["max_drawdown"] >= -0.05,
            "kraken_sealed": True,
            "no_post_result_pruning_or_threshold_tuning": True,
        }
        arm_gates["arm_pass"] = bool(all(v for k, v in arm_gates.items() if k != "arm_pass"))
        gates[arm] = arm_gates
        if winner is None and arm_gates["arm_pass"]:
            winner = arm

    decision = {
        "version": "v0.47",
        "experiment": "POSTHOC_PROBABILITY_CALIBRATION_ABLATION",
        "decision": "V47_DEVELOPMENT_CANDIDATE" if winner else "V47_CALIBRATION_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": winner,
        "selection_order": [C1, C2],
        "arms": aggregate,
        "gates": gates,
        "control_reproduces_v46": True,
        "source_v44_run": EXPECTED_SOURCE_RUN,
        "source_v44_head": EXPECTED_SOURCE_HEAD,
        "source_v46_run": EXPECTED_V46_RUN,
        "cross_arm_metric_space": "TARGET_STOP_TIME_COMMON_SPACE",
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_tuning": False,
        "post_result_asset_pruning": False,
        "post_result_regime_pruning": False,
    }
    (outdir / "decision_v47.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    output_hashes = {}
    for path in sorted(outdir.iterdir()):
        if path.is_file() and path.name != "output_hashes_v47.json":
            output_hashes[path.name] = _sha256(path)
    (outdir / "output_hashes_v47.json").write_text(json.dumps(output_hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
