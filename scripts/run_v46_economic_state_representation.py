from __future__ import annotations

"""Execute frozen v0.46 economic-state representation ablation on consumed development data."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import DataQualityPolicyV43, V43_DEVELOPMENT_VENUES
from research_bot.economic_state_v46 import (
    R0,
    R1,
    V46_ARMS,
    V46ModelPolicy,
    encode_state_v46,
    expected_r_from_state_probs_v46,
    fit_multinomial_v46,
    forecast_metrics_v46,
    predict_v46,
    reliability_resolution_v46,
    states_for_arm_v46,
)
from research_bot.event_competing_risk_v41 import V41_FEATURES

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


base = _load(V39_BASE, "v39_base_for_v46")
corrected = _load(V39_CORRECTED, "v39_corrected_for_v46")

EXPECTED_SOURCE_RUN = 34700944062
EXPECTED_SOURCE_HEAD = "5efb843d385d194c549a69dc4b2fd38d56ab906c"
EXPECTED_EVENTS_SHA256 = "c1_placeholder_replaced_by_runtime_manifest"
ARMS = V46_ARMS
FEATURES = tuple(V41_FEATURES) + tuple(f"venue_{v}_v43" for v in V43_DEVELOPMENT_VENUES)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _profit_factor(x: pd.Series) -> float | None:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    pos = float(arr[arr > 0].sum())
    neg = float(-arr[arr < 0].sum())
    if neg <= 0:
        return None if pos <= 0 else float("inf")
    return pos / neg


def _max_drawdown_from_net_r(x: pd.Series, risk_fraction: float = 0.0025) -> float | None:
    r = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return None
    equity = np.cumprod(1.0 + risk_fraction * r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _economic_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n": 0,
            "expectancy_r": None,
            "profit_factor": None,
            "stress_profit_factor": None,
            "max_drawdown": None,
        }
    return {
        "n": int(len(trades)),
        "expectancy_r": float(pd.to_numeric(trades["net_r"], errors="coerce").mean()),
        "profit_factor": _profit_factor(trades["net_r"]),
        "stress_profit_factor": _profit_factor(trades["stress_net_r"]),
        "max_drawdown": _max_drawdown_from_net_r(trades["net_r"]),
    }


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
    event_path = indir / "events_v44.pkl.gz"
    if _sha256(event_path) != hashes.get("events_v44_pickle_sha256"):
        raise RuntimeError("source v0.44 event table hash mismatch")

    events = pd.read_pickle(event_path, compression="gzip")
    folds = pd.read_csv(
        indir / "fold_index_v44.csv",
        parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"],
    )
    if len(folds) != 5 or sorted(folds["fold"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("v0.46 requires the exact five frozen v0.44 common folds")
    missing = sorted(set(FEATURES) - set(events.columns))
    if missing:
        raise RuntimeError(f"missing frozen v0.46 features: {missing}")

    policy = V46ModelPolicy()
    support_rows: list[dict] = []
    metric_rows: list[dict] = []
    reliability_rows: list[dict] = []
    state_value_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for arm in ARMS:
        states = states_for_arm_v46(arm)
        for fold_no in range(1, 6):
            row = folds.loc[folds["fold"].astype(int).eq(fold_no)].iloc[0]
            fit_all, cal_all, test_all = _window_slice(events, row)
            for symbol in sorted(test_all["symbol"].astype(str).unique()):
                fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy()
                cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy()
                test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy()
                try:
                    fit_state = encode_state_v46(fit, arm)
                    cal_state = encode_state_v46(cal, arm)
                    test_state = encode_state_v46(test, arm)
                    state_counts = fit_state.value_counts().to_dict()
                    supported = (
                        len(fit) >= policy.minimum_fit_events
                        and len(cal) >= policy.minimum_calibration_events
                        and len(test) >= policy.minimum_test_events
                        and set(state_counts) == set(states)
                    )
                except Exception:
                    supported = False
                    state_counts = {}
                support_rows.append({
                    "arm": arm,
                    "fold": fold_no,
                    "symbol": symbol,
                    "fit_events": int(len(fit)),
                    "calibration_events": int(len(cal)),
                    "test_events": int(len(test)),
                    "supported": bool(supported),
                    "fit_state_counts": json.dumps({str(k): int(v) for k, v in state_counts.items()}, sort_keys=True),
                })
                if not supported:
                    continue

                model, state_means = fit_multinomial_v46(fit, FEATURES, arm, policy)
                p = predict_v46(model, test, FEATURES, states)
                expected = expected_r_from_state_probs_v46(p, state_means, states)
                pred = test.reset_index(drop=True).copy()
                pp = p.reset_index(drop=True)
                for c in pp.columns:
                    pred[c] = pp[c]
                pred["arm"] = arm
                pred["fold"] = fold_no
                pred["state_v46"] = test_state.reset_index(drop=True)
                pred["expected_r_v46"] = expected.reset_index(drop=True)
                pred["selected_model"] = np.isfinite(pred["expected_r_v46"]) & pred["expected_r_v46"].gt(0.0)
                pred["expected_r"] = pred["expected_r_v46"]
                pred["lower_expected_r"] = pred["expected_r_v46"]
                pred["upper_expected_r"] = pred["expected_r_v46"]
                predictions.append(pred)

                metrics = forecast_metrics_v46(test_state.reset_index(drop=True), pp, states)
                metric_rows.append({"arm": arm, "fold": fold_no, "symbol": symbol, **metrics})
                for state in states:
                    y = test_state.astype(str).eq(state).to_numpy(dtype=float)
                    ps = pp[f"p_{state.lower()}_v46"].to_numpy(dtype=float)
                    rel, res, bins = reliability_resolution_v46(y, ps, bins=10)
                    for _, br in bins.iterrows():
                        reliability_rows.append({
                            "arm": arm, "fold": fold_no, "symbol": symbol, "state": state,
                            "reliability": rel, "resolution": res,
                            "bin": int(br["bin"]), "n": int(br["n"]),
                            "mean_p": br["mean_p"], "mean_y": br["mean_y"],
                        })
                for state, mean_r in state_means.items():
                    state_value_rows.append({
                        "arm": arm, "fold": fold_no, "symbol": symbol, "state": state,
                        "train_mean_net_r": float(mean_r), "source": "FIT_ONLY",
                    })

    pred_all = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    support_df = pd.DataFrame(support_rows)
    metrics_df = pd.DataFrame(metric_rows)
    reliability_df = pd.DataFrame(reliability_rows)
    state_values_df = pd.DataFrame(state_value_rows)

    support_df.to_csv(outdir / "support_manifest_v46.csv", index=False)
    metrics_df.to_csv(outdir / "forecast_metrics_v46.csv", index=False)
    reliability_df.to_csv(outdir / "reliability_bins_v46.csv", index=False)
    state_values_df.to_csv(outdir / "training_state_values_v46.csv", index=False)
    pred_all.to_csv(outdir / "oos_predictions_v46.csv.gz", index=False, compression="gzip")

    trade_parts: list[pd.DataFrame] = []
    fold_econ_rows: list[dict] = []
    for arm in ARMS:
        arm_pred = pred_all.loc[pred_all["arm"].astype(str).eq(arm)].copy() if not pred_all.empty else pred_all
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
            fold_econ_rows.append({"arm": arm, "fold": fold_no, **_economic_summary(executed)})

    trades = pd.concat(trade_parts, ignore_index=True, sort=False) if trade_parts else pd.DataFrame()
    folds_econ = pd.DataFrame(fold_econ_rows)
    trades.to_csv(outdir / "post_governor_trades_v46.csv.gz", index=False, compression="gzip")
    folds_econ.to_csv(outdir / "fold_economics_v46.csv", index=False)

    aggregate: dict[str, dict] = {}
    for arm in ARMS:
        arm_metrics = metrics_df.loc[metrics_df["arm"].astype(str).eq(arm)]
        arm_trades = trades.loc[trades["arm"].astype(str).eq(arm)] if not trades.empty else trades
        econ = _economic_summary(arm_trades)
        fold_arm = folds_econ.loc[folds_econ["arm"].astype(str).eq(arm)]
        positive_fold_fraction = float(
            np.mean(pd.to_numeric(fold_arm["expectancy_r"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float) > 0.0)
        ) if len(fold_arm) else 0.0
        aggregate[arm] = {
            "supported_asset_folds": int(len(arm_metrics)),
            "median_multiclass_brier": float(arm_metrics["multiclass_brier"].median()) if len(arm_metrics) else None,
            "median_mean_reliability": float(arm_metrics["mean_reliability"].median()) if len(arm_metrics) else None,
            "positive_fold_fraction": positive_fold_fraction,
            **econ,
        }

    r0 = aggregate[R0]
    r1 = aggregate[R1]
    gates = {
        "brier_better_than_r0": bool(r1["median_multiclass_brier"] is not None and r0["median_multiclass_brier"] is not None and r1["median_multiclass_brier"] < r0["median_multiclass_brier"]),
        "reliability_better_than_r0": bool(r1["median_mean_reliability"] is not None and r0["median_mean_reliability"] is not None and r1["median_mean_reliability"] < r0["median_mean_reliability"]),
        "positive_fold_fraction_ge_0_60": bool(r1["positive_fold_fraction"] >= 0.60),
        "aggregate_expectancy_gt_0": bool(r1["expectancy_r"] is not None and r1["expectancy_r"] > 0.0),
        "aggregate_pf_ge_1_05": bool(r1["profit_factor"] is not None and r1["profit_factor"] >= 1.05),
        "max_drawdown_le_0_05": bool(r1["max_drawdown"] is not None and r1["max_drawdown"] >= -0.05),
        "stress_pf_ge_1_00": bool(r1["stress_profit_factor"] is not None and r1["stress_profit_factor"] >= 1.00),
        "kraken_sealed": True,
    }
    passed = all(gates.values())
    decision = {
        "version": "v0.46",
        "experiment": "ECONOMIC_STATE_REPRESENTATION_REDESIGN",
        "decision": "V46_DEVELOPMENT_CANDIDATE" if passed else "V46_REPRESENTATION_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": R1 if passed else None,
        "arms": aggregate,
        "gates": gates,
        "source_v44_run": EXPECTED_SOURCE_RUN,
        "source_v44_head": EXPECTED_SOURCE_HEAD,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_tuning": False,
        "post_result_asset_pruning": False,
    }
    (outdir / "decision_v46.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    output_hashes = {}
    for path in sorted(outdir.iterdir()):
        if path.is_file() and path.name != "output_hashes_v46.json":
            output_hashes[path.name] = _sha256(path)
    (outdir / "output_hashes_v46.json").write_text(json.dumps(output_hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
