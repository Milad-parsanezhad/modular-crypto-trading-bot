from __future__ import annotations

"""Frozen v0.42 breadth-expansion + cluster-robust characterization.

The only model carried forward is the v0.41 HistGB cause-specific competing-risk
candidate. Eligibility is determined from exchange availability and bar count
before any event outcomes are inspected. Kraken is never instantiated.
"""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.breadth_cluster_v42 import (
    BreadthClusterPolicyV42,
    V42_CANDIDATE,
    V42_DEVELOPMENT_VENUES,
    V42_RESERVED_HOLDOUT,
    V42_SOURCE_MODEL,
    V42_SYMBOL_CANDIDATES,
    hierarchical_symbol_block_bootstrap_low_v42,
    preregistration_manifest_v42,
    select_common_symbols_v42,
    v42_gate_from_metrics,
)
from research_bot.event_competing_risk_v41 import (
    CompetingRiskPolicyV41,
    V41_FEATURES,
    V41_SEEDS,
    attach_event_family_v41,
    calibrate_expected_r_bounds_v41,
    fit_competing_risk_bundle_v41,
    median_seed_prediction_v41,
)
from research_bot.event_competing_risk_vectorized_v41 import (
    predict_competing_risks_vectorized_v41,
)
from research_bot.financial_system_v39 import (
    FinancialRiskPolicyV39,
    ValidationPolicyV39,
    causal_robust_normalize,
)
from research_bot.mother_strategy_v39 import (
    NEURAL_FEATURES_V39,
    MotherStrategyPolicyV39,
    build_mother_features_v39,
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


base = _load(V39_BASE, "v39_base_for_v42")
corrected = _load(V39_CORRECTED, "v39_corrected_for_v42")

TIMEFRAME = base.TIMEFRAME
START = base.START
END = base.END
PURGED_FOLDS = base.PURGED_FOLDS
EMBARGO_BARS = base.EMBARGO_BARS
BASE_ROUNDTRIP_BPS = base.BASE_ROUNDTRIP_BPS
STRESS_ROUNDTRIP_BPS = base.STRESS_ROUNDTRIP_BPS


def _jsonable(obj):
    return base._jsonable(obj)


def _fetch_availability() -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, dict[str, int]], pd.DataFrame]:
    """Fetch candidate data without looking at event outcomes or model scores."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    bars: dict[str, dict[str, int]] = {v: {} for v in V42_DEVELOPMENT_VENUES}
    rows: list[dict] = []

    for venue in V42_DEVELOPMENT_VENUES:
        if venue == V42_RESERVED_HOLDOUT:
            raise RuntimeError("Kraken is sealed in v0.42")
        ex = base._exchange(venue)
        markets = ex.load_markets()
        for symbol in V42_SYMBOL_CANDIDATES:
            if symbol not in markets:
                bars[venue][symbol] = 0
                rows.append({
                    "venue": venue,
                    "symbol": symbol,
                    "available": False,
                    "bars": 0,
                    "first_bar": None,
                    "last_bar": None,
                    "eligibility_reason": "MISSING_MARKET",
                })
                continue
            try:
                frame = base.fetch_fixed_ohlcv(ex, symbol)
            except Exception as exc:
                bars[venue][symbol] = 0
                rows.append({
                    "venue": venue,
                    "symbol": symbol,
                    "available": False,
                    "bars": 0,
                    "first_bar": None,
                    "last_bar": None,
                    "eligibility_reason": f"FETCH_OR_HISTORY_ERROR:{type(exc).__name__}",
                })
                continue
            frames[(venue, symbol)] = frame
            bars[venue][symbol] = int(len(frame))
            rows.append({
                "venue": venue,
                "symbol": symbol,
                "available": True,
                "bars": int(len(frame)),
                "first_bar": frame["timestamp"].iloc[0],
                "last_bar": frame["timestamp"].iloc[-1],
                "eligibility_reason": "AVAILABLE",
            })
    return frames, bars, pd.DataFrame(rows)


def _prepare_expanded_events(
    frames: dict[tuple[str, str], pd.DataFrame],
    eligible_symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_panels: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    manifest: list[dict] = []
    cfg = MotherStrategyPolicyV39()

    for venue in V42_DEVELOPMENT_VENUES:
        for symbol in eligible_symbols:
            frame = frames[(venue, symbol)]
            f = build_mother_features_v39(frame, cfg).copy()
            f["venue"] = venue
            f["symbol"] = symbol
            f["series_id"] = f"{venue}::{symbol}"
            feature_panels.append(f)
            e = base._event_labels(f, venue, symbol, cfg)
            if not e.empty:
                events.append(e)
            manifest.append({
                "venue": venue,
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "requested_start": START,
                "requested_end": END,
                "first_bar": frame["timestamp"].iloc[0],
                "last_bar": frame["timestamp"].iloc[-1],
                "bars": int(len(frame)),
                "mother_events": int(f["mother_event_v39"].sum()),
                "candidate_long": int((f["research_candidate_side_v39"] == 1).sum()),
                "candidate_short": int((f["research_candidate_side_v39"] == -1).sum()),
            })

    panel = pd.concat(feature_panels, ignore_index=True, sort=False)
    event_frame = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    normalized = causal_robust_normalize(
        panel,
        NEURAL_FEATURES_V39,
        timestamp_col="timestamp",
        group_col="series_id",
    ).rename(columns={"timestamp": "signal_time"})
    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])
    event_frame = event_frame.merge(
        normalized[["signal_time", "series_id", *feature_cols]],
        on=["signal_time", "series_id"],
        how="left",
        validate="many_to_one",
    )
    event_frame = event_frame.replace([np.inf, -np.inf], np.nan)
    event_frame[feature_cols] = event_frame[feature_cols].fillna(0.0).astype("float32")
    event_frame = attach_event_family_v41(panel, event_frame)
    missing = sorted(set(V41_FEATURES) - set(event_frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen v0.41 features in v0.42: {missing}")
    event_frame.loc[:, list(V41_FEATURES)] = (
        event_frame.loc[:, list(V41_FEATURES)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    return panel, event_frame, pd.DataFrame(manifest)


def _predict_fold(fold: dict) -> tuple[pd.DataFrame, dict[int, float], list[dict], dict]:
    seed_outputs: list[pd.DataFrame] = []
    seed_expectancies: dict[int, float] = {}
    seed_diags: list[dict] = []
    policy = CompetingRiskPolicyV41()

    for seed in V41_SEEDS:
        bundle = fit_competing_risk_bundle_v41(
            V42_SOURCE_MODEL,
            fold["fit"],
            V41_FEATURES,
            seed,
            policy,
        )
        cal_pred = predict_competing_risks_vectorized_v41(bundle, fold["cal"])
        test_point = predict_competing_risks_vectorized_v41(bundle, fold["test"])
        test_pred = calibrate_expected_r_bounds_v41(
            fold["cal"], cal_pred, test_point, policy
        )
        test_pred["candidate"] = V42_CANDIDATE
        seed_outputs.append(test_pred)
        selected = test_pred[test_pred["selected_v41"]]
        seed_expectancies[int(seed)] = (
            float(selected["net_r"].mean()) if not selected.empty else float("nan")
        )
        y_target = fold["test"]["outcome"].astype(str).eq("TARGET").to_numpy(dtype=float)
        y_stop = fold["test"]["outcome"].astype(str).eq("STOP").to_numpy(dtype=float)
        seed_diags.append({
            "candidate": V42_CANDIDATE,
            "fold": int(fold["fold"]),
            "seed": int(seed),
            "fit_events": int(len(fold["fit"])),
            "calibration_events": int(len(fold["cal"])),
            "test_events": int(len(fold["test"])),
            "selected_events": int(test_pred["selected_v41"].sum()),
            "expert_fraction": float(test_pred["expert_used_v41"].mean()),
            "brier_target": float(np.mean((test_pred["p_target_v41"].to_numpy(dtype=float) - y_target) ** 2)),
            "brier_stop": float(np.mean((test_pred["p_stop_v41"].to_numpy(dtype=float) - y_stop) ** 2)),
        })

    combined = median_seed_prediction_v41(seed_outputs, policy)
    combined["candidate"] = V42_CANDIDATE
    combined["fold"] = int(fold["fold"])
    combined["expected_r"] = combined["expected_r_v41"]
    combined["lower_expected_r"] = combined["lower_expected_r_v41"]
    combined["upper_expected_r"] = combined["upper_expected_r_v41"]
    combined["selected_model"] = combined["selected_v41"]
    selected = combined[combined["selected_model"]]
    fold_diag = {
        "candidate": V42_CANDIDATE,
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


def _venue_metrics_v42(trades: pd.DataFrame, eligible_symbols: tuple[str, ...]) -> dict:
    if trades.empty:
        return {
            "n": 0,
            "profit_factor": None,
            "expectancy_r": None,
            "positive_asset_fraction": 0.0,
            "block_ci_low": None,
            "cluster_ci_low": None,
            "positive_quarter_fraction": 0.0,
            "stress_profit_factor": None,
            "max_account_drawdown": None,
        }
    x = trades.sort_values(["exit_time", "symbol"]).copy()
    asset = x.groupby("symbol")["net_r"].mean()
    breadth = sum(float(asset.get(s, -np.inf)) > 0.0 for s in eligible_symbols) / len(eligible_symbols)
    quarters = pd.to_datetime(x["exit_time"], utc=True).dt.to_period("Q")
    q = x.groupby(quarters)["net_r"].sum()
    if "equity_after_exit" in x and x["equity_after_exit"].notna().any():
        eq = x.sort_values("exit_time")["equity_after_exit"].dropna().to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
        max_dd = float(np.min(eq / peak - 1.0)) if len(eq) else None
    else:
        max_dd = None
    return {
        "n": int(len(x)),
        "profit_factor": base._profit_factor(x["net_r"]),
        "expectancy_r": float(x["net_r"].mean()),
        "positive_asset_fraction": float(breadth),
        "block_ci_low": base._block_ci_low(x["net_r"]),
        "cluster_ci_low": hierarchical_symbol_block_bootstrap_low_v42(x),
        "positive_quarter_fraction": float((q > 0).mean()) if len(q) else 0.0,
        "stress_profit_factor": base._profit_factor(x["stress_net_r"]),
        "max_account_drawdown": max_dd,
    }


def _family_breakdown(executed: pd.DataFrame) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame(columns=["event_family_v41", "side", "n", "expectancy_r", "profit_factor"])
    rows: list[dict] = []
    for (family, side), g in executed.groupby(["event_family_v41", "side"], sort=True):
        rows.append({
            "event_family_v41": str(family),
            "side": int(side),
            "n": int(len(g)),
            "expectancy_r": float(g["net_r"].mean()),
            "profit_factor": base._profit_factor(g["net_r"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = preregistration_manifest_v42()
    if manifest["kraken_touched"] is not False or manifest["reserved_holdout"] != "kraken":
        raise RuntimeError("v0.42 holdout governance invalid")

    frames, bars, availability = _fetch_availability()
    availability.to_csv(outdir / "availability_v42.csv", index=False)
    try:
        eligible_symbols = select_common_symbols_v42(bars)
    except RuntimeError as exc:
        decision = {
            "version": "v0.42",
            "experiment": manifest["experiment"],
            "decision": "V42_DATA_UNAVAILABLE",
            "reason": str(exc),
            "candidate_symbol_universe": list(V42_SYMBOL_CANDIDATES),
            "kraken_touched": False,
            "paper_execution": False,
            "live_execution": False,
        }
        (outdir / "decision_v42.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(decision, indent=2, sort_keys=True))
        return

    pd.DataFrame({"eligible_symbol": list(eligible_symbols)}).to_csv(outdir / "eligible_symbols_v42.csv", index=False)
    _, events, data_manifest = _prepare_expanded_events(frames, eligible_symbols)
    data_manifest.to_csv(outdir / "data_manifest_v42.csv", index=False)
    if events.empty:
        raise RuntimeError("v0.42 generated no mother events")

    folds = base._folds(events)
    if len(folds) != PURGED_FOLDS:
        raise RuntimeError("v0.42 requires all five frozen folds")

    oos_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    fold_expectancies: list[float] = []
    seed_fold_values: dict[int, list[float]] = {s: [] for s in V41_SEEDS}

    for fold in folds:
        pred, seed_exps, sdiag, fdiag = _predict_fold(fold)
        oos_parts.append(pred)
        seed_rows.extend(sdiag)
        fold_rows.append(fdiag)
        selected = pred[pred["selected_model"]]
        fold_expectancies.append(float(selected["net_r"].mean()) if not selected.empty else float("nan"))
        for seed, value in seed_exps.items():
            seed_fold_values[int(seed)].append(value)

    oos = pd.concat(oos_parts, ignore_index=True).sort_values(["signal_time", "venue", "symbol"])
    selected = oos[oos["selected_model"]].copy()
    realized = base._realize_nonoverlap(selected)
    financially_realized = corrected._financial_simulation_independent_by_venue(realized)
    executed = financially_realized[financially_realized["executed"]].copy() if not financially_realized.empty else financially_realized
    if not executed.empty:
        executed["candidate"] = V42_CANDIDATE

    venue_results: dict[str, dict] = {}
    for venue in V42_DEVELOPMENT_VENUES:
        vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
        metrics = _venue_metrics_v42(vt, eligible_symbols)
        gates = v42_gate_from_metrics(metrics)
        venue_results[venue] = {"metrics": metrics, "gates": gates}

    positive_fold_fraction = _positive_fraction(fold_expectancies)
    seed_expectancies: list[float] = []
    for seed in V41_SEEDS:
        vals = [v for v in seed_fold_values[int(seed)] if np.isfinite(v)]
        seed_expectancies.append(float(np.mean(vals)) if vals else float("nan"))
    positive_seed_fraction = float(np.mean([np.isfinite(v) and v > 0.0 for v in seed_expectancies]))
    cross_ok = positive_fold_fraction >= 0.60 and positive_seed_fraction >= (2.0 / 3.0)
    venue_ok = all(bool(venue_results[v]["gates"]["venue_pass"]) for v in V42_DEVELOPMENT_VENUES)
    passed = bool(cross_ok and venue_ok)

    oos.to_csv(outdir / "oos_predictions_v42.csv.gz", index=False, compression="gzip")
    executed.to_csv(outdir / "trades_v42.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v42.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(outdir / "seed_hazard_diagnostics_v42.csv", index=False)
    _family_breakdown(executed).to_csv(outdir / "event_family_results_v42.csv", index=False)
    base._worst_groups(executed).to_csv(outdir / "worst_groups_v42.csv", index=False)

    summary = {
        "candidate": V42_CANDIDATE,
        "eligible_symbol_count": int(len(eligible_symbols)),
        "eligible_symbols": list(eligible_symbols),
        "oos_events": int(len(oos)),
        "model_selected_events": int(len(selected)),
        "nonoverlap_events": int(len(realized)),
        "financially_executed_events": int(len(executed)),
        "positive_fold_fraction": positive_fold_fraction,
        "seed_expectancies_r": seed_expectancies,
        "positive_seed_fraction": positive_seed_fraction,
        "all_venue_gates_pass": venue_ok,
        "cross_fold_seed_gate_pass": cross_ok,
        "candidate_pass": passed,
        "venue_results": venue_results,
    }
    pd.DataFrame([{k: v for k, v in summary.items() if k not in {"venue_results", "eligible_symbols"}}]).to_csv(
        outdir / "candidate_summary_v42.csv", index=False
    )

    decision = {
        "version": "v0.42",
        "experiment": "BREADTH_EXPANSION_CLUSTER_ROBUST_VALIDATION",
        "decision": "V42_DEVELOPMENT_WINNER" if passed else "V42_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": V42_CANDIDATE if passed else None,
        "source_model": V42_SOURCE_MODEL,
        "eligible_symbols": list(eligible_symbols),
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "purged_folds": PURGED_FOLDS,
        "embargo_bars": EMBARGO_BARS,
        "base_roundtrip_bps": BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": STRESS_ROUNDTRIP_BPS,
        "breadth_cluster_policy": asdict(BreadthClusterPolicyV42()),
        "competing_risk_policy": asdict(CompetingRiskPolicyV41()),
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "candidate_summary": summary,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_symbol_pruning_by_performance": False,
    }
    (outdir / "decision_v42.json").write_text(
        json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
