from __future__ import annotations

"""Audit-corrected wrapper for the frozen v0.39 characterization.

This file reuses the original frozen event generation, model fitting, folds,
labels, costs and thresholds from run_v39_development_characterization.py.
Only two bookkeeping defects are corrected:
1) financial simulation is independent per venue;
2) zero-selection folds count as non-positive in the five-fold denominator.
It also writes worst-group diagnostics for every evaluated candidate.
"""

import argparse
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.financial_system_v39 import (
    FinancialRiskPolicyV39,
    LearningPolicyV39,
    ValidationPolicyV39,
)
from research_bot.mother_strategy_v39 import NEURAL_FEATURES_V39


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts" / "run_v39_development_characterization.py"
spec = importlib.util.spec_from_file_location("v39_base_runner", BASE_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.39 base runner")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def _financial_simulation_independent_by_venue(realized: pd.DataFrame) -> pd.DataFrame:
    if realized.empty:
        return realized.copy()
    parts: list[pd.DataFrame] = []
    for venue in base.VENUES:
        z = realized.loc[realized["venue"] == venue].copy()
        if z.empty:
            continue
        parts.append(base._financial_simulation(z))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else realized.iloc[0:0].copy()


def _positive_fraction_all_folds(values: list[float]) -> float:
    if len(values) != base.PURGED_FOLDS:
        raise RuntimeError(f"expected exactly {base.PURGED_FOLDS} fold values; got {len(values)}")
    return float(np.mean([bool(np.isfinite(v) and v > 0.0) for v in values]))


def _worst_groups_all_candidates(candidate_trades: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for x in candidate_trades:
        if x.empty:
            continue
        candidate = str(x["candidate"].iloc[0]) if "candidate" in x.columns else "UNKNOWN"
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


def _jsonable(obj):
    return base._jsonable(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel, events, data_manifest = base._prepare_panel()
    if events.empty:
        raise RuntimeError("v0.39 generated no candidate events")
    folds = base._folds(events)
    if len(folds) != base.PURGED_FOLDS:
        raise RuntimeError("corrected v0.39 requires all five frozen folds")

    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])

    candidate_summaries: list[dict] = []
    fold_rows: list[dict] = []
    all_oos: list[pd.DataFrame] = []
    all_candidate_executed: list[pd.DataFrame] = []
    winner: str | None = None
    winner_trades = pd.DataFrame()
    winner_venue_results: dict[str, dict] = {}

    for candidate in base.CANDIDATES:
        oos_parts: list[pd.DataFrame] = []
        seed_fold_expectancies = {s: [] for s in base.SEEDS}
        fold_expectancies: list[float] = []

        for fold in folds:
            pred, seed_exps, diag = base._predict_fold(candidate, fold, feature_cols)
            oos_parts.append(pred)
            fold_rows.append(diag)
            sel = pred[pred["selected_model"]]
            fold_expectancies.append(float(sel["net_r"].mean()) if not sel.empty else float("nan"))
            for s, value in zip(base.SEEDS, seed_exps):
                seed_fold_expectancies[s].append(value)

        oos = pd.concat(oos_parts, ignore_index=True).sort_values("signal_time")
        all_oos.append(oos)
        selected = oos[oos["selected_model"]].copy()
        realized = base._realize_nonoverlap(selected)
        financially_realized = _financial_simulation_independent_by_venue(realized)
        executed = (
            financially_realized[financially_realized["executed"]].copy()
            if not financially_realized.empty
            else financially_realized
        )
        if not executed.empty:
            executed["candidate"] = candidate
            all_candidate_executed.append(executed)

        venue_results: dict[str, dict] = {}
        for venue in base.VENUES:
            vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
            metrics = base._venue_metrics(vt)
            gates = base._venue_gate(metrics)
            venue_results[venue] = {"metrics": metrics, "gates": gates}

        positive_fold_fraction = _positive_fraction_all_folds(fold_expectancies)

        # Seed stability is aggregate across whatever the seed actually selected;
        # a seed with no finite OOS selection evidence is counted as non-positive.
        seed_expectancies: list[float] = []
        for s in base.SEEDS:
            vals = [v for v in seed_fold_expectancies[s] if np.isfinite(v)]
            seed_expectancies.append(float(np.mean(vals)) if vals else float("nan"))
        positive_seed_fraction = float(np.mean([np.isfinite(v) and v > 0 for v in seed_expectancies]))

        cross_ok = (
            positive_fold_fraction >= 0.60
            and positive_seed_fraction >= (2.0 / 3.0)
        )
        venue_ok = all(venue_results[v]["gates"]["venue_pass"] for v in base.VENUES)
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

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    executed_all = pd.concat(all_candidate_executed, ignore_index=True, sort=False) if all_candidate_executed else pd.DataFrame()

    oos_all.to_csv(outdir / "oos_predictions_v39.csv.gz", index=False, compression="gzip")
    winner_trades.to_csv(outdir / "trades_v39.csv.gz", index=False, compression="gzip")
    executed_all.to_csv(outdir / "trades_all_candidates_v39.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v39.csv", index=False)
    data_manifest.to_csv(outdir / "data_manifest_v39.csv", index=False)
    _worst_groups_all_candidates(all_candidate_executed).to_csv(outdir / "worst_groups_v39.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "venue_results"} for s in candidate_summaries]).to_csv(
        outdir / "candidate_summary_v39.csv", index=False
    )

    decision = {
        "version": "v0.39",
        "implementation_revision": 2,
        "experiment": "ROBUST_MOTHER_STRATEGY_DEVELOPMENT_CHARACTERIZATION",
        "prior_audit_run_not_final": 34674711993,
        "audit_corrections": [
            "venue-independent financial accounting",
            "zero-selection folds count as non-positive",
            "worst-group diagnostics for every candidate",
        ],
        "decision": "V39_DEVELOPMENT_WINNER" if winner else "V39_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": winner,
        "candidate_order": list(base.CANDIDATES),
        "selection_rule": "first/simplest candidate passing all frozen gates",
        "venues": list(base.VENUES),
        "symbols": list(base.SYMBOLS),
        "timeframe": base.TIMEFRAME,
        "start": base.START,
        "end": base.END,
        "purged_folds": base.PURGED_FOLDS,
        "embargo_bars": base.EMBARGO_BARS,
        "base_roundtrip_bps": base.BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": base.STRESS_ROUNDTRIP_BPS,
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "learning_policy": asdict(LearningPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "candidate_summaries": candidate_summaries,
        "winner_venue_results": winner_venue_results,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_feature_selection": False,
    }
    (outdir / "decision_v39.json").write_text(
        json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
