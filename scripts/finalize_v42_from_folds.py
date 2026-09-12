from __future__ import annotations

"""Finalize frozen v0.42 evidence from independently computed fold artifacts.

This reproduces the post-fold section of the monolithic v0.42 runner: OOS concat,
non-overlap realization, venue-independent financial simulation, cluster-robust
metrics, frozen gates, and the final decision. No scientific rule is changed.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "scripts" / "run_v42_breadth_cluster_characterization_fast.py"
spec = importlib.util.spec_from_file_location("v42_fast_finalize", FAST)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load v0.42 fast runner")
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
r = fast.runner


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep = json.loads((indir / "prep_status_v42.json").read_text(encoding="utf-8"))
    for name in ("availability_v42.csv", "eligible_symbols_v42.csv", "data_manifest_v42.csv", "fold_index_v42.json", "prep_status_v42.json"):
        _copy_if_exists(indir / name, outdir / name)

    if prep.get("status") != "READY":
        decision = {
            "version": "v0.42",
            "experiment": r.preregistration_manifest_v42()["experiment"],
            "decision": "V42_DATA_UNAVAILABLE",
            "reason": prep.get("reason", "prepared dataset unavailable"),
            "implementation_route": "PARALLEL_FOLD_CHECKPOINTED",
            "candidate_symbol_universe": list(r.V42_SYMBOL_CANDIDATES),
            "kraken_touched": False,
            "paper_execution": False,
            "live_execution": False,
        }
        (outdir / "decision_v42.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return

    eligible = pd.read_csv(indir / "eligible_symbols_v42.csv")
    eligible_symbols = tuple(eligible["eligible_symbol"].astype(str).tolist())
    if len(eligible_symbols) < r.BreadthClusterPolicyV42().minimum_common_symbols:
        raise RuntimeError("prepared eligible symbol count violates frozen v0.42 policy")

    oos_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    seed_rows: list[pd.DataFrame] = []
    fold_expectancies: list[float] = []
    seed_fold_values: dict[int, list[float]] = {int(s): [] for s in r.V41_SEEDS}

    for fold_num in range(1, r.PURGED_FOLDS + 1):
        pred_path = indir / f"oos_fold_{fold_num}_v42.pkl.gz"
        if not pred_path.exists():
            raise RuntimeError(f"missing parallel fold prediction: {pred_path}")
        pred = pd.read_pickle(pred_path, compression="gzip")
        oos_parts.append(pred)

        fold_diag = json.loads((indir / f"fold_diag_{fold_num}_v42.json").read_text(encoding="utf-8"))
        fold_rows.append(fold_diag)
        selected = pred[pred["selected_model"]]
        fold_expectancies.append(
            float(selected["net_r"].mean()) if not selected.empty else float("nan")
        )

        seed_exp = json.loads((indir / f"seed_expectancies_fold_{fold_num}_v42.json").read_text(encoding="utf-8"))
        for seed in r.V41_SEEDS:
            value = seed_exp.get(str(int(seed)))
            seed_fold_values[int(seed)].append(float(value) if value is not None else float("nan"))

        sdiag = pd.read_csv(indir / f"seed_diags_fold_{fold_num}_v42.csv")
        seed_rows.append(sdiag)

    if len(oos_parts) != r.PURGED_FOLDS:
        raise RuntimeError("v0.42 parallel finalizer requires all five frozen folds")

    oos = pd.concat(oos_parts, ignore_index=True).sort_values(
        ["signal_time", "venue", "symbol"]
    )
    selected = oos[oos["selected_model"]].copy()
    realized = r.base._realize_nonoverlap(selected)
    financially_realized = r.corrected._financial_simulation_independent_by_venue(realized)
    executed = (
        financially_realized[financially_realized["executed"]].copy()
        if not financially_realized.empty
        else financially_realized
    )
    if not executed.empty:
        executed["candidate"] = r.V42_CANDIDATE

    venue_results: dict[str, dict] = {}
    for venue in r.V42_DEVELOPMENT_VENUES:
        vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
        metrics = r._venue_metrics_v42(vt, eligible_symbols)
        gates = r.v42_gate_from_metrics(metrics)
        venue_results[venue] = {"metrics": metrics, "gates": gates}

    positive_fold_fraction = r._positive_fraction(fold_expectancies)
    seed_expectancies: list[float] = []
    for seed in r.V41_SEEDS:
        vals = [v for v in seed_fold_values[int(seed)] if np.isfinite(v)]
        seed_expectancies.append(float(np.mean(vals)) if vals else float("nan"))
    positive_seed_fraction = float(
        np.mean([np.isfinite(v) and v > 0.0 for v in seed_expectancies])
    )
    cross_ok = (
        positive_fold_fraction >= 0.60
        and positive_seed_fraction >= (2.0 / 3.0)
    )
    venue_ok = all(
        bool(venue_results[v]["gates"]["venue_pass"])
        for v in r.V42_DEVELOPMENT_VENUES
    )
    passed = bool(cross_ok and venue_ok)

    oos.to_csv(outdir / "oos_predictions_v42.csv.gz", index=False, compression="gzip")
    executed.to_csv(outdir / "trades_v42.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v42.csv", index=False)
    pd.concat(seed_rows, ignore_index=True).to_csv(
        outdir / "seed_hazard_diagnostics_v42.csv", index=False
    )
    r._family_breakdown(executed).to_csv(
        outdir / "event_family_results_v42.csv", index=False
    )
    r.base._worst_groups(executed).to_csv(outdir / "worst_groups_v42.csv", index=False)

    summary = {
        "candidate": r.V42_CANDIDATE,
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
    pd.DataFrame([
        {k: v for k, v in summary.items() if k not in {"venue_results", "eligible_symbols"}}
    ]).to_csv(outdir / "candidate_summary_v42.csv", index=False)

    decision = {
        "version": "v0.42",
        "experiment": "BREADTH_EXPANSION_CLUSTER_ROBUST_VALIDATION",
        "implementation_route": "PARALLEL_FOLD_CHECKPOINTED_EQUIVALENT",
        "decision": (
            "V42_DEVELOPMENT_WINNER"
            if passed
            else "V42_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE"
        ),
        "winner": r.V42_CANDIDATE if passed else None,
        "source_model": r.V42_SOURCE_MODEL,
        "eligible_symbols": list(eligible_symbols),
        "timeframe": r.TIMEFRAME,
        "start": r.START,
        "end": r.END,
        "purged_folds": r.PURGED_FOLDS,
        "embargo_bars": r.EMBARGO_BARS,
        "base_roundtrip_bps": r.BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": r.STRESS_ROUNDTRIP_BPS,
        "breadth_cluster_policy": r.asdict(r.BreadthClusterPolicyV42()),
        "competing_risk_policy": r.asdict(r.CompetingRiskPolicyV41()),
        "financial_risk_policy": r.asdict(r.FinancialRiskPolicyV39()),
        "validation_policy": r.asdict(r.ValidationPolicyV39()),
        "candidate_summary": summary,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_symbol_pruning_by_performance": False,
    }
    (outdir / "decision_v42.json").write_text(
        json.dumps(r._jsonable(decision), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(r._jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
