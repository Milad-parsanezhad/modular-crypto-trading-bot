from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from research_bot.ccxt_external_v24c import fetch_ccxt_spot_ohlcv
from research_bot.frozen_snapshot_v24d import (
    FrozenSnapshotContract,
    assert_runtime_compatible,
    load_exact_frozen_bundles,
    replay_archived_validation,
    verify_snapshot_files,
)
from research_bot.ml_framework_v23r import dataframe_sha256, probability_score
from research_bot.portfolio_mtm_v24c import MTMRiskContract, simulate_mtm_portfolio
from research_bot.strategy_meta_v24 import (
    V24Contract,
    breadth_uplift,
    build_strategy_event_panel,
    cost_stress_table,
    paired_block_uplift_ci,
    target_specs,
)
from research_bot.v24c_external_plan import (
    EXTERNAL_BYBIT_SYMBOLS_V24C,
    FROZEN_CANDIDATES_V24C,
    INTERNAL_LAST_CLOSED_BAR_UTC,
)


PERIOD = "4h"
START = pd.Timestamp("2025-01-19T12:00:00Z")
END = INTERNAL_LAST_CLOSED_BAR_UTC
VENUES = ("okx", "kucoin")
EXTERNAL_SYMBOLS = EXTERNAL_BYBIT_SYMBOLS_V24C
MIN_BARS = 1200
MAX_MISSING_FRACTION = 0.02
MIN_USABLE_SYMBOLS = 5


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _gap_audit(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "rows": 0, "gap_count": 0, "estimated_missing_bars": 0,
            "missing_fraction": 1.0, "first": None, "last": None,
        }
    x = frame.copy()
    ts = pd.to_datetime(x["timestamp"], utc=True).sort_values().drop_duplicates()
    step = pd.Timedelta(hours=4)
    delta = ts.diff().dropna()
    gaps = delta > step * 1.5
    missing = int(sum(max(0, round(d / step) - 1) for d in delta[gaps]))
    denominator = max(len(ts) + missing, 1)
    return {
        "rows": int(len(ts)),
        "gap_count": int(gaps.sum()),
        "estimated_missing_bars": missing,
        "missing_fraction": float(missing / denominator),
        "first": ts.iloc[0].isoformat(),
        "last": ts.iloc[-1].isoformat(),
        "median_spacing_seconds": float(delta.median().total_seconds()) if len(delta) else None,
    }


def fetch_venue_frames(venue: str) -> tuple[dict[str, pd.DataFrame], dict]:
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for symbol in EXTERNAL_SYMBOLS:
        try:
            frame = fetch_ccxt_spot_ohlcv(
                venue,
                symbol,
                timeframe=PERIOD,
                start_ms=_ms(START),
                end_ms=_ms(END),
                max_bars=5000,
                page_limit=300,
                max_pages=30,
            )
            if len(frame):
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                frame = frame[
                    (frame["timestamp"] >= START) & (frame["timestamp"] <= END)
                ].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            audit = _gap_audit(frame)
            usable = bool(
                len(frame) >= MIN_BARS
                and float(audit["missing_fraction"]) <= MAX_MISSING_FRACTION
            )
            provenance[symbol] = {
                "venue": venue,
                "source": f"CCXT {venue} public spot OHLCV",
                **audit,
                "usable_by_preregistered_rule": usable,
                "sha256": dataframe_sha256(frame) if len(frame) else None,
            }
            if usable:
                frames[symbol] = frame
        except Exception as exc:
            provenance[symbol] = {
                "venue": venue,
                "source": f"CCXT {venue} public spot OHLCV",
                "usable_by_preregistered_rule": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return frames, provenance


def build_external_events(frames: dict[str, pd.DataFrame], venue: str) -> pd.DataFrame:
    specs = {s.name: s for s in target_specs()}
    parts: list[pd.DataFrame] = []
    for candidate in FROZEN_CANDIDATES_V24C:
        spec = specs[candidate.strategy]
        for symbol, frame in frames.items():
            panel = build_strategy_event_panel(spec, frame, symbol, contract=V24Contract())
            if panel.empty:
                continue
            panel = panel.copy()
            # Do not reuse the v0.24c helper that hard-coded the Bybit source label.
            panel["external_source"] = f"ccxt_{venue}_public_spot"
            panel["external_venue"] = venue
            panel["external_symbol_disjoint_from_v24b"] = True
            parts.append(panel)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for col in ("signal_time", "entry_time", "exit_time", "label_end_time"):
        if col in out:
            out[col] = pd.to_datetime(out[col], utc=True)
    return out.sort_values(["signal_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)


def score_exact_frozen(events: pd.DataFrame, bundles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    predictions: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for candidate in FROZEN_CANDIDATES_V24C:
        strategy = candidate.strategy
        bundle = bundles[strategy]
        fam = events[events["strategy"] == strategy].copy().reset_index(drop=True)
        if fam.empty:
            summaries.append({"strategy": strategy, "status": "DATA_INSUFFICIENT", "events": 0})
            continue
        missing = [c for c in bundle.features if c not in fam.columns]
        if missing:
            raise RuntimeError(f"EXTERNAL_FEATURE_SCHEMA_MISMATCH {strategy}: {missing}")
        score = probability_score(bundle.model, fam[list(bundle.features)])
        selected = np.asarray(score, dtype=float) >= float(candidate.threshold)
        summaries.append({
            "strategy": strategy,
            "status": "EXACT_FROZEN_MODEL_SCORED",
            "events": int(len(fam)),
            "symbols": int(fam["symbol"].nunique()),
            "selected": int(selected.sum()),
            "threshold": float(candidate.threshold),
            "model_class": bundle.model.named_steps["model"].__class__.__name__,
            "model_refit": False,
            "threshold_retuned": False,
        })
        pred = fam[[
            "strategy", "symbol", "signal_time", "entry_time", "exit_time", "side",
            "entry", "exit", "stop", "target", "r_multiple", "exit_reason",
            "external_source", "external_venue",
        ]].copy()
        pred["frozen_score"] = score
        pred["frozen_selected"] = selected
        pred["frozen_threshold"] = float(candidate.threshold)
        predictions.append(pred)
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(summaries),
    )


def positive_symbol_fraction(rows: pd.DataFrame, selected: np.ndarray) -> float:
    if rows.empty:
        return 0.0
    x = rows.reset_index(drop=True)
    mask = np.asarray(selected, dtype=bool)
    flags: list[bool] = []
    for _, group in x.groupby("symbol", sort=True):
        idx = group.index.to_numpy()
        r = 0.0025 * pd.to_numeric(group.loc[mask[idx], "r_multiple"], errors="coerce").dropna()
        flags.append(bool(len(r) and (1.0 + r).prod() - 1.0 > 0))
    return float(np.mean(flags)) if flags else 0.0


def evaluate_venue(
    venue: str,
    frames: dict[str, pd.DataFrame],
    bundles: dict,
    output: Path,
) -> dict:
    venue_out = output / venue
    venue_out.mkdir(parents=True, exist_ok=True)
    events = build_external_events(frames, venue)
    events.to_csv(venue_out / "external_candidate_events.csv", index=False)
    if events.empty:
        return {
            "venue": venue,
            "state": "DATA_INSUFFICIENT",
            "reason": "NO_EXTERNAL_CANDIDATE_EVENTS",
            "economic_pass": False,
        }

    predictions, model_summary = score_exact_frozen(events, bundles)
    predictions.to_csv(venue_out / "external_predictions.csv", index=False)
    model_summary.to_csv(venue_out / "frozen_model_scoring_summary.csv", index=False)
    if predictions.empty:
        return {
            "venue": venue,
            "state": "DATA_INSUFFICIENT",
            "reason": "NO_FROZEN_MODEL_PREDICTIONS",
            "economic_pass": False,
        }

    keys = ["strategy", "symbol", "signal_time", "entry_time", "exit_time"]
    scored = events.merge(
        predictions[keys + ["frozen_score", "frozen_selected", "frozen_threshold"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).sort_values(["entry_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)
    selected = scored["frozen_selected"].to_numpy(dtype=bool)
    scored.to_csv(venue_out / "external_scored_events_full.csv", index=False)

    ci = paired_block_uplift_ci(scored, selected, V24Contract())
    stress = cost_stress_table(scored, selected, V24Contract())
    stress.to_csv(venue_out / "cost_stress_24_36_60bps.csv", index=False)
    breadth = breadth_uplift(
        scored.reset_index(drop=True), selected, group_col="symbol", risk_per_trade=0.0025
    )
    pos_fraction = positive_symbol_fraction(scored, selected)

    mtm_contract = MTMRiskContract()
    base_summary, base_ledger, base_curve = simulate_mtm_portfolio(
        scored, frames, contract=mtm_contract, mode=f"{venue}_external_base"
    )
    filtered_summary, filtered_ledger, filtered_curve = simulate_mtm_portfolio(
        scored,
        frames,
        selected=selected,
        contract=mtm_contract,
        mode=f"{venue}_external_exact_frozen_filter",
    )
    base_ledger.to_csv(venue_out / "mtm_base_ledger.csv", index=False)
    base_curve.to_csv(venue_out / "mtm_base_curve.csv", index=False)
    filtered_ledger.to_csv(venue_out / "mtm_filtered_ledger.csv", index=False)
    filtered_curve.to_csv(venue_out / "mtm_filtered_curve.csv", index=False)
    _dump(venue_out / "mtm_base_summary.json", base_summary)
    _dump(venue_out / "mtm_filtered_summary.json", filtered_summary)

    stress36 = stress.loc[np.isclose(stress["roundtrip_bps"], 36.0)].iloc[0]
    gates = {
        "accepted_trades": int(filtered_summary["accepted"]) >= 100,
        "profit_factor": bool(
            np.isfinite(float(filtered_summary["profit_factor"]))
            and float(filtered_summary["profit_factor"]) >= 1.05
        ),
        "positive_mean_r": bool(
            np.isfinite(float(filtered_summary["mean_r_accepted"]))
            and float(filtered_summary["mean_r_accepted"]) > 0
        ),
        "mtm_drawdown": abs(float(filtered_summary["max_intrabar_stress_drawdown"])) <= 0.05,
        "paired_bootstrap_uplift": bool(np.isfinite(ci[0]) and float(ci[0]) > 0),
        "positive_symbol_fraction": float(pos_fraction) >= 0.60,
        "stress_36bps_pf": bool(
            np.isfinite(float(stress36["filtered_profit_factor"]))
            and float(stress36["filtered_profit_factor"]) >= 1.0
        ),
        "stress_36bps_mean_r": bool(
            np.isfinite(float(stress36["filtered_mean_r"]))
            and float(stress36["filtered_mean_r"]) > 0
        ),
    }
    economic_pass = bool(all(gates.values()))
    summary = {
        "venue": venue,
        "state": "PASS" if economic_pass else "FAIL",
        "economic_pass": economic_pass,
        "usable_symbols": sorted(frames),
        "external_events": int(len(scored)),
        "frozen_selected_events": int(selected.sum()),
        "paired_block_uplift_ci": {"low": float(ci[0]), "high": float(ci[1])},
        "symbol_uplift_breadth": breadth,
        "positive_symbol_fraction": float(pos_fraction),
        "mtm_base": base_summary,
        "mtm_filtered": filtered_summary,
        "stress_36bps": {
            "filtered_profit_factor": float(stress36["filtered_profit_factor"]),
            "filtered_mean_r": float(stress36["filtered_mean_r"]),
        },
        "gates": gates,
        "model_refit_on_external": False,
        "threshold_retuned_on_external": False,
        "live_execution_authorized": False,
    }
    _dump(venue_out / "venue_evidence_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.24d exact frozen-snapshot dual-venue external triangulation")
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v24d-frozen-snapshot-external")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = FrozenSnapshotContract()
    _dump(output / "frozen_snapshot_contract.json", contract.to_dict())

    try:
        runtime = assert_runtime_compatible(contract)
        hashes = verify_snapshot_files(args.snapshot_dir, contract)
        bundles = load_exact_frozen_bundles(args.snapshot_dir, contract)
        replay = replay_archived_validation(args.snapshot_dir, bundles, contract)
        replay.to_csv(output / "exact_archived_validation_replay.csv", index=False)
        _dump(output / "frozen_snapshot_integrity.json", {
            "status": "PASS",
            "runtime": runtime,
            "verified_file_sha256": hashes,
            "strategies": sorted(bundles),
            "model_refit": False,
            "threshold_retuned": False,
        })
    except Exception as exc:
        decision = {
            "version": "v0.24d",
            "stage": "FROZEN_SNAPSHOT_EXTERNAL_TRIANGULATION",
            "state": "BLOCKED",
            "reason": "FROZEN_SNAPSHOT_INTEGRITY_OR_REPLAY_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "external_outcomes_read": False,
            "next_plan": "REPAIR_REPRODUCIBILITY_WITHOUT_RETUNING",
            "forward_paper_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
        _dump(output / "decision.json", decision)
        print(json.dumps(decision, indent=2))
        return

    venue_results: dict[str, dict] = {}
    provenance_all: dict[str, dict] = {}
    for venue in VENUES:
        frames, provenance = fetch_venue_frames(venue)
        provenance_all[venue] = provenance
        _dump(output / f"{venue}_data_provenance.json", provenance)
        if len(frames) < MIN_USABLE_SYMBOLS:
            venue_results[venue] = {
                "venue": venue,
                "state": "DATA_INSUFFICIENT",
                "economic_pass": False,
                "usable_symbols": sorted(frames),
                "usable_symbol_count": int(len(frames)),
                "required_symbol_count": MIN_USABLE_SYMBOLS,
                "reason": "PREREGISTERED_DATA_QUALITY_OR_TRANSPORT_GATE",
            }
            continue
        try:
            venue_results[venue] = evaluate_venue(venue, frames, bundles, output)
        except Exception as exc:
            venue_results[venue] = {
                "venue": venue,
                "state": "BLOCKED",
                "economic_pass": False,
                "usable_symbols": sorted(frames),
                "reason": "VENUE_EVALUATION_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }

    _dump(output / "all_external_data_provenance.json", provenance_all)
    _dump(output / "venue_results.json", venue_results)

    evaluable = [r for r in venue_results.values() if r.get("state") in {"PASS", "FAIL"}]
    failed = [r for r in evaluable if r.get("state") == "FAIL"]
    passed = [r for r in evaluable if r.get("state") == "PASS"]

    if len(evaluable) == len(VENUES) and len(passed) == len(VENUES):
        state = "PASS"
        label = "DUAL_VENUE_EXTERNAL_REPLICATION_PRE_FUTURE_TIME"
        next_plan = "PLAN_B_FUTURE_TIME_FORWARD"
    elif failed:
        state = "FAIL"
        label = "EXTERNAL_REPLICATION_FAILED"
        next_plan = "PLAN_D_RESEARCH_REDESIGN"
    else:
        state = "DATA_INSUFFICIENT"
        label = "EXTERNAL_TRIANGULATION_INCOMPLETE"
        next_plan = "PLAN_B_FUTURE_TIME_FORWARD"

    decision = {
        "version": "v0.24d",
        "stage": "EXACT_FROZEN_SNAPSHOT_DUAL_VENUE_EXTERNAL_TRIANGULATION",
        "state": state,
        "scientific_label": label,
        "venues_preregistered": list(VENUES),
        "symbols_preregistered": list(EXTERNAL_SYMBOLS),
        "venue_results": venue_results,
        "external_calendar_overlap_with_v24b": True,
        "external_symbols_disjoint_from_v24b_internal_panel": True,
        "model_refit_on_external": False,
        "threshold_retuned_on_external": False,
        "same_test_rescue_tuning": False,
        "strict_promotion_ready": False,
        "strict_promotion_blockers": [
            "FUTURE_TIME_PLAN_B_NOT_COMPLETED",
            "CPCV_PBO_DSR_SEARCH_AUDIT_NOT_COMPLETED_FOR_THIS_FROZEN_CHAIN",
            "PROSPECTIVE_FORWARD_PAPER_NOT_COMPLETED",
        ],
        "next_plan": next_plan,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    raw = json.dumps(decision, sort_keys=True, default=str, allow_nan=True).encode("utf-8")
    decision["decision_sha256"] = sha256(raw).hexdigest()
    _dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
