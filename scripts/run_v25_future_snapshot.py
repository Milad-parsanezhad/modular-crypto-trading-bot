from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_bot.ccxt_external_v24c import fetch_ccxt_spot_ohlcv
from research_bot.frozen_snapshot_v24d import (
    FrozenSnapshotContract,
    assert_runtime_compatible,
    load_exact_frozen_bundles,
    verify_snapshot_files,
)
from research_bot.future_evidence_v25 import (
    V25FutureEvidenceContract,
    allocator_gate,
    assert_no_preboundary_evidence,
    context_start,
    estimated_missing_fraction,
    future_event_slice,
    last_safe_completed_4h_open,
    maturity_state,
)
from research_bot.multitimeframe_strategies_v19 import moving_block_mean_ci
from research_bot.portfolio_mtm_v24c import MTMRiskContract
from research_bot.portfolio_ranked_v25 import simulate_ranked_mtm_portfolio
from research_bot.risk_ranker_v25 import attach_exact_frozen_gate, score_ranker
from research_bot.strategy_meta_v24 import V24Contract, cost_stress_table
from research_bot.v24c_external_plan import build_external_candidate_events


RANKER_SNAPSHOT_SHA256 = "737c5196dff449fbb587912bedf5f89e1a7eca13643a207bb57fb3e259f8240b"
RANKER_MANIFEST_SHA256 = "ffcc1fa73b548bb5bfd06191b8ef1e8f967b07a171f3d903ceacfe99d53ac7ed"
RANKER_DECISION_SHA256 = "934ce6952b580b1d10853f7e3b3f66cb54a771483c6097557ab83fccccd23cac"
RANKER_SOURCE_RUN = "34590214474"
RANKER_SOURCE_ARTIFACT = "v25-risk-aware-ranking-34590214474"


def _sha(path: Path) -> str:
    h = sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str, allow_nan=True), encoding="utf-8")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _load_ranker(root: Path) -> dict:
    checks = {
        "ranker_snapshot.joblib": RANKER_SNAPSHOT_SHA256,
        "ranker_manifest.json": RANKER_MANIFEST_SHA256,
        "decision.json": RANKER_DECISION_SHA256,
    }
    observed = {}
    for name, expected in checks.items():
        p = root / name
        if not p.is_file():
            raise RuntimeError(f"V25_RANKER_FILE_MISSING {name}")
        observed[name] = _sha(p)
        if observed[name] != expected:
            raise RuntimeError(f"V25_RANKER_HASH_MISMATCH {name}: {observed[name]} != {expected}")
    snapshot = joblib.load(root / "ranker_snapshot.joblib")
    if snapshot.get("champion") != "hgb_expected_r":
        raise RuntimeError(f"V25_RANKER_IDENTITY_MISMATCH champion={snapshot.get('champion')}")
    if snapshot.get("future_start_utc") != "2026-09-11T12:00:00Z":
        raise RuntimeError("V25_FUTURE_BOUNDARY_MISMATCH")
    if any(bool(snapshot.get(k)) for k in ("forward_paper_authorized", "paper_replacement_authorized", "live_execution_authorized")):
        raise RuntimeError("V25_UNSAFE_RANKER_SNAPSHOT")
    return {"snapshot": snapshot, "hashes": observed}


def _fetch_venue(venue: str, contract: V25FutureEvidenceContract, end: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], dict]:
    start = context_start(contract)
    frames: dict[str, pd.DataFrame] = {}
    provenance = {}
    for symbol in contract.symbols:
        try:
            frame = fetch_ccxt_spot_ohlcv(
                venue, symbol, timeframe="4h", start_ms=_ms(start), end_ms=_ms(end),
                max_bars=contract.max_bars, page_limit=300, max_pages=20,
            )
            if len(frame):
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            missing = estimated_missing_fraction(frame, start=start, end=end)
            usable = bool(len(frame) >= contract.min_context_bars and missing <= contract.max_missing_fraction)
            provenance[symbol] = {
                "venue": venue,
                "rows": int(len(frame)),
                "first": frame["timestamp"].min().isoformat() if len(frame) else None,
                "last": frame["timestamp"].max().isoformat() if len(frame) else None,
                "missing_fraction": float(missing),
                "usable": usable,
                "source": f"CCXT {venue} public spot OHLCV",
            }
            if usable:
                frames[symbol] = frame
        except Exception as exc:
            provenance[symbol] = {"venue": venue, "usable": False, "error": f"{type(exc).__name__}: {exc}"}
    return frames, provenance


def _ledger_mask_and_scale(events: pd.DataFrame, ledger: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    keys = ["entry_time", "exit_time", "strategy", "symbol", "side"]
    e = events.copy().reset_index(drop=True)
    l = ledger[keys + ["accepted", "risk_scale"]].copy()
    for c in ("entry_time", "exit_time"):
        e[c] = pd.to_datetime(e[c], utc=True)
        l[c] = pd.to_datetime(l[c], utc=True)
    if l.duplicated(keys).any() or e.duplicated(keys).any():
        raise RuntimeError("V25_FUTURE_EVENT_KEY_NOT_UNIQUE")
    merged = e[keys].merge(l, on=keys, how="left", validate="one_to_one")
    if merged["accepted"].isna().any():
        raise RuntimeError("V25_FUTURE_LEDGER_ALIGNMENT_FAILED")
    return merged["accepted"].astype(bool).to_numpy(), pd.to_numeric(merged["risk_scale"], errors="coerce").fillna(1.0).to_numpy(dtype=float)


def _venue_evidence(
    venue: str,
    frames: dict[str, pd.DataFrame],
    bundles: dict,
    ranker_snapshot: dict,
    contract: V25FutureEvidenceContract,
    end: pd.Timestamp,
    out: Path,
) -> dict:
    venue_dir = out / venue; venue_dir.mkdir(parents=True, exist_ok=True)
    events_all = build_external_candidate_events(frames)
    future = future_event_slice(events_all, last_completed_bar_open=end, contract=contract)
    assert_no_preboundary_evidence(future, contract)
    if future.empty:
        summary = {"venue": venue, "usable_symbols": len(frames), "future_events": 0, "ranked_accepted": 0, "state": "NO_CLOSED_FUTURE_EVENTS_YET"}
        _dump(venue_dir / "venue_summary.json", summary)
        return summary

    future = attach_exact_frozen_gate(future, bundles)
    future["ranker_priority"] = score_ranker(future, ranker_snapshot)
    future["external_source"] = f"CCXT {venue} public spot OHLCV"
    future.to_csv(venue_dir / "future_scored_events.csv", index=False)

    selected = future["frozen_selected"].astype(bool).to_numpy()
    risk_contract = MTMRiskContract()
    base_summary, base_ledger, base_curve = simulate_ranked_mtm_portfolio(
        future, frames, future["frozen_score"].to_numpy(dtype=float), selected=selected,
        contract=risk_contract, mode=f"{venue}_future_frozen_score",
    )
    rank_summary, rank_ledger, rank_curve = simulate_ranked_mtm_portfolio(
        future, frames, future["ranker_priority"].to_numpy(dtype=float), selected=selected,
        contract=risk_contract, mode=f"{venue}_future_hgb_ranker",
    )
    base_ledger.to_csv(venue_dir / "baseline_ledger.csv", index=False); base_curve.to_csv(venue_dir / "baseline_curve.csv", index=False)
    rank_ledger.to_csv(venue_dir / "ranker_ledger.csv", index=False); rank_curve.to_csv(venue_dir / "ranker_curve.csv", index=False)
    _dump(venue_dir / "baseline_summary.json", base_summary); _dump(venue_dir / "ranker_summary.json", rank_summary)

    base_mask, base_scale = _ledger_mask_and_scale(future, base_ledger)
    rank_mask, rank_scale = _ledger_mask_and_scale(future, rank_ledger)
    v24 = V24Contract()
    base_stress = cost_stress_table(future, base_mask, v24); base_stress.to_csv(venue_dir / "baseline_cost_stress.csv", index=False)
    rank_stress = cost_stress_table(future, rank_mask, v24); rank_stress.to_csv(venue_dir / "ranker_cost_stress.csv", index=False)

    rr = pd.to_numeric(future["r_multiple"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    delta_proxy = v24.risk_per_trade * rr * (rank_mask.astype(float) * rank_scale - base_mask.astype(float) * base_scale)
    ci = moving_block_mean_ci(delta_proxy, samples=1000, block=max(5, min(20, max(5, len(delta_proxy) // 10))), seed=250314)
    stress36 = rank_stress.loc[np.isclose(rank_stress["roundtrip_bps"], contract.required_cost_stress_bps)].iloc[0]

    summary = {
        "venue": venue,
        "usable_symbols": len(frames),
        "future_events": int(len(future)),
        "frozen_selected_events": int(selected.sum()),
        "baseline_accepted": int(base_summary.get("accepted", 0)),
        "ranked_accepted": int(rank_summary.get("accepted", 0)),
        "baseline": base_summary,
        "ranked": rank_summary,
        "paired_allocator_uplift_ci": {"low": float(ci[0]), "high": float(ci[1])},
        "ranked_36bps_pf": float(stress36["filtered_profit_factor"]),
        "ranked_36bps_mean_r": float(stress36["filtered_mean_r"]),
        "allocator_gate": allocator_gate(base_summary, rank_summary, contract=contract),
        "model_refit_on_future": False,
        "threshold_tuned_on_future": False,
        "ranker_refit_on_future": False,
    }
    _dump(venue_dir / "venue_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24b-dir", required=True)
    parser.add_argument("--v25-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v25-future-evidence")
    parser.add_argument("--now-utc", default=None, help="testing/reproduction override only")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    contract = V25FutureEvidenceContract()
    end = last_safe_completed_4h_open(pd.Timestamp(args.now_utc) if args.now_utc else None)

    frozen_contract = FrozenSnapshotContract()
    v24b_hashes = verify_snapshot_files(Path(args.v24b_dir), frozen_contract)
    runtime = assert_runtime_compatible(frozen_contract)
    bundles = load_exact_frozen_bundles(Path(args.v24b_dir), frozen_contract)
    ranker = _load_ranker(Path(args.v25_dir))

    identity = {
        "version": "v0.25",
        "future_contract": contract.to_dict(),
        "last_safe_completed_4h_open": end.isoformat(),
        "v24b_hashes": v24b_hashes,
        "ranker_hashes": ranker["hashes"],
        "ranker_source_run": RANKER_SOURCE_RUN,
        "ranker_source_artifact": RANKER_SOURCE_ARTIFACT,
        "runtime": runtime,
    }
    _dump(out / "future_snapshot_identity.json", identity)

    if end < contract.start:
        decision = {
            "version": "v0.25",
            "state": "WAITING_FOR_FUTURE_BOUNDARY",
            "last_safe_completed_4h_open": end.isoformat(),
            "future_start_utc": contract.future_start_utc,
            "forward_paper_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
        _dump(out / "decision.json", decision); print(json.dumps(decision, indent=2)); return

    venue_results = {}; provenance = {}
    for venue in contract.venues:
        frames, prov = _fetch_venue(venue, contract, end)
        provenance[venue] = prov
        if len(frames) < contract.min_usable_symbols_per_venue:
            venue_results[venue] = {"venue": venue, "usable_symbols": len(frames), "future_events": 0, "ranked_accepted": 0, "state": "DATA_INSUFFICIENT"}
            continue
        venue_results[venue] = _venue_evidence(venue, frames, bundles, ranker["snapshot"], contract, end, out)
    _dump(out / "future_data_provenance.json", provenance)
    _dump(out / "venue_results.json", venue_results)

    maturity = maturity_state(venue_results=venue_results, last_completed_bar_open=end, contract=contract)
    venue_gates = {v: r.get("allocator_gate", {"passed": False, "gates": {}}) for v, r in venue_results.items()}
    scientific_pass = bool(maturity["mature"] and all(venue_gates.get(v, {}).get("passed") for v in contract.venues))
    if not maturity["mature"]:
        state = maturity["state"]
        label = "FORWARD_SAMPLE_IMMATURE"
    elif scientific_pass:
        state = "FUTURE_ALLOCATOR_GATE_PASS_PRE_SEARCH_AUDIT"
        label = "FUTURE_TIME_CANDIDATE_PRE_CPCV_PBO_DSR"
    else:
        state = "FUTURE_ALLOCATOR_GATE_FAIL"
        label = "REJECTED_ON_FRESH_FUTURE_EVIDENCE"

    decision = {
        "version": "v0.25",
        "state": state,
        "scientific_label": label,
        "future_start_utc": contract.future_start_utc,
        "last_safe_completed_4h_open": end.isoformat(),
        "maturity": maturity,
        "venue_gates": venue_gates,
        "scientific_pass": scientific_pass,
        "next_gate": "CPCV_PBO_DSR_AND_FORWARD_PAPER_REVIEW" if scientific_pass else ("CONTINUE_FROZEN_COLLECTION" if not maturity["mature"] else "PLAN_D_RESEARCH_REDESIGN_NEW_CLOCK"),
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    decision["decision_sha256"] = sha256(json.dumps(decision, sort_keys=True, default=str).encode()).hexdigest()
    _dump(out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
