from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

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
    future_state_is_terminal,
    last_safe_completed_4h_open,
    maturity_state,
)
from research_bot.multitimeframe_strategies_v19 import moving_block_mean_ci
from research_bot.portfolio_mtm_v24c import MTMRiskContract
from research_bot.portfolio_ranked_v25 import simulate_ranked_mtm_portfolio
from research_bot.risk_ranker_v25 import attach_exact_frozen_gate, score_ranker
from research_bot.v24c_external_plan import build_external_candidate_events


RANKER_SNAPSHOT_SHA256 = "737c5196dff449fbb587912bedf5f89e1a7eca13643a207bb57fb3e259f8240b"
RANKER_MANIFEST_SHA256 = "ffcc1fa73b548bb5bfd06191b8ef1e8f967b07a171f3d903ceacfe99d53ac7ed"
RANKER_DECISION_SHA256 = "934ce6952b580b1d10853f7e3b3f66cb54a771483c6097557ab83fccccd23cac"
RANKER_SOURCE_RUN = "34590214474"
RANKER_SOURCE_ARTIFACT = "v25-risk-aware-ranking-34590214474"
PRIMARY_COST_BPS = 24.0
STRESS_COSTS_BPS: tuple[float, ...] = (24.0, 36.0, 60.0)


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str, allow_nan=True), encoding="utf-8")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "source"]
    x = frame.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan if c != "source" else ""
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[cols].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _frame_sha256(frame: pd.DataFrame) -> str:
    x = _canonical_frame(frame).copy()
    x["timestamp"] = x["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    raw = x.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    return sha256(raw).hexdigest()


def _load_previous_decision(root: Path | None) -> dict | None:
    if root is None:
        return None
    p = root / "decision.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_previous_frame(root: Path | None, venue: str, symbol: str) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    p = root / "ohlcv" / venue / f"{_safe_symbol(symbol)}.csv"
    if not p.is_file():
        return pd.DataFrame()
    return _canonical_frame(pd.read_csv(p))


def _merge_append_only(previous: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Preserve first-observed rows and append only newly observed timestamps.

    Exchanges can occasionally restate OHLCV history.  We count such differences
    but never silently replace an already archived bar in the prospective chain.
    """
    prev = _canonical_frame(previous) if len(previous) else pd.DataFrame()
    new = _canonical_frame(fresh) if len(fresh) else pd.DataFrame()
    if prev.empty:
        return new, 0
    if new.empty:
        return prev, 0

    numeric = ["open", "high", "low", "close", "volume"]
    overlap = prev[["timestamp", *numeric]].merge(new[["timestamp", *numeric]], on="timestamp", suffixes=("_old", "_new"))
    revisions = 0
    for _, row in overlap.iterrows():
        changed = False
        for c in numeric:
            a = float(row[f"{c}_old"]) if pd.notna(row[f"{c}_old"]) else np.nan
            b = float(row[f"{c}_new"]) if pd.notna(row[f"{c}_new"]) else np.nan
            if (np.isnan(a) != np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and not np.isclose(a, b, rtol=1e-10, atol=1e-12)):
                changed = True
                break
        revisions += int(changed)

    last_prev = prev["timestamp"].max()
    appended = new[new["timestamp"] > last_prev]
    merged = pd.concat([prev, appended], ignore_index=True)
    return _canonical_frame(merged), revisions


def _load_ranker(root: Path, contract: V25FutureEvidenceContract) -> dict:
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
    ranker_start = pd.Timestamp(snapshot.get("future_start_utc"))
    if ranker_start > contract.start:
        raise RuntimeError("V25_RANKER_WAS_NOT_FROZEN_BEFORE_OPERATIONAL_FUTURE_BOUNDARY")
    if any(bool(snapshot.get(k)) for k in ("forward_paper_authorized", "paper_replacement_authorized", "live_execution_authorized")):
        raise RuntimeError("V25_UNSAFE_RANKER_SNAPSHOT")
    return {"snapshot": snapshot, "hashes": observed, "ranker_frozen_boundary": ranker_start.isoformat()}


def _fetch_venue(
    venue: str,
    contract: V25FutureEvidenceContract,
    end: pd.Timestamp,
    out: Path,
    previous_root: Path | None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    start = context_start(contract)
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    raw_dir = out / "ohlcv" / venue
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in contract.symbols:
        previous = _load_previous_frame(previous_root, venue, symbol)
        fetch_error = None
        try:
            fresh = fetch_ccxt_spot_ohlcv(
                venue,
                symbol,
                timeframe="4h",
                start_ms=_ms(start),
                end_ms=_ms(end),
                max_bars=contract.max_bars,
                page_limit=300,
                max_pages=20,
            )
        except Exception as exc:
            fresh = pd.DataFrame()
            fetch_error = f"{type(exc).__name__}: {exc}"

        merged, revisions = _merge_append_only(previous, fresh)
        if len(merged):
            merged = merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)].reset_index(drop=True)
        merged.to_csv(raw_dir / f"{_safe_symbol(symbol)}.csv", index=False)

        missing = estimated_missing_fraction(merged, start=start, end=end)
        usable = bool(len(merged) >= contract.min_context_bars and missing <= contract.max_missing_fraction)
        provenance[symbol] = {
            "venue": venue,
            "rows": int(len(merged)),
            "first": merged["timestamp"].min().isoformat() if len(merged) else None,
            "last": merged["timestamp"].max().isoformat() if len(merged) else None,
            "missing_fraction": float(missing),
            "usable": usable,
            "source": f"CCXT {venue} public spot OHLCV",
            "first_observation_preserved": bool(len(previous)),
            "overlap_restatements_detected": int(revisions),
            "frame_sha256": _frame_sha256(merged),
            "fetch_error": fetch_error,
        }
        if usable:
            frames[symbol] = merged
    return frames, provenance


def _future_rows(frames: dict[str, pd.DataFrame], contract: V25FutureEvidenceContract, end: pd.Timestamp) -> pd.DataFrame:
    if len(frames) < contract.min_usable_symbols_per_venue:
        return pd.DataFrame()
    events_all = build_external_candidate_events(frames)
    future = future_event_slice(events_all, last_completed_bar_open=end, contract=contract)
    assert_no_preboundary_evidence(future, contract)
    return future


def _stress_r_multiple(events: pd.DataFrame, roundtrip_bps: float) -> pd.Series:
    required = {"entry", "stop", "gross_return"}
    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"V25_COST_STRESS_COLUMNS_MISSING {sorted(missing)}")
    entry = pd.to_numeric(events["entry"], errors="coerce")
    stop = pd.to_numeric(events["stop"], errors="coerce")
    gross = pd.to_numeric(events["gross_return"], errors="coerce")
    stop_fraction = (entry - stop).abs() / entry.abs().replace(0, np.nan)
    return (gross - float(roundtrip_bps) / 10_000.0) / stop_fraction.replace(0, np.nan)


def _paired_curve_uplift_ci(base_curve: pd.DataFrame, ranked_curve: pd.DataFrame) -> dict:
    if base_curve.empty or ranked_curve.empty:
        return {"low": np.nan, "high": np.nan, "observations": 0}
    left = base_curve[["timestamp", "close_mtm_equity"]].copy()
    right = ranked_curve[["timestamp", "close_mtm_equity"]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    merged = left.merge(right, on="timestamp", suffixes=("_base", "_ranked"), validate="one_to_one").sort_values("timestamp")
    if len(merged) < 20:
        return {"low": np.nan, "high": np.nan, "observations": int(max(0, len(merged) - 1))}
    base_ret = merged["close_mtm_equity_base"].pct_change()
    ranked_ret = merged["close_mtm_equity_ranked"].pct_change()
    delta = (ranked_ret - base_ret).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(delta) < 20:
        return {"low": np.nan, "high": np.nan, "observations": int(len(delta))}
    block = max(5, min(20, max(5, len(delta) // 8)))
    low, high = moving_block_mean_ci(delta, samples=1200, block=block, seed=250911)
    return {"low": float(low), "high": float(high), "observations": int(len(delta)), "block": int(block)}


def _evaluate_venue(
    venue: str,
    future: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    bundles: dict,
    ranker_snapshot: dict,
    contract: V25FutureEvidenceContract,
    out: Path,
) -> dict:
    venue_dir = out / venue
    venue_dir.mkdir(parents=True, exist_ok=True)
    if future.empty:
        return {"venue": venue, "future_events": 0, "state": "NO_CLOSED_FUTURE_EVENTS"}

    scored = attach_exact_frozen_gate(future, bundles)
    scored["ranker_priority"] = score_ranker(scored, ranker_snapshot)
    scored["external_source"] = f"CCXT {venue} public spot OHLCV"
    scored.to_csv(venue_dir / "future_scored_events_first_mature_read.csv", index=False)
    selected = scored["frozen_selected"].astype(bool).to_numpy()

    stress_summaries: dict[str, dict] = {}
    curves: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for bps in STRESS_COSTS_BPS:
        stressed = scored.copy()
        if not np.isclose(bps, PRIMARY_COST_BPS):
            stressed["r_multiple"] = _stress_r_multiple(stressed, bps)
        risk_contract = MTMRiskContract(liquidation_roundtrip_bps=float(bps))
        base_summary, base_ledger, base_curve = simulate_ranked_mtm_portfolio(
            stressed,
            frames,
            stressed["frozen_score"].to_numpy(dtype=float),
            selected=selected,
            contract=risk_contract,
            mode=f"{venue}_future_frozen_score_{int(bps)}bps",
        )
        rank_summary, rank_ledger, rank_curve = simulate_ranked_mtm_portfolio(
            stressed,
            frames,
            stressed["ranker_priority"].to_numpy(dtype=float),
            selected=selected,
            contract=risk_contract,
            mode=f"{venue}_future_hgb_ranker_{int(bps)}bps",
        )
        stress_summaries[str(int(bps))] = {"baseline": base_summary, "ranked": rank_summary}
        _dump(venue_dir / f"baseline_summary_{int(bps)}bps.json", base_summary)
        _dump(venue_dir / f"ranker_summary_{int(bps)}bps.json", rank_summary)
        base_ledger.to_csv(venue_dir / f"baseline_ledger_{int(bps)}bps.csv", index=False)
        rank_ledger.to_csv(venue_dir / f"ranker_ledger_{int(bps)}bps.csv", index=False)
        base_curve.to_csv(venue_dir / f"baseline_curve_{int(bps)}bps.csv", index=False)
        rank_curve.to_csv(venue_dir / f"ranker_curve_{int(bps)}bps.csv", index=False)
        curves[float(bps)] = (base_curve, rank_curve)

    primary = stress_summaries[str(int(PRIMARY_COST_BPS))]
    stress36 = stress_summaries[str(int(contract.required_cost_stress_bps))]["ranked"]
    paired_ci = _paired_curve_uplift_ci(*curves[PRIMARY_COST_BPS])
    gate = allocator_gate(
        primary["baseline"],
        primary["ranked"],
        ranked_cost_stress=stress36,
        paired_uplift_ci=paired_ci,
        contract=contract,
    )

    summary = {
        "venue": venue,
        "future_events": int(len(scored)),
        "frozen_selected_events": int(selected.sum()),
        "stress_summaries": stress_summaries,
        "paired_portfolio_uplift_ci": paired_ci,
        "allocator_gate": gate,
        "model_refit_on_future": False,
        "threshold_tuned_on_future": False,
        "ranker_refit_on_future": False,
        "economics_were_blinded_before_first_mature_read": True,
    }
    _dump(venue_dir / "venue_first_mature_evaluation.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24b-dir", required=True)
    parser.add_argument("--v25-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/v25-future-evidence")
    parser.add_argument("--previous-dir", default=None, help="previous scheduled artifact; preserves first-observed bars and terminal first-look state")
    parser.add_argument("--now-utc", default=None, help="testing/reproduction override only")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous_root = Path(args.previous_dir) if args.previous_dir else None
    previous_decision = _load_previous_decision(previous_root)

    # A mature PASS or FAIL spends the sample. Later scheduled runs are forbidden
    # from expanding the same test window and re-reading economics.
    if future_state_is_terminal(previous_decision):
        _dump(out / "decision.json", previous_decision)
        _dump(out / "terminal_lock.json", {
            "state": "FUTURE_EVIDENCE_ALREADY_FROZEN",
            "terminal_state": previous_decision.get("state"),
            "evaluation_window_end": previous_decision.get("evaluation_window_end"),
            "no_repeated_economic_look": True,
        })
        print(json.dumps({"state": "FUTURE_EVIDENCE_ALREADY_FROZEN", "terminal_state": previous_decision.get("state")}, indent=2))
        return

    contract = V25FutureEvidenceContract()
    end = last_safe_completed_4h_open(pd.Timestamp(args.now_utc) if args.now_utc else None)
    frozen_contract = FrozenSnapshotContract()
    v24b_hashes = verify_snapshot_files(Path(args.v24b_dir), frozen_contract)
    runtime = assert_runtime_compatible(frozen_contract)
    bundles = load_exact_frozen_bundles(Path(args.v24b_dir), frozen_contract)
    ranker = _load_ranker(Path(args.v25_dir), contract)

    identity = {
        "version": "v0.25-prospective",
        "future_contract": contract.to_dict(),
        "last_safe_completed_4h_open": end.isoformat(),
        "v24b_hashes": v24b_hashes,
        "ranker_hashes": ranker["hashes"],
        "ranker_source_run": RANKER_SOURCE_RUN,
        "ranker_source_artifact": RANKER_SOURCE_ARTIFACT,
        "ranker_frozen_boundary": ranker["ranker_frozen_boundary"],
        "runtime": runtime,
        "previous_artifact_present": bool(previous_root and previous_root.exists()),
    }
    _dump(out / "future_snapshot_identity.json", identity)

    if end < contract.start:
        decision = {
            "version": "v0.25-prospective",
            "state": "WAITING_FOR_FUTURE_BOUNDARY",
            "scientific_label": "BLINDED_NOT_STARTED",
            "last_safe_completed_4h_open": end.isoformat(),
            "future_start_utc": contract.future_start_utc,
            "economics_blinded": True,
            "sample_spent": False,
            "forward_paper_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
        _dump(out / "decision.json", decision)
        print(json.dumps(decision, indent=2))
        return

    venue_counts: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    frames_by_venue: dict[str, dict[str, pd.DataFrame]] = {}
    future_by_venue: dict[str, pd.DataFrame] = {}

    for venue in contract.venues:
        frames, prov = _fetch_venue(venue, contract, end, out, previous_root)
        frames_by_venue[venue] = frames
        provenance[venue] = prov
        future = _future_rows(frames, contract, end)
        future_by_venue[venue] = future
        venue_counts[venue] = {
            "venue": venue,
            "usable_symbols": int(len(frames)),
            "future_events": int(len(future)),
            "economic_metrics_exposed": False,
        }

    _dump(out / "future_data_provenance.json", provenance)
    _dump(out / "blinded_venue_counts.json", venue_counts)
    maturity = maturity_state(venue_results=venue_counts, last_completed_bar_open=end, contract=contract)

    if not maturity["mature"]:
        decision = {
            "version": "v0.25-prospective",
            "state": maturity["state"],
            "scientific_label": "BLINDED_FUTURE_SAMPLE_ACCUMULATING",
            "future_start_utc": contract.future_start_utc,
            "last_safe_completed_4h_open": end.isoformat(),
            "maturity": maturity,
            "venue_counts": venue_counts,
            "economics_blinded": True,
            "sample_spent": False,
            "next_gate": "CONTINUE_APPEND_ONLY_FROZEN_COLLECTION",
            "forward_paper_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
        decision["decision_sha256"] = sha256(json.dumps(decision, sort_keys=True, default=str).encode()).hexdigest()
        _dump(out / "decision.json", decision)
        print(json.dumps(decision, indent=2, default=str))
        return

    # FIRST AND ONLY economic look for this prospective sample.
    evaluations: dict[str, dict] = {}
    for venue in contract.venues:
        evaluations[venue] = _evaluate_venue(
            venue,
            future_by_venue[venue],
            frames_by_venue[venue],
            bundles,
            ranker["snapshot"],
            contract,
            out,
        )

    venue_gates = {venue: evaluations[venue].get("allocator_gate", {"passed": False, "gates": {}}) for venue in contract.venues}
    scientific_pass = bool(all(venue_gates[venue].get("passed", False) for venue in contract.venues))
    state = "FUTURE_ALLOCATOR_GATE_PASS_PRE_SEARCH_AUDIT" if scientific_pass else "FUTURE_ALLOCATOR_GATE_FAIL"
    label = "FUTURE_TIME_CANDIDATE_PRE_CPCV_PBO_DSR" if scientific_pass else "REJECTED_ON_FRESH_FUTURE_EVIDENCE"

    decision = {
        "version": "v0.25-prospective",
        "state": state,
        "scientific_label": label,
        "future_start_utc": contract.future_start_utc,
        "evaluation_window_end": end.isoformat(),
        "maturity": maturity,
        "venue_counts": venue_counts,
        "venue_gates": venue_gates,
        "scientific_pass": scientific_pass,
        "economics_blinded_until_first_mature_read": True,
        "first_mature_read_is_terminal": True,
        "sample_spent": True,
        "same_sample_rescue_tuning_allowed": False,
        "next_gate": "CPCV_PBO_DSR_AND_FORWARD_PAPER_REVIEW" if scientific_pass else "PLAN_D_RESEARCH_REDESIGN_NEW_CLOCK",
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    decision["decision_sha256"] = sha256(json.dumps(decision, sort_keys=True, default=str).encode()).hexdigest()
    _dump(out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
