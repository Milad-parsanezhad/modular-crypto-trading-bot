from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from research_bot.bybit_public import audit_bybit_ohlcv, fetch_bybit_spot_ohlcv
from research_bot.coinex_public import fetch_coinex_klines
from research_bot.evidence_ladder_v24c import promotion_ready
from research_bot.ml_framework_v23r import dataframe_sha256
from research_bot.portfolio_mtm_v24c import MTMRiskContract, simulate_mtm_portfolio
from research_bot.strategy_meta_v24 import V24Contract, breadth_uplift, cost_stress_table, paired_block_uplift_ci
from research_bot.v24c_external_plan import (
    EXTERNAL_BYBIT_SYMBOLS_V24C,
    FROZEN_CANDIDATES_V24C,
    INTERNAL_LAST_CLOSED_BAR_UTC,
    INTERNAL_SYMBOLS_V24B,
    build_external_candidate_events,
    build_strategy_dataset_from_frames,
    frozen_manifest,
    reproduce_frozen_models,
    score_external_events,
)

PERIOD = "4hour"
INTERNAL_START = pd.Timestamp("2025-01-19T12:00:00Z")
EXTERNAL_START = INTERNAL_START
EXTERNAL_END = INTERNAL_LAST_CLOSED_BAR_UTC


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=True), encoding="utf-8")


def fetch_frozen_internal_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for symbol in INTERNAL_SYMBOLS_V24B:
        frame = fetch_coinex_klines(
            symbol=symbol, period=PERIOD, market_type="spot",
            start_ms=_ms(INTERNAL_START), end_ms=_ms(INTERNAL_LAST_CLOSED_BAR_UTC), bars=5000,
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame[(frame["timestamp"] >= INTERNAL_START) & (frame["timestamp"] <= INTERNAL_LAST_CLOSED_BAR_UTC)].copy()
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if len(frame) != 3599 or frame["timestamp"].min() != INTERNAL_START or frame["timestamp"].max() != INTERNAL_LAST_CLOSED_BAR_UTC:
            raise RuntimeError(
                f"INTERNAL_FROZEN_DATA_MISMATCH {symbol}: rows={len(frame)} "
                f"first={frame.timestamp.min()} last={frame.timestamp.max()}"
            )
        frames[symbol] = frame
        provenance[symbol] = {
            "source": "CoinEx public spot OHLCV",
            "rows": int(len(frame)),
            "first": frame.timestamp.min().isoformat(),
            "last": frame.timestamp.max().isoformat(),
            "sha256": dataframe_sha256(frame),
        }
    return frames, provenance


def fetch_external_frames(min_bars: int = 1200) -> tuple[dict[str, pd.DataFrame], dict]:
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for symbol in EXTERNAL_BYBIT_SYMBOLS_V24C:
        try:
            frame = fetch_bybit_spot_ohlcv(
                symbol=symbol, period=PERIOD, start_ms=_ms(EXTERNAL_START), end_ms=_ms(EXTERNAL_END), bars=5000,
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame[frame["timestamp"] <= EXTERNAL_END].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            audit = audit_bybit_ohlcv(frame, PERIOD)
            missing_fraction = float(audit.get("estimated_missing_bars", 0)) / max(int(audit.get("rows", 0)), 1)
            usable = bool(len(frame) >= min_bars and missing_fraction <= 0.02)
            provenance[symbol] = {
                "source": "Bybit public spot OHLCV",
                **audit,
                "missing_fraction": missing_fraction,
                "usable_by_predefined_data_quality": usable,
                "sha256": dataframe_sha256(frame) if len(frame) else None,
            }
            if usable:
                frames[symbol] = frame
        except Exception as exc:
            provenance[symbol] = {
                "source": "Bybit public spot OHLCV",
                "usable_by_predefined_data_quality": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return frames, provenance


def positive_symbol_fraction(rows: pd.DataFrame, selected: np.ndarray) -> float:
    if rows.empty:
        return 0.0
    flags = []
    r = rows.reset_index(drop=True)
    s = np.asarray(selected, dtype=bool)
    for _, g in r.groupby("symbol", sort=True):
        idx = g.index.to_numpy()
        vals = 0.0025 * pd.to_numeric(g.loc[s[idx], "r_multiple"], errors="coerce").dropna()
        flags.append(bool(len(vals) and (1.0 + vals).prod() - 1.0 > 0))
    return float(np.mean(flags)) if flags else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.24c Plan A frozen cross-venue/cross-sectional external validation")
    ap.add_argument("--output-dir", default="artifacts/v24c-plan-a-external")
    ap.add_argument("--min-external-bars", type=int, default=1200)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "frozen_manifest.json", frozen_manifest())

    try:
        internal_frames, internal_prov = fetch_frozen_internal_frames()
        _write_json(out / "internal_data_provenance.json", internal_prov)
        internal_dataset = build_strategy_dataset_from_frames(internal_frames)
        internal_dataset.to_csv(out / "internal_reproduced_strategy_dataset.csv", index=False)
        bundles = reproduce_frozen_models(internal_dataset)
        _write_json(out / "frozen_reproduction.json", {
            "status": "EXACT_V24B_CANDIDATES_REPRODUCED",
            "strategies": list(bundles),
            "model_search_performed": False,
            "threshold_search_performed": False,
        })
    except Exception as exc:
        decision = {
            "version": "v0.24c", "plan": "PLAN_A_EXTERNAL_UNTOUCHED",
            "state": "BLOCKED", "reason": "INTERNAL_REPRODUCTION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
            "next_plan": "PLAN_B_FUTURE_TIME_FORWARD",
            "forward_paper_authorized": False, "live_execution_authorized": False,
        }
        _write_json(out / "decision.json", decision)
        print(json.dumps(decision, indent=2)); return

    external_frames, external_prov = fetch_external_frames(args.min_external_bars)
    _write_json(out / "external_data_provenance.json", external_prov)
    if len(external_frames) < 5:
        failures = [v for v in external_prov.values() if v.get("error")]
        state = "BLOCKED" if len(failures) >= len(EXTERNAL_BYBIT_SYMBOLS_V24C) // 2 else "DATA_INSUFFICIENT"
        decision = {
            "version": "v0.24c", "plan": "PLAN_A_EXTERNAL_UNTOUCHED", "state": state,
            "usable_external_symbols": sorted(external_frames),
            "usable_symbol_count": len(external_frames), "required_symbol_count": 5,
            "reason": "EXTERNAL_SOURCE_UNAVAILABLE_OR_INSUFFICIENT_HISTORY",
            "next_plan": "PLAN_B_FUTURE_TIME_FORWARD",
            "forward_paper_authorized": False, "live_execution_authorized": False,
        }
        _write_json(out / "decision.json", decision)
        print(json.dumps(decision, indent=2)); return

    events = build_external_candidate_events(external_frames)
    events.to_csv(out / "external_candidate_events.csv", index=False)
    predictions, ablation = score_external_events(events, bundles)
    predictions.to_csv(out / "external_predictions.csv", index=False)
    ablation.to_csv(out / "external_strategy_ablation.csv", index=False)
    if predictions.empty:
        decision = {
            "version": "v0.24c", "plan": "PLAN_A_EXTERNAL_UNTOUCHED", "state": "DATA_INSUFFICIENT",
            "reason": "NO_EXTERNAL_CANDIDATE_EVENTS", "next_plan": "PLAN_B_FUTURE_TIME_FORWARD",
            "forward_paper_authorized": False, "live_execution_authorized": False,
        }
        _write_json(out / "decision.json", decision); print(json.dumps(decision, indent=2)); return

    keys = ["strategy", "symbol", "signal_time", "entry_time", "exit_time"]
    scored = events.merge(predictions[keys + ["external_score", "external_selected", "frozen_threshold"]], on=keys, how="inner", validate="one_to_one")
    scored = scored.sort_values(["entry_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)
    selected = scored["external_selected"].to_numpy(dtype=bool)
    scored.to_csv(out / "external_scored_events_full.csv", index=False)

    ci = paired_block_uplift_ci(scored, selected, V24Contract())
    stress = cost_stress_table(scored, selected, V24Contract())
    stress.to_csv(out / "cost_stress_24_36_60bps.csv", index=False)
    breadth = breadth_uplift(scored.reset_index(drop=True), selected, group_col="symbol", risk_per_trade=0.0025)
    pos_frac = positive_symbol_fraction(scored, selected)

    mtm_contract = MTMRiskContract()
    base_summary, base_ledger, base_curve = simulate_mtm_portfolio(scored, external_frames, contract=mtm_contract, mode="external_base")
    filt_summary, filt_ledger, filt_curve = simulate_mtm_portfolio(scored, external_frames, selected=selected, contract=mtm_contract, mode="external_frozen_ml_filter")
    base_ledger.to_csv(out / "mtm_base_ledger.csv", index=False); base_curve.to_csv(out / "mtm_base_curve.csv", index=False)
    filt_ledger.to_csv(out / "mtm_filtered_ledger.csv", index=False); filt_curve.to_csv(out / "mtm_filtered_curve.csv", index=False)
    _write_json(out / "mtm_base_summary.json", base_summary); _write_json(out / "mtm_filtered_summary.json", filt_summary)

    stress36 = stress.loc[np.isclose(stress["roundtrip_bps"], 36.0)].iloc[0]
    gate_input = {
        "trades": int(filt_summary["accepted"]),
        "profit_factor": float(filt_summary["profit_factor"]),
        "mean_r": float(filt_summary["mean_r_accepted"]),
        "mtm_drawdown": float(filt_summary["max_intrabar_stress_drawdown"]),
        "positive_symbol_fraction": pos_frac,
        "bootstrap_uplift_low": float(ci[0]),
        # Full CPCV/PBO + DSR must be generated in the internal selection audit;
        # Plan A cannot invent them after seeing the external sample.
        "pbo": float("nan"),
        "dsr_probability": float("nan"),
        "stress_36bps_pass": bool(
            np.isfinite(float(stress36["filtered_profit_factor"]))
            and float(stress36["filtered_profit_factor"]) >= 1.0
            and float(stress36["filtered_mean_r"]) > 0
        ),
        "fresh_external_or_forward": True,
    }
    promotion, gate_failures = promotion_ready(validation=gate_input)
    economic_replication = bool(
        int(filt_summary["accepted"]) >= 100
        and np.isfinite(float(filt_summary["profit_factor"]))
        and float(filt_summary["profit_factor"]) >= 1.05
        and np.isfinite(float(filt_summary["mean_r_accepted"]))
        and float(filt_summary["mean_r_accepted"]) > 0
        and abs(float(filt_summary["max_intrabar_stress_drawdown"])) <= 0.05
        and float(ci[0]) > 0
    )
    decision = {
        "version": "v0.24c", "plan": "PLAN_A_EXTERNAL_UNTOUCHED", "state": "PASS" if economic_replication else "FAIL",
        "scientific_label": "EXTERNAL_REPLICATION_EVIDENCE_PRE_CPCV" if economic_replication else "EXTERNAL_REPLICATION_FAILED",
        "external_design": {
            "venue": "Bybit spot", "calendar_overlap_with_v24b_internal_period": True,
            "symbols_disjoint_from_v24b_12_symbol_training_panel": True,
            "note": "Cross-venue/cross-sectional OOD evidence; future-time Plan B remains stronger evidence against common-market regime overlap.",
        },
        "usable_external_symbols": sorted(external_frames),
        "external_events": int(len(scored)), "external_selected": int(selected.sum()),
        "paired_block_uplift_ci": {"low": float(ci[0]), "high": float(ci[1])},
        "symbol_uplift_breadth": breadth, "positive_symbol_fraction": pos_frac,
        "mtm_base": base_summary, "mtm_filtered": filt_summary,
        "strict_promotion_gate_input": gate_input,
        "strict_promotion_ready": bool(promotion),
        "strict_promotion_failures": gate_failures,
        "pbo_dsr_status": "PENDING_INTERNAL_CPCV_MULTIPLE_TESTING_AUDIT",
        "next_plan": "PLAN_B_FUTURE_TIME_FORWARD" if economic_replication else "PLAN_D_RESEARCH_REDESIGN",
        "model_refit_on_external": False, "threshold_tuned_on_external": False,
        "forward_paper_authorized": False, "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    raw = json.dumps(decision, sort_keys=True, default=str, allow_nan=True).encode("utf-8")
    decision["decision_sha256"] = sha256(raw).hexdigest()
    _write_json(out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
