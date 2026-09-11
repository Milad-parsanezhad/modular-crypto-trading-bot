from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from research_bot.ccxt_external_v24c import fetch_ccxt_spot_ohlcv
from research_bot.coinex_public import fetch_coinex_klines
from research_bot.ml_framework_v23r import dataframe_sha256
from research_bot.portfolio_mtm_v24c import MTMRiskContract, simulate_mtm_portfolio
from research_bot.strategy_meta_v24 import V24Contract, breadth_uplift, cost_stress_table, paired_block_uplift_ci
from research_bot.v24c_external_plan import (
    EXTERNAL_BYBIT_SYMBOLS_V24C,
    INTERNAL_LAST_CLOSED_BAR_UTC,
    INTERNAL_SYMBOLS_V24B,
    build_external_candidate_events,
    build_strategy_dataset_from_frames,
    reproduce_frozen_models,
    score_external_events,
)

START = pd.Timestamp("2025-01-19T12:00:00Z")
END = INTERNAL_LAST_CLOSED_BAR_UTC
EXCHANGE_ID = "okx"
MIN_BARS = 1200


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _internal() -> tuple[dict[str, pd.DataFrame], dict]:
    frames = {}; prov = {}
    for symbol in INTERNAL_SYMBOLS_V24B:
        f = fetch_coinex_klines(symbol=symbol, period="4hour", market_type="spot", start_ms=_ms(START), end_ms=_ms(END), bars=5000)
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        f = f[(f.timestamp >= START) & (f.timestamp <= END)].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if len(f) != 3599 or f.timestamp.min() != START or f.timestamp.max() != END:
            raise RuntimeError(f"INTERNAL_FROZEN_DATA_MISMATCH {symbol}: rows={len(f)}")
        frames[symbol] = f
        prov[symbol] = {"rows": len(f), "sha256": dataframe_sha256(f), "source": "CoinEx public spot OHLCV"}
    return frames, prov


def _external() -> tuple[dict[str, pd.DataFrame], dict]:
    frames = {}; prov = {}
    for symbol in EXTERNAL_BYBIT_SYMBOLS_V24C:
        try:
            f = fetch_ccxt_spot_ohlcv(EXCHANGE_ID, symbol, timeframe="4h", start_ms=_ms(START), end_ms=_ms(END), max_bars=5000)
            if len(f):
                f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
            expected = max(1, int((END - max(START, f.timestamp.min() if len(f) else END)) / pd.Timedelta(hours=4)) + 1) if len(f) else 1
            missing_fraction = max(0.0, 1.0 - len(f) / expected) if len(f) else 1.0
            usable = bool(len(f) >= MIN_BARS and missing_fraction <= 0.02)
            prov[symbol] = {
                "source": f"CCXT {EXCHANGE_ID} public spot OHLCV", "rows": int(len(f)),
                "first": f.timestamp.min().isoformat() if len(f) else None,
                "last": f.timestamp.max().isoformat() if len(f) else None,
                "missing_fraction_approx": float(missing_fraction), "usable": usable,
                "sha256": dataframe_sha256(f) if len(f) else None,
            }
            if usable: frames[symbol] = f
        except Exception as exc:
            prov[symbol] = {"source": f"CCXT {EXCHANGE_ID}", "usable": False, "error": f"{type(exc).__name__}: {exc}"}
    return frames, prov


def _positive_symbol_fraction(rows: pd.DataFrame, selected: np.ndarray) -> float:
    r = rows.reset_index(drop=True); s = np.asarray(selected, bool); flags = []
    for _, g in r.groupby("symbol", sort=True):
        idx = g.index.to_numpy(); vals = 0.0025 * pd.to_numeric(g.loc[s[idx], "r_multiple"], errors="coerce").dropna()
        flags.append(bool(len(vals) and (1 + vals).prod() - 1 > 0))
    return float(np.mean(flags)) if flags else 0.0


def main() -> None:
    out = Path("artifacts/v24c-plan-a-okx-fallback"); out.mkdir(parents=True, exist_ok=True)
    try:
        internal, iprov = _internal(); _dump(out / "internal_data_provenance.json", iprov)
        dataset = build_strategy_dataset_from_frames(internal)
        bundles = reproduce_frozen_models(dataset)
        _dump(out / "frozen_reproduction.json", {"status": "EXACT_V24B_CANDIDATES_REPRODUCED", "strategies": list(bundles)})
    except Exception as exc:
        d = {"state": "BLOCKED", "reason": "INTERNAL_REPRODUCTION_FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=4), "next_plan": "PLAN_B_FUTURE_TIME_FORWARD", "forward_paper_authorized": False, "live_execution_authorized": False}
        _dump(out / "decision.json", d); print(json.dumps(d, indent=2)); return

    external, eprov = _external(); _dump(out / "external_data_provenance.json", eprov)
    if len(external) < 5:
        d = {"version": "v0.24c", "plan": "PLAN_A_TECHNICAL_FALLBACK_OKX", "state": "BLOCKED" if not external else "DATA_INSUFFICIENT", "usable_symbols": sorted(external), "reason": "OKX_FALLBACK_INSUFFICIENT", "next_plan": "PLAN_B_FUTURE_TIME_FORWARD", "forward_paper_authorized": False, "live_execution_authorized": False}
        _dump(out / "decision.json", d); print(json.dumps(d, indent=2)); return

    events = build_external_candidate_events(external)
    preds, ablation = score_external_events(events, bundles)
    events.to_csv(out / "external_candidate_events.csv", index=False); preds.to_csv(out / "external_predictions.csv", index=False); ablation.to_csv(out / "external_strategy_ablation.csv", index=False)
    keys = ["strategy", "symbol", "signal_time", "entry_time", "exit_time"]
    scored = events.merge(preds[keys + ["external_score", "external_selected", "frozen_threshold"]], on=keys, how="inner", validate="one_to_one").sort_values(["entry_time", "strategy", "symbol"]).reset_index(drop=True)
    selected = scored.external_selected.to_numpy(bool)
    scored.to_csv(out / "external_scored_events_full.csv", index=False)

    ci = paired_block_uplift_ci(scored, selected, V24Contract())
    stress = cost_stress_table(scored, selected, V24Contract()); stress.to_csv(out / "cost_stress_24_36_60bps.csv", index=False)
    breadth = breadth_uplift(scored, selected, group_col="symbol", risk_per_trade=0.0025)
    posfrac = _positive_symbol_fraction(scored, selected)
    c = MTMRiskContract()
    bsum, bled, bcurve = simulate_mtm_portfolio(scored, external, contract=c, mode="okx_external_base")
    fsum, fled, fcurve = simulate_mtm_portfolio(scored, external, selected=selected, contract=c, mode="okx_external_frozen_ml")
    bled.to_csv(out / "mtm_base_ledger.csv", index=False); bcurve.to_csv(out / "mtm_base_curve.csv", index=False)
    fled.to_csv(out / "mtm_filtered_ledger.csv", index=False); fcurve.to_csv(out / "mtm_filtered_curve.csv", index=False)
    _dump(out / "mtm_base_summary.json", bsum); _dump(out / "mtm_filtered_summary.json", fsum)

    stress36 = stress[np.isclose(stress.roundtrip_bps, 36.0)].iloc[0]
    economic_pass = bool(
        fsum["accepted"] >= 100 and np.isfinite(fsum["profit_factor"]) and fsum["profit_factor"] >= 1.05
        and np.isfinite(fsum["mean_r_accepted"]) and fsum["mean_r_accepted"] > 0
        and abs(fsum["max_intrabar_stress_drawdown"]) <= 0.05 and ci[0] > 0
        and np.isfinite(stress36["filtered_profit_factor"]) and stress36["filtered_profit_factor"] >= 1.0
        and stress36["filtered_mean_r"] > 0
    )
    d = {
        "version": "v0.24c", "plan": "PLAN_A_TECHNICAL_FALLBACK_OKX", "state": "PASS" if economic_pass else "FAIL",
        "scientific_label": "EXTERNAL_REPLICATION_EVIDENCE_PRE_CPCV" if economic_pass else "EXTERNAL_REPLICATION_FAILED",
        "source_fallback_reason": "PRIMARY_BYBIT_RETURNED_HTTP_403_FOR_ALL_PRE_REGISTERED_SYMBOLS",
        "venue": "OKX public spot via CCXT", "venue_note": "OKX was used in earlier project research, but not to tune v0.24b family meta models; this is not claimed as globally unseen strategy evidence.",
        "calendar_overlap_with_internal": True, "symbols_disjoint_from_v24b_12_symbol_panel": True,
        "usable_symbols": sorted(external), "external_events": int(len(scored)), "external_selected": int(selected.sum()),
        "paired_block_uplift_ci": {"low": float(ci[0]), "high": float(ci[1])}, "symbol_uplift_breadth": breadth,
        "positive_symbol_fraction": posfrac, "mtm_base": bsum, "mtm_filtered": fsum,
        "cost_36bps_filtered_pf": float(stress36["filtered_profit_factor"]), "cost_36bps_filtered_mean_r": float(stress36["filtered_mean_r"]),
        "pbo_dsr_status": "PENDING_CPCV_MULTIPLE_TESTING_AUDIT", "model_refit_on_external": False, "threshold_tuned_on_external": False,
        "next_plan": "PLAN_B_FUTURE_TIME_FORWARD" if economic_pass else "PLAN_D_RESEARCH_REDESIGN",
        "forward_paper_authorized": False, "paper_replacement_authorized": False, "live_execution_authorized": False,
    }
    d["decision_sha256"] = sha256(json.dumps(d, sort_keys=True, default=str, allow_nan=True).encode()).hexdigest()
    _dump(out / "decision.json", d); print(json.dumps(d, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
