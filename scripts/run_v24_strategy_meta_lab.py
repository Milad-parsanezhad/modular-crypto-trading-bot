from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ml_framework_v23r import (
    SplitContract,
    assert_temporal_separation,
    chronological_purged_split,
    dataframe_sha256,
    select_feature_columns,
)
from research_bot.strategy_meta_v24 import (
    V24Contract,
    build_strategy_event_panel,
    economic_metrics,
    fit_meta_models,
    target_specs,
)

PERIOD = "4hour"
DEFAULT_SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LINK/USDT", "LTC/USDT", "BCH/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
)


def fetch_closed(symbol: str, bars: int) -> pd.DataFrame:
    x = fetch_coinex_klines(symbol=symbol, period=PERIOD, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    closed = x["timestamp"] + pd.Timedelta(milliseconds=PERIOD_MS[PERIOD]) <= pd.Timestamp.now(tz="UTC")
    return x[closed].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((f for s, f in frames.items() if s != symbol), None)


def audit_strategy_dataset(df: pd.DataFrame) -> dict:
    required = {
        "signal_time", "entry_time", "exit_time", "label_end_time", "strategy", "symbol",
        "side", "r_multiple", "label_meta_execute", "segment",
    }
    missing = sorted(required - set(df.columns))
    fatal = []
    if missing:
        fatal.append(f"MISSING_REQUIRED:{','.join(missing)}")
    duplicate_fraction = float(df.duplicated(["strategy", "symbol", "signal_time"]).mean()) if len(df) else 0.0
    if duplicate_fraction > 0:
        fatal.append("DUPLICATE_STRATEGY_SYMBOL_SIGNAL")
    times_ok = True
    if not missing:
        s = pd.to_datetime(df["signal_time"], utc=True)
        e = pd.to_datetime(df["entry_time"], utc=True)
        x = pd.to_datetime(df["exit_time"], utc=True)
        times_ok = bool(((s < e) & (e <= x)).all())
        if not times_ok:
            fatal.append("EVENT_TIME_ORDER")
    try:
        split_map = {k: df[df.segment == k].sort_values("signal_time").copy() for k in ("development", "validation", "test")}
        assert_temporal_separation(split_map, timestamp_col="signal_time")
        split_ok = True
    except Exception as exc:
        split_ok = False
        fatal.append(f"TEMPORAL_SPLIT:{type(exc).__name__}")
    try:
        numeric, categorical = select_feature_columns(df, include_context=True, include_identity=False)
        feature_ok = True
    except Exception as exc:
        numeric, categorical = [], []
        feature_ok = False
        fatal.append(f"FEATURE_WHITELIST:{type(exc).__name__}")
    label_consistency = bool((df["label_meta_execute"].astype(int) == (pd.to_numeric(df["r_multiple"], errors="coerce") > 0).astype(int)).all()) if len(df) else False
    if not label_consistency:
        fatal.append("META_LABEL_INCONSISTENT")
    positive = float(df["label_meta_execute"].mean()) if len(df) else np.nan
    if not (np.isfinite(positive) and 0.03 <= positive <= 0.97):
        fatal.append("META_LABEL_DEGENERATE")
    worst_missing = max((float(df[c].isna().mean()) for c in numeric), default=0.0)
    if worst_missing > 0.40:
        fatal.append("FEATURE_MISSINGNESS")
    return {
        "decision": "AUDIT_PASS" if not fatal else "AUDIT_FAIL",
        "rows": int(len(df)),
        "strategies": int(df["strategy"].nunique()) if "strategy" in df else 0,
        "symbols": int(df["symbol"].nunique()) if "symbol" in df else 0,
        "positive_fraction": positive,
        "duplicate_event_fraction": duplicate_fraction,
        "event_time_order_pass": times_ok,
        "temporal_split_pass": split_ok,
        "feature_whitelist_pass": feature_ok,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "worst_feature_missing_fraction": worst_missing,
        "meta_label_consistency_pass": label_consistency,
        "dataframe_sha256": dataframe_sha256(df),
        "fatal_codes": fatal,
    }


def write_strategy_ablation(test: pd.DataFrame, predictions: pd.DataFrame, risk: float, output: Path) -> pd.DataFrame:
    selection = predictions.set_index(["strategy", "symbol", "signal_time"])["meta_selected"]
    rows = []
    for strategy, group in test.groupby("strategy", sort=True):
        keys = pd.MultiIndex.from_frame(group[["strategy", "symbol", "signal_time"]])
        mask = selection.reindex(keys).fillna(False).to_numpy(dtype=bool)
        base = economic_metrics(group, risk_per_trade=risk)
        filt = economic_metrics(group, mask, risk_per_trade=risk)
        rows.append({
            "strategy": strategy,
            **{f"base_{k}": v for k, v in base.items()},
            **{f"filtered_{k}": v for k, v in filt.items()},
            "uplift_total_return": float(filt["total_return"]) - float(base["total_return"]),
            "uplift_mean_r": float(filt["mean_r"]) - float(base["mean_r"]) if np.isfinite(float(filt["mean_r"])) else np.nan,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "strategy_ablation_test.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.24 strategy-aware meta-labeling research lab")
    ap.add_argument("--output-dir", default="artifacts/v24-strategy-meta")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--bars", type=int, default=3600)
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = V24Contract()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    specs = target_specs()
    (output / "contract.json").write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")

    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for symbol in symbols:
        try:
            frame = fetch_closed(symbol, args.bars)
            if len(frame) < 800:
                raise RuntimeError(f"insufficient closed bars={len(frame)}")
            frames[symbol] = frame
            provenance[symbol] = {
                "rows": int(len(frame)), "first": str(frame.timestamp.min()), "last": str(frame.timestamp.max()),
                "source": "CoinEx public spot OHLCV", "period": PERIOD,
            }
        except Exception as exc:
            provenance[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    if len(frames) < 5:
        raise RuntimeError(f"v0.24 requires >=5 successful symbols, got {len(frames)}")

    parts: list[pd.DataFrame] = []
    coverage = []
    for spec in specs:
        for symbol, frame in frames.items():
            needs_peer = "CORRELATION" in spec.name or spec.family == "correlation_divergence"
            peer = peer_for(symbol, frames) if needs_peer else None
            panel = build_strategy_event_panel(spec, frame, symbol, peer=peer, contract=contract)
            if not panel.empty:
                parts.append(panel)
            coverage.append({
                "strategy": spec.name, "symbol": symbol, "events": int(len(panel)),
                "positive_fraction": float(panel["label_meta_execute"].mean()) if not panel.empty else np.nan,
            })
            print(json.dumps(coverage[-1], default=str))
    if not parts:
        raise RuntimeError("no v0.24 strategy events generated")

    raw = pd.concat(parts, ignore_index=True)
    raw["signal_time"] = pd.to_datetime(raw["signal_time"], utc=True)
    raw["label_end_time"] = pd.to_datetime(raw["label_end_time"], utc=True)
    raw = raw.sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)

    split_contract = SplitContract(development_fraction=0.60, validation_fraction=0.20, test_fraction=0.20, embargo_rows=2)
    split = chronological_purged_split(
        raw,
        timestamp_col="signal_time",
        label_end_time_col="label_end_time",
        contract=split_contract,
    )
    tagged = []
    for name, part in split.items():
        z = part.copy()
        z["segment"] = name
        tagged.append(z)
    dataset = pd.concat(tagged, ignore_index=True).sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)
    dataset.to_csv(output / "strategy_event_dataset.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "strategy_symbol_coverage.csv", index=False)
    (output / "data_provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    audit = audit_strategy_dataset(dataset)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    if audit["decision"] != "AUDIT_PASS":
        raise RuntimeError(f"v0.24 dataset audit failed: {audit['fatal_codes']}")

    split_map = {k: dataset[dataset.segment == k].copy().reset_index(drop=True) for k in ("development", "validation", "test")}
    board, decision, champion, predictions, stress = fit_meta_models(
        split_map["development"], split_map["validation"], split_map["test"], contract=contract,
    )
    board.to_csv(output / "validation_model_leaderboard.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    stress.to_csv(output / "cost_stress_test.csv", index=False)
    joblib.dump(champion, output / "validation_frozen_champion.joblib", compress=3)

    ablation = write_strategy_ablation(split_map["test"], predictions, contract.risk_per_trade, output)
    champion_manifest = {
        "family": decision["champion_family"],
        "seed": decision["champion_seed"],
        "threshold": decision["frozen_threshold"],
        "fit_segment": decision["fit_segment"],
        "selection_segment": decision["selection_segment"],
        "test_used_for_selection": decision["test_used_for_selection"],
        "numeric_features": decision["numeric_features"],
        "categorical_features": decision["categorical_features"],
        "dataset_sha256": audit["dataframe_sha256"],
        "live_execution_authorized": False,
    }
    manifest_raw = json.dumps(champion_manifest, sort_keys=True, default=str).encode("utf-8")
    champion_manifest["manifest_sha256"] = sha256(manifest_raw).hexdigest()
    (output / "champion_manifest.json").write_text(json.dumps(champion_manifest, indent=2, default=str), encoding="utf-8")

    final = {
        "version": "v0.24",
        "stage": "STRATEGY_AWARE_META_LABELING_AND_INCREMENTAL_ECONOMIC_VALUE",
        "dataset_audit": audit,
        "strategies_requested": [s.name for s in specs],
        "successful_symbols": sorted(frames),
        "development_events": int(len(split_map["development"])),
        "validation_events": int(len(split_map["validation"])),
        "test_events": int(len(split_map["test"])),
        "meta_model": decision,
        "strategy_ablation_positive_uplift_fraction": float((ablation["uplift_total_return"] > 0).mean()) if len(ablation) else 0.0,
        "evidence_contract": {
            "base_vs_ml_filter": True,
            "triple_barrier_equivalent": "stop + target + max_hold vertical barrier; next-open entry; same-bar collision stop-first",
            "cusum_event_feature": True,
            "purged_panel_split": True,
            "validation_only_selection": True,
            "untouched_test_once": True,
            "cost_stress_bps": [contract.roundtrip_bps, 36.0, 60.0],
            "validation_pbo_diagnostic": True,
            "deflated_sharpe_diagnostic": True,
        },
        "external_replication_required_if_candidate": True,
        "full_cpcv_retraining_required_if_candidate": True,
        "portfolio_risk_backtest_required_if_candidate": True,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    (output / "decision.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
