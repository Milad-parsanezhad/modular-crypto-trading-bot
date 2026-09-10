from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import ccxt
import pandas as pd

from research_bot.external_ml_replication_v23r import (
    ExternalReplicationContract,
    evaluate_frozen_external,
    load_frozen_champion,
)
from research_bot.ml_framework_v23r import CostContract, cost_aware_next_open_labels
from scripts.run_v23r_ml_rebuild_audit import TIMEFRAME, build_causal_features

DEFAULT_SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LINK/USDT", "LTC/USDT", "BCH/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
)


def fetch_external_closed(exchange, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    if symbol not in exchange.markets:
        raise RuntimeError("market unavailable")
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    now_ms = int(exchange.milliseconds())
    cursor = now_ms - int((bars + 100) * step_ms)
    rows: list[list[float]] = []
    loops = 0
    while cursor < now_ms and len(rows) < bars + 100 and loops < 100:
        loops += 1
        limit = min(300, bars + 100 - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + step_ms
        if nxt <= cursor:
            break
        cursor = nxt
        if exchange.rateLimit:
            time.sleep(float(exchange.rateLimit) / 1000.0)
    if not rows:
        raise RuntimeError("no OHLCV rows")
    x = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x["timestamp_ms"], unit="ms", utc=True)
    x = x[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").sort_values("timestamp")
    x = x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= pd.Timestamp.now(tz="UTC")]
    return x.tail(bars).reset_index(drop=True)


def assemble_external_symbol(exchange, symbol: str, bars: int, cost: CostContract) -> pd.DataFrame:
    raw = fetch_external_closed(exchange, symbol, TIMEFRAME, bars)
    if len(raw) < 500:
        raise RuntimeError(f"insufficient closed bars={len(raw)}")
    features = build_causal_features(raw)
    labels = cost_aware_next_open_labels(raw[["timestamp", "open"]], cost)
    out = labels[[
        "signal_time", "label_end_time", "label_future_gross_return",
        "label_future_net_return", "label_positive_net", "label_direction_3class",
    ]].merge(features.rename(columns={"timestamp": "signal_time"}), on="signal_time", how="left", validate="one_to_one")
    out["symbol"] = symbol
    out["timeframe"] = TIMEFRAME
    out["side"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-champion external venue replication; no refit or external threshold tuning")
    ap.add_argument("--internal-artifact", required=True)
    ap.add_argument("--output-dir", default="artifacts/v23r-external-ml-replication")
    ap.add_argument("--exchange", default="okx")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--bars", type=int, default=2200)
    ap.add_argument("--min-selected", type=int, default=100)
    ap.add_argument("--min-symbols", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    champion = load_frozen_champion(args.internal_artifact)
    exchange_cls = getattr(ccxt, args.exchange, None)
    if exchange_cls is None:
        raise RuntimeError(f"unsupported ccxt exchange: {args.exchange}")
    exchange = exchange_cls({"enableRateLimit": True})
    parts: list[pd.DataFrame] = []
    provenance: dict[str, dict] = {}
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        exchange.load_markets()
        for symbol in symbols:
            try:
                part = assemble_external_symbol(exchange, symbol, args.bars, CostContract())
                parts.append(part)
                provenance[symbol] = {
                    "rows": int(len(part)),
                    "first": str(part["signal_time"].min()),
                    "last": str(part["signal_time"].max()),
                }
            except Exception as exc:
                provenance[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            close()

    if not parts:
        raise RuntimeError("external venue returned no usable labeled rows")
    dataset = pd.concat(parts, ignore_index=True).sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    dataset.to_csv(out / "external_labeled_dataset.csv", index=False)
    (out / "external_provenance.json").write_text(json.dumps({
        "exchange": args.exchange,
        "market_type": "spot",
        "timeframe": TIMEFRAME,
        "requested_bars_per_symbol": args.bars,
        "symbols_requested": symbols,
        "symbols": provenance,
        "feature_builder": "scripts.run_v23r_ml_rebuild_audit.build_causal_features",
        "cost_contract": "10 bps fee + 2 bps slippage each way (24 bps round trip)",
        "model_refit_on_external": False,
        "threshold_tuned_on_external": False,
    }, indent=2, default=str), encoding="utf-8")

    predictions, decision = evaluate_frozen_external(
        champion,
        dataset,
        contract=ExternalReplicationContract(min_selected=args.min_selected, min_symbols=args.min_symbols),
    )
    predictions.to_csv(out / "external_predictions.csv", index=False)
    decision.update({
        "exchange": args.exchange,
        "market_type": "spot",
        "timeframe": TIMEFRAME,
        "venue_replication_only": True,
        "independent_asset_class_holdout": False,
        "fresh_forward_evidence": False,
    })
    (out / "external_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
