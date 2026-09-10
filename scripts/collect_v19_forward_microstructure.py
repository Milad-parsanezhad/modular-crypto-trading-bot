from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import ccxt
import pandas as pd

from research_bot.coinex_depth import fetch_coinex_depth
from research_bot.coinex_public import fetch_coinex_market_deals
from research_bot.forward_microstructure_v19 import (
    V19MicrostructureConfig,
    VenueMicrostructureObservation,
    build_snapshot,
    observation_from_orderbook_and_trades,
)
from research_bot.venue_adapter_v19 import normalized_orderbook_limit


def _trade_totals(trades: pd.DataFrame) -> tuple[float, float, float, int]:
    buy = sell = unknown = 0.0
    count = 0
    if trades is None or trades.empty:
        return buy, sell, unknown, count
    x = trades.copy()
    if "notional" not in x:
        x["notional"] = pd.to_numeric(x["price"], errors="coerce") * pd.to_numeric(x["amount"], errors="coerce")
    for row in x.itertuples(index=False):
        notion = float(getattr(row, "notional", 0.0) or 0.0)
        if notion < 0:
            continue
        side = str(getattr(row, "side", "") or "").lower()
        count += 1
        if side == "buy":
            buy += notion
        elif side == "sell":
            sell += notion
        else:
            unknown += notion
    return buy, sell, unknown, count


def _coinex_observation(symbol: str, cfg: V19MicrostructureConfig) -> VenueMicrostructureObservation:
    depth = fetch_coinex_depth(symbol, limit=cfg.depth_levels)
    trades = fetch_coinex_market_deals(symbol, market_type="spot", pages=1, limit=cfg.trades_limit)
    buy, sell, unknown, count = _trade_totals(trades)
    return VenueMicrostructureObservation(
        venue="coinex",
        symbol=symbol,
        observed_at=depth.timestamp.isoformat(),
        best_bid=depth.best_bid,
        best_ask=depth.best_ask,
        bid_depth_notional=depth.bid_depth_notional,
        ask_depth_notional=depth.ask_depth_notional,
        trade_buy_notional=buy,
        trade_sell_notional=sell,
        trade_unknown_notional=unknown,
        trade_count=count,
        source="coinex_public_depth_and_deals",
    )


def _ccxt_observations(exchange_id: str, symbols: tuple[str, ...], cfg: V19MicrostructureConfig) -> tuple[list[VenueMicrostructureObservation], list[dict]]:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    observations: list[VenueMicrostructureObservation] = []
    failures: list[dict] = []
    try:
        exchange.load_markets()
        for symbol in symbols:
            if symbol not in exchange.markets:
                failures.append({"venue": exchange_id, "symbol": symbol, "reason": "SYMBOL_NOT_LISTED"})
                continue
            try:
                limit = normalized_orderbook_limit(exchange_id, cfg.depth_levels)
                book = exchange.fetch_order_book(symbol, limit=limit)
                raw_trades = exchange.fetch_trades(symbol, limit=cfg.trades_limit)
                trades = pd.DataFrame([
                    {
                        "timestamp": x.get("timestamp"),
                        "side": x.get("side"),
                        "price": x.get("price"),
                        "amount": x.get("amount"),
                        "notional": (float(x.get("price") or 0.0) * float(x.get("amount") or 0.0)),
                    }
                    for x in raw_trades
                ])
                ts_ms = book.get("timestamp") or exchange.milliseconds()
                observed_at = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
                observations.append(observation_from_orderbook_and_trades(
                    venue=exchange_id,
                    symbol=symbol,
                    observed_at=observed_at,
                    bids=book.get("bids") or [],
                    asks=book.get("asks") or [],
                    trades=trades,
                    source=f"{exchange_id}_public_ccxt_orderbook_trades",
                    levels=cfg.depth_levels,
                ))
            except Exception as exc:
                failures.append({"venue": exchange_id, "symbol": symbol, "reason": f"FETCH_ERROR:{type(exc).__name__}:{exc}"})
    finally:
        exchange.close()
    return observations, failures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/v19/forward_microstructure_snapshot.json")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    p.add_argument("--venues", default="coinex,okx,kucoin")
    args = p.parse_args()

    symbols = tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip())
    venues = tuple(x.strip().lower() for x in args.venues.split(",") if x.strip())
    cfg = V19MicrostructureConfig(symbols=symbols, venues=venues)

    observations: list[VenueMicrostructureObservation] = []
    failures: list[dict] = []
    for venue in venues:
        if venue == "coinex":
            for symbol in symbols:
                try:
                    observations.append(_coinex_observation(symbol, cfg))
                except Exception as exc:
                    failures.append({"venue": venue, "symbol": symbol, "reason": f"FETCH_ERROR:{type(exc).__name__}:{exc}"})
        else:
            rows, errs = _ccxt_observations(venue, symbols, cfg)
            observations.extend(rows)
            failures.extend(errs)

    snapshot = build_snapshot(observations, cfg)
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    snapshot["provider_failures"] = failures
    snapshot["raw_observation_count"] = len(observations)
    snapshot["collection_rule"] = "prospective_only_no_backfill_no_signal"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "authorized_symbol_count": snapshot["authorized_symbol_count"],
        "raw_observation_count": snapshot["raw_observation_count"],
        "failures": len(failures),
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
