from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
    summarize_trade_window,
)
from research_bot.integrity_v19 import finalize_payload_hash
from research_bot.phase_q_v20 import phase_q_protocol_metadata
from research_bot.venue_adapter_v19 import normalized_orderbook_limit


def _coinex_observation(symbol: str, cfg: V19MicrostructureConfig) -> VenueMicrostructureObservation:
    # Freeze the book clock first, then summarize only trades at-or-before that clock.
    # CoinEx's public deals endpoint is recent-trade based, so we fetch extra history
    # and enforce the same fixed time window locally.
    depth = fetch_coinex_depth(symbol, limit=cfg.depth_levels)
    observed_at = depth.timestamp.isoformat()
    trades = fetch_coinex_market_deals(
        symbol,
        market_type="spot",
        pages=2,
        limit=cfg.trades_limit,
    )
    tw = summarize_trade_window(
        trades,
        observed_at=observed_at,
        window_seconds=cfg.trade_window_seconds,
    )
    return VenueMicrostructureObservation(
        venue="coinex",
        symbol=symbol,
        observed_at=observed_at,
        best_bid=depth.best_bid,
        best_ask=depth.best_ask,
        bid_depth_notional=depth.bid_depth_notional,
        ask_depth_notional=depth.ask_depth_notional,
        trade_buy_notional=tw["trade_buy_notional"],
        trade_sell_notional=tw["trade_sell_notional"],
        trade_unknown_notional=tw["trade_unknown_notional"],
        trade_count=tw["trade_count"],
        source="coinex_public_depth_and_deals_fixed_window",
        trade_window_start=tw["trade_window_start"],
        trade_window_end=tw["trade_window_end"],
        trade_first_at=tw["trade_first_at"],
        trade_last_at=tw["trade_last_at"],
        trade_window_seconds=tw["trade_window_seconds"],
        raw_trade_count=tw["raw_trade_count"],
        future_trade_count_excluded=tw["future_trade_count_excluded"],
        trade_staleness_seconds=tw["trade_staleness_seconds"],
        trade_side_semantics=tw["trade_side_semantics"],
    )


def _ccxt_observations(
    exchange_id: str,
    symbols: tuple[str, ...],
    cfg: V19MicrostructureConfig,
) -> tuple[list[VenueMicrostructureObservation], list[dict]]:
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
                ts_ms = int(book.get("timestamp") or exchange.milliseconds())
                observed_at = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()

                # CCXT documents that omitting `since` makes the returned public-trade
                # range exchange-specific. Always ask for the same point-in-time window,
                # then filter again locally to exclude provider overshoot/future trades.
                since_ms = ts_ms - int(cfg.trade_window_seconds * 1000)
                raw_trades = exchange.fetch_trades(
                    symbol,
                    since=since_ms,
                    limit=cfg.trades_limit,
                )
                trades = pd.DataFrame([
                    {
                        "timestamp": x.get("timestamp"),
                        "side": x.get("side"),
                        "price": x.get("price"),
                        "amount": x.get("amount"),
                        "notional": (
                            float(x.get("cost"))
                            if x.get("cost") is not None
                            else float(x.get("price") or 0.0) * float(x.get("amount") or 0.0)
                        ),
                    }
                    for x in raw_trades
                ])
                observations.append(observation_from_orderbook_and_trades(
                    venue=exchange_id,
                    symbol=symbol,
                    observed_at=observed_at,
                    bids=book.get("bids") or [],
                    asks=book.get("asks") or [],
                    trades=trades,
                    source=f"{exchange_id}_public_ccxt_orderbook_trades_fixed_window",
                    levels=cfg.depth_levels,
                    trade_window_seconds=cfg.trade_window_seconds,
                ))
            except Exception as exc:
                failures.append({
                    "venue": exchange_id,
                    "symbol": symbol,
                    "reason": f"FETCH_ERROR:{type(exc).__name__}:{exc}",
                })
    finally:
        exchange.close()
    return observations, failures


def _ci_provenance() -> dict:
    return {
        "event_name": os.getenv("GITHUB_EVENT_NAME", "local"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "workflow": os.getenv("GITHUB_WORKFLOW"),
        "ref": os.getenv("GITHUB_REF"),
        "sha": os.getenv("GITHUB_SHA"),
        "repository": os.getenv("GITHUB_REPOSITORY"),
        "actor": os.getenv("GITHUB_ACTOR"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/v19/forward_microstructure_snapshot.json")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    p.add_argument("--venues", default="coinex,okx,kucoin")
    p.add_argument("--trade-window-seconds", type=int, default=60)
    args = p.parse_args()

    symbols = tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip())
    venues = tuple(x.strip().lower() for x in args.venues.split(",") if x.strip())
    cfg = V19MicrostructureConfig(
        symbols=symbols,
        venues=venues,
        trade_window_seconds=int(args.trade_window_seconds),
    )

    observations: list[VenueMicrostructureObservation] = []
    failures: list[dict] = []
    for venue in venues:
        if venue == "coinex":
            for symbol in symbols:
                try:
                    observations.append(_coinex_observation(symbol, cfg))
                except Exception as exc:
                    failures.append({
                        "venue": venue,
                        "symbol": symbol,
                        "reason": f"FETCH_ERROR:{type(exc).__name__}:{exc}",
                    })
        else:
            rows, errs = _ccxt_observations(venue, symbols, cfg)
            observations.extend(rows)
            failures.extend(errs)

    snapshot = build_snapshot(observations, cfg)
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    snapshot["provider_failures"] = failures
    snapshot["raw_observation_count"] = len(observations)
    snapshot["collection_rule"] = "prospective_only_fixed_window_no_backfill_no_signal"
    snapshot["phase_q_protocol"] = phase_q_protocol_metadata()
    snapshot["ci_provenance"] = _ci_provenance()
    finalize_payload_hash(snapshot, "snapshot_sha256")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "authorized_symbol_count": snapshot["authorized_symbol_count"],
        "raw_observation_count": snapshot["raw_observation_count"],
        "failures": len(failures),
        "trade_window_seconds": cfg.trade_window_seconds,
        "phase_q_protocol_version": snapshot["phase_q_protocol"]["protocol_version"],
        "ci_event": snapshot["ci_provenance"]["event_name"],
        "ci_ref": snapshot["ci_provenance"]["ref"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
