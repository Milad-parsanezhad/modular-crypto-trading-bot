from __future__ import annotations

import argparse
import json

from research_bot.bitpin_depth import fetch_bitpin_depth
from research_bot.bitpin_public import fetch_bitpin_markets, fetch_bitpin_tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Bitpin public API smoke test")
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()

    markets = fetch_bitpin_markets()
    tickers = fetch_bitpin_tickers()
    depth = fetch_bitpin_depth(args.symbol, limit=10)
    payload = {
        "mode": "READ_ONLY_PUBLIC",
        "markets": int(len(markets)),
        "tickers": int(len(tickers)),
        "symbol": args.symbol,
        "best_bid": depth.best_bid,
        "best_ask": depth.best_ask,
        "spread_bps": depth.spread_bps,
        "quote_timestamp": depth.timestamp.isoformat(),
        "live_orders_enabled": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
