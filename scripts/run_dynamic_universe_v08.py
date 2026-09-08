from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from research_bot.market_discovery import discover_multi_exchange


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.8 dynamic multi-exchange public market discovery")
    parser.add_argument(
        "--exchanges",
        default="coinex,kucoin,okx,gate,mexc,bybit,binance",
        help="Comma-separated CCXT exchange IDs. Provider failures are isolated and recorded.",
    )
    parser.add_argument("--quotes", default="USDT,USDC,USD")
    parser.add_argument("--timeout-ms", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", default="artifacts/v08/dynamic_universe_discovery.json")
    args = parser.parse_args()

    exchanges = tuple(x.strip() for x in args.exchanges.split(",") if x.strip())
    quotes = tuple(x.strip().upper() for x in args.quotes.split(",") if x.strip())
    batch = discover_multi_exchange(
        exchanges,
        allowed_quotes=quotes,
        timeout_ms=args.timeout_ms,
        retries=args.retries,
    )

    by_exchange: dict[str, dict] = {}
    grouped = defaultdict(list)
    for listing in batch.listings:
        grouped[listing.exchange].append(listing)
    failure_counts = Counter(f.exchange for f in batch.failures)

    for exchange_id in exchanges:
        rows = grouped.get(exchange_id, [])
        by_exchange[exchange_id] = {
            "discovered_listings": len(rows),
            "spot": sum(1 for x in rows if x.market_type == "spot"),
            "perpetual": sum(1 for x in rows if x.market_type == "perpetual"),
            "with_quote_volume": sum(1 for x in rows if x.volume_24h_quote is not None),
            "with_spread": sum(1 for x in rows if x.spread_bps is not None),
            "provider_failures": failure_counts.get(exchange_id, 0),
        }

    report = {
        "research_status": "DISCOVERY_ONLY_NOT_ELIGIBILITY_VALIDATED",
        "started_at": batch.started_at.isoformat(),
        "completed_at": batch.completed_at.isoformat(),
        "requested_exchanges": exchanges,
        "allowed_quotes": quotes,
        "total_discovered_listings": len(batch.listings),
        "unique_assets": len({x.asset_id for x in batch.listings}),
        "by_exchange": by_exchange,
        "failures": [asdict(f) for f in batch.failures],
        "coverage_warning": (
            "This is exchange-market discovery coverage only. It is NOT market-cap coverage. "
            "Eligibility requires separate history/data-quality audits and a valid market-cap denominator."
        ),
        "listings": [asdict(x) for x in batch.listings],
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "listings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
