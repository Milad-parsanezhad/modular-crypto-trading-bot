from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research_bot.live_eligibility import LiveEligibilityConfig, run_live_eligibility_pipeline
from research_bot.market_discovery import discover_multi_exchange


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.9 live eligibility audit + fast scan")
    parser.add_argument("--exchanges", default="kucoin,okx,mexc,gate,coinex")
    parser.add_argument("--quotes", default="USDT,USDC,USD")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--history-bars", type=int, default=650)
    parser.add_argument("--max-assets", type=int, default=30)
    parser.add_argument("--min-volume", type=float, default=2_000_000.0)
    parser.add_argument("--max-spread-bps", type=float, default=40.0)
    parser.add_argument("--min-history-bars", type=int, default=500)
    parser.add_argument("--candidate-quantile", type=float, default=0.80)
    parser.add_argument("--timeout-ms", type=int, default=15_000)
    parser.add_argument("--output", default="artifacts/v09/live_eligibility_scan.json")
    args = parser.parse_args()

    exchanges = tuple(x.strip() for x in args.exchanges.split(",") if x.strip())
    quotes = tuple(x.strip().upper() for x in args.quotes.split(",") if x.strip())
    discovery = discover_multi_exchange(
        exchanges,
        allowed_quotes=quotes,
        timeout_ms=args.timeout_ms,
        retries=1,
    )
    cfg = LiveEligibilityConfig(
        timeframe=args.timeframe,
        history_bars=args.history_bars,
        max_assets=args.max_assets,
        allowed_quotes=quotes,
        min_volume_24h_quote=args.min_volume,
        max_spread_bps=args.max_spread_bps,
        min_history_bars=args.min_history_bars,
        candidate_quantile=args.candidate_quantile,
        timeout_ms=args.timeout_ms,
    )
    report = run_live_eligibility_pipeline(discovery.listings, cfg)
    report["discovery"] = {
        "started_at": discovery.started_at.isoformat(),
        "completed_at": discovery.completed_at.isoformat(),
        "listings": len(discovery.listings),
        "unique_assets": len({x.asset_id for x in discovery.listings}),
        "failures": [f.__dict__ for f in discovery.failures],
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    summary = {
        "research_status": report["research_status"],
        "discovered_listings": report["discovery"]["listings"],
        "unique_assets_discovered": report["discovery"]["unique_assets"],
        "selected_for_audit": report["selected_for_audit"],
        "eligible_assets": report["eligible_assets"],
        "rejected_assets": report["rejected_assets"],
        "deep_analysis_candidates": [
            {
                "symbol": x.get("symbol"),
                "score_percentile": x.get("score_percentile"),
                "trend_state": x.get("trend_state"),
                "triangle_candidate": x.get("triangle_candidate"),
            }
            for x in report["deep_analysis_candidates"]
        ],
        "provider_failures": len(report["provider_failures"]),
        "rejection_counts": report["rejection_counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
