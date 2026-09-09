from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_bot.live_eligibility import CCXTLiveAuditor, LiveEligibilityConfig, audit_ohlcv, prefilter_discovery
from research_bot.market_discovery import discover_multi_exchange
from research_bot.oos_tournament_v10 import TournamentConfig
from research_bot.robustness_v11 import RobustnessConfig, run_robustness_v11
from research_bot.universe import EligibilityPolicy, evaluate_listing


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _parse_ints(value: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not out:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v0.11 multi-seed robustness and inference")
    p.add_argument("--exchanges", default="kucoin,okx,mexc,gate,coinex")
    p.add_argument("--quotes", default="USDT,USDC,USD")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--max-assets", type=int, default=20)
    p.add_argument("--history-bars", type=int, default=1500)
    p.add_argument("--min-history-bars", type=int, default=1000)
    p.add_argument("--min-volume", type=float, default=5_000_000.0)
    p.add_argument("--max-spread-bps", type=float, default=40.0)
    p.add_argument("--top-quantile", type=float, default=0.25)
    p.add_argument("--cost-bps", type=float, default=12.0)
    p.add_argument("--label-hurdle-bps", type=float, default=12.0)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-assets-per-timestamp", type=int, default=5)
    p.add_argument("--seeds", type=_parse_ints, default=(11, 23, 42, 77, 101))
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--block-length", type=int, default=12)
    p.add_argument("--fdr-alpha", type=float, default=0.10)
    p.add_argument("--timeout-ms", type=int, default=15_000)
    p.add_argument("--output", default="artifacts/v11/robustness_inference.json")
    args = p.parse_args()

    exchanges = tuple(x.strip() for x in args.exchanges.split(",") if x.strip())
    quotes = tuple(x.strip().upper() for x in args.quotes.split(",") if x.strip())
    discovery = discover_multi_exchange(exchanges, allowed_quotes=quotes, timeout_ms=args.timeout_ms, retries=1)

    live_cfg = LiveEligibilityConfig(
        timeframe=args.timeframe,
        history_bars=args.history_bars,
        max_assets=args.max_assets,
        allowed_quotes=quotes,
        min_volume_24h_quote=args.min_volume,
        max_spread_bps=args.max_spread_bps,
        min_history_bars=args.min_history_bars,
        candidate_quantile=0.80,
        timeout_ms=args.timeout_ms,
    )
    selected = prefilter_discovery(discovery.listings, live_cfg)
    auditor = CCXTLiveAuditor(args.timeout_ms)
    policy = EligibilityPolicy(
        allowed_quotes=quotes,
        allowed_market_types=("spot",),
        min_volume_24h_quote=args.min_volume,
        max_spread_bps=args.max_spread_bps,
        min_history_bars=args.min_history_bars,
        max_missing_fraction=0.01,
        max_abnormal_fraction=0.01,
    )

    symbol_bars = {}
    audits = []
    failures = []
    for listing in selected:
        try:
            bars = auditor.fetch_bars(listing, args.timeframe, args.history_bars)
            hist = audit_ohlcv(bars, args.timeframe)
        except Exception as exc:
            failures.append(
                {
                    "exchange": listing.exchange,
                    "symbol": listing.symbol,
                    "stage": "history",
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                }
            )
            audits.append(
                {
                    "asset_id": listing.asset_id,
                    "exchange": listing.exchange,
                    "symbol": listing.symbol,
                    "eligible": False,
                    "reasons": ["HISTORY_PROVIDER_FAILURE"],
                }
            )
            continue

        try:
            spread = auditor.fetch_spread_bps(listing)
        except Exception as exc:
            spread = listing.spread_bps
            failures.append(
                {
                    "exchange": listing.exchange,
                    "symbol": listing.symbol,
                    "stage": "spread",
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                }
            )

        enriched = replace(
            listing,
            spread_bps=spread,
            history_bars=hist.rows,
            missing_fraction=hist.missing_fraction,
            abnormal_fraction=hist.abnormal_fraction,
            provenance=listing.provenance + f"|v11_history_audit:{datetime.now(timezone.utc).isoformat()}",
        )
        result = evaluate_listing(enriched, policy)
        audits.append(
            {
                "asset_id": enriched.asset_id,
                "exchange": enriched.exchange,
                "symbol": enriched.symbol,
                "volume_24h_quote": enriched.volume_24h_quote,
                "spread_bps": enriched.spread_bps,
                "eligible": result.eligible,
                "reasons": list(result.reasons),
                "history": asdict(hist),
                "provenance": enriched.provenance,
            }
        )
        if result.eligible:
            symbol_bars[enriched.symbol] = bars

    if len(symbol_bars) < args.min_assets_per_timestamp:
        raise RuntimeError(
            f"Only {len(symbol_bars)} eligible assets; need at least {args.min_assets_per_timestamp}"
        )

    tcfg = TournamentConfig(
        timeframe=args.timeframe,
        horizon_bars=1,
        n_folds=args.folds,
        train_fraction=0.55,
        top_quantile=args.top_quantile,
        one_way_cost_bps=args.cost_bps,
        label_hurdle_bps=args.label_hurdle_bps,
        min_assets_per_timestamp=args.min_assets_per_timestamp,
        min_train_timestamps=300,
        min_test_timestamps=60,
        random_state=42,
    )
    rcfg = RobustnessConfig(
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
        alpha=0.05,
        fdr_alpha=args.fdr_alpha,
        min_regime_periods=30,
        positive_seed_fraction_required=0.80,
        max_allowed_drawdown=-0.35,
        random_state=11011,
    )
    robustness = run_robustness_v11(symbol_bars, tcfg, rcfg)

    report = {
        "research_status": "V11_LIVE_ELIGIBLE_MULTI_SEED_ROBUSTNESS_NOT_TRADING_SIGNAL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery": {
            "listings": len(discovery.listings),
            "unique_assets": len({x.asset_id for x in discovery.listings}),
            "failures": [asdict(x) for x in discovery.failures],
        },
        "eligibility": {
            "selected_for_deep_history": len(selected),
            "eligible_assets": len(symbol_bars),
            "eligible_symbols": sorted(symbol_bars),
            "audit_rows": audits,
            "provider_failures": failures,
        },
        "robustness": robustness,
        "decision_contract": (
            "No result in this artifact authorizes paper or live execution. "
            "Any promotion remains subject to the full Great Filter and independent risk engine."
        ),
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    best = robustness.get("provisional_best_model")
    strongest = robustness.get("strongest_baseline")
    key = f"{best}__vs__{strongest}" if best and strongest else None
    strongest_test = robustness.get("pairwise_bootstrap_inference", {}).get(key, {}) if key else {}
    print(
        json.dumps(
            {
                "discovered_listings": report["discovery"]["listings"],
                "eligible_assets": len(symbol_bars),
                "seed_count": robustness["seed_count"],
                "provisional_best_model": best,
                "strongest_baseline": strongest,
                "best_vs_strongest_baseline": strongest_test,
                "promotion_status": robustness["promotion_status"],
                "promotion_reasons": robustness["promotion_reasons"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
