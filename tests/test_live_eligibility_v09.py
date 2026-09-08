from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.live_eligibility import (
    LiveEligibilityConfig,
    audit_ohlcv,
    prefilter_discovery,
    run_live_eligibility_pipeline,
)
from research_bot.universe import MarketListing


def _listing(base: str, volume: float, spread: float | None = 2.0, exchange: str = "kucoin") -> MarketListing:
    return MarketListing(
        exchange=exchange,
        symbol=f"{base}/USDT",
        base=base,
        quote="USDT",
        market_type="spot",
        active=True,
        volume_24h_quote=volume,
        spread_bps=spread,
        provenance="synthetic-test",
    )


def _bars(seed: int, n: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.006, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.006, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 10000, n),
        }
    )


def test_audit_ohlcv_detects_gap_without_inventing_rows():
    bars = _bars(1, 600).drop(index=[100, 101]).reset_index(drop=True)
    audit = audit_ohlcv(bars, "4h")
    assert audit.rows == 598
    assert audit.expected_rows == 600
    assert audit.gap_count >= 1
    assert audit.missing_fraction > 0
    assert audit.abnormal_fraction == 0.0


def test_prefilter_is_dynamic_deduplicated_and_excludes_stable_or_low_liquidity():
    listings = [
        _listing("BTC", 100_000_000, 4.0, "kucoin"),
        _listing("BTC", 200_000_000, 6.0, "okx"),
        _listing("ETH", 80_000_000, 3.0, "mexc"),
        _listing("USDC", 500_000_000, 1.0, "okx"),
        _listing("SMALL", 100_000, 2.0, "mexc"),
    ]
    selected = prefilter_discovery(listings, LiveEligibilityConfig(max_assets=10, min_volume_24h_quote=1_000_000))
    assert {x.asset_id for x in selected} == {"BTC", "ETH"}
    assert next(x for x in selected if x.asset_id == "BTC").exchange == "okx"


class FakeAuditor:
    def fetch_bars(self, listing, timeframe, limit):
        seed = sum(ord(c) for c in listing.base)
        return _bars(seed, limit)

    def fetch_spread_bps(self, listing):
        return 2.0 if listing.spread_bps is None else listing.spread_bps


def test_pipeline_stops_at_deep_analysis_candidate_not_entry_signal():
    listings = [
        _listing("BTC", 120_000_000, 2.0),
        _listing("ETH", 100_000_000, None, "coinex"),
        _listing("SOL", 80_000_000, 3.0, "okx"),
        _listing("XRP", 60_000_000, 4.0, "mexc"),
        _listing("ADA", 40_000_000, 5.0, "mexc"),
    ]
    cfg = LiveEligibilityConfig(
        history_bars=600,
        min_history_bars=500,
        max_assets=5,
        min_volume_24h_quote=1_000_000,
        candidate_quantile=0.80,
    )
    report = run_live_eligibility_pipeline(listings, cfg, auditor=FakeAuditor())
    assert report["eligible_assets"] == 5
    assert len(report["scan_rows"]) == 5
    statuses = {x["status"] for x in report["scan_rows"]}
    assert statuses.issubset({"SCANNED", "DEEP_ANALYSIS_CANDIDATE", "REJECT"})
    assert "CONFIRMED_ENTRY" not in statuses
    assert report["research_status"] == "LIVE_ELIGIBILITY_AND_FAST_SCAN_NOT_ENTRY_SIGNAL"
