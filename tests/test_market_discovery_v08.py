from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research_bot.market_discovery import discover_multi_exchange, normalize_market_listing


def test_market_normalization_preserves_provenance_and_microstructure():
    listing = normalize_market_listing(
        exchange_id="fixture",
        market={
            "symbol": "BTC/USDT",
            "base": "BTC",
            "quote": "USDT",
            "spot": True,
            "active": True,
        },
        ticker={
            "bid": 99_990,
            "ask": 100_010,
            "quoteVolume": 123_000_000,
        },
        observed_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )
    assert listing.market_type == "spot"
    assert listing.volume_24h_quote == pytest.approx(123_000_000)
    assert listing.spread_bps == pytest.approx(2.0, rel=1e-3)
    assert listing.history_bars is None
    assert "ccxt:fixture" in listing.provenance


def test_unknown_exchange_is_isolated_not_raised_from_batch():
    batch = discover_multi_exchange(["definitely_not_a_real_ccxt_exchange"], retries=0)
    assert batch.listings == ()
    assert len(batch.failures) == 1
    assert batch.failures[0].stage == "initialization"
