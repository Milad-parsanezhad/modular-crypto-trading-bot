import pandas as pd

from research_bot.forward_microstructure_v19 import (
    V19MicrostructureConfig,
    aggregate_symbol,
    observation_from_orderbook_and_trades,
)
from research_bot.integrity_v19 import finalize_payload_hash, payload_sha256
from research_bot.venue_adapter_v19 import normalized_orderbook_limit


def _obs(venue: str, observed_at: str):
    end = pd.Timestamp(observed_at)
    trades = pd.DataFrame([
        {"timestamp": end - pd.Timedelta(seconds=10), "side": "buy", "price": 100.05, "amount": 2.0},
        {"timestamp": end - pd.Timedelta(seconds=8), "side": "sell", "price": 100.04, "amount": 1.0},
        {"timestamp": end - pd.Timedelta(seconds=6), "side": "buy", "price": 100.03, "amount": 1.0},
        {"timestamp": end - pd.Timedelta(seconds=4), "side": "sell", "price": 100.02, "amount": 1.0},
        {"timestamp": end - pd.Timedelta(seconds=2), "side": "buy", "price": 100.01, "amount": 1.0},
    ])
    return observation_from_orderbook_and_trades(
        venue=venue,
        symbol="BTC/USDT",
        observed_at=observed_at,
        bids=[[100.0, 2.0], [99.9, 1.0]],
        asks=[[100.1, 2.0], [100.2, 1.0]],
        trades=trades,
        source=f"{venue}_public",
        levels=2,
        trade_window_seconds=60,
    )


def test_kucoin_orderbook_limit_is_normalized():
    assert normalized_orderbook_limit("kucoin", 10) == 20
    assert normalized_orderbook_limit("kucoin", 20) == 20
    assert normalized_orderbook_limit("kucoin", 50) == 100
    assert normalized_orderbook_limit("okx", 10) == 10


def test_cross_venue_clock_skew_is_quality_gated():
    cfg = V19MicrostructureConfig(min_venues_per_symbol=2, max_venue_clock_skew_seconds=60.0)
    out = aggregate_symbol([
        _obs("coinex", "2026-09-10T06:00:00+00:00"),
        _obs("okx", "2026-09-10T06:02:01+00:00"),
    ], cfg)
    assert out["feature_authorized"] is False
    assert "CROSS_VENUE_CLOCK_SKEW_TOO_LARGE" in out["quality_flags"]
    assert out["venue_clock_skew_seconds"] == 121.0


def test_final_payload_hash_covers_late_metadata():
    payload = {"value": 1, "generated_at": "t1"}
    finalize_payload_hash(payload, "sha256")
    first = payload["sha256"]
    assert first == payload_sha256(payload, exclude_keys=("sha256",))
    payload["generated_at"] = "t2"
    finalize_payload_hash(payload, "sha256")
    assert payload["sha256"] != first
