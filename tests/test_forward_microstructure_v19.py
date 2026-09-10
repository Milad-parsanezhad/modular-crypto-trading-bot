import pandas as pd

from research_bot.forward_microstructure_v19 import (
    V19MicrostructureConfig,
    aggregate_symbol,
    build_snapshot,
    observation_from_orderbook_and_trades,
)


def _obs(venue: str, symbol: str, bid: float, ask: float, buys: float, sells: float):
    trades = pd.DataFrame([
        {"side": "buy", "price": (bid + ask) / 2, "amount": buys / ((bid + ask) / 2)},
        {"side": "sell", "price": (bid + ask) / 2, "amount": sells / ((bid + ask) / 2)},
    ])
    return observation_from_orderbook_and_trades(
        venue=venue,
        symbol=symbol,
        observed_at="2026-09-10T06:00:00+00:00",
        bids=[[bid, 2.0], [bid * 0.999, 3.0]],
        asks=[[ask, 1.5], [ask * 1.001, 2.5]],
        trades=trades,
        source=f"{venue}_public",
        levels=2,
    )


def test_microstructure_aggregates_multiple_venues():
    cfg = V19MicrostructureConfig(min_venues_per_symbol=2)
    rows = [
        _obs("coinex", "BTC/USDT", 100.0, 100.1, 6000, 4000),
        _obs("okx", "BTC/USDT", 100.02, 100.12, 5500, 4500),
    ]
    out = aggregate_symbol(rows, cfg)
    assert out["status"] == "OK"
    assert out["accepted_venues"] == 2
    assert out["feature_authorized"] is True
    assert out["mean_trade_imbalance"] > 0
    assert 0 <= out["trade_sign_agreement"] <= 1


def test_microstructure_fails_closed_on_single_venue():
    cfg = V19MicrostructureConfig(min_venues_per_symbol=2)
    out = aggregate_symbol([_obs("coinex", "ETH/USDT", 200, 200.2, 1000, 1000)], cfg)
    assert out["status"] == "INSUFFICIENT_VENUE_COVERAGE"
    assert out["feature_authorized"] is False


def test_snapshot_never_authorizes_trading():
    cfg = V19MicrostructureConfig(min_venues_per_symbol=2)
    rows = [
        _obs("coinex", "BTC/USDT", 100.0, 100.1, 6000, 4000),
        _obs("okx", "BTC/USDT", 100.02, 100.12, 5500, 4500),
        _obs("coinex", "ETH/USDT", 2000.0, 2001.0, 7000, 3000),
        _obs("okx", "ETH/USDT", 2000.2, 2001.2, 6500, 3500),
    ]
    out = build_snapshot(rows, cfg)
    assert out["authorized_symbol_count"] == 2
    assert out["signal_authorized"] is False
    assert out["paper_strategy_replacement_authorized"] is False
    assert out["live_execution_authorized"] is False
    assert len(out["snapshot_sha256"]) == 64
