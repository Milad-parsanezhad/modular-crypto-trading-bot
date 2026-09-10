import pandas as pd

from research_bot.forward_microstructure_v19 import (
    TRADE_SIDE_SEMANTICS,
    V19MicrostructureConfig,
    aggregate_symbol,
    build_snapshot,
    observation_from_orderbook_and_trades,
    summarize_trade_window,
    validate_observation,
)


OBSERVED_AT = pd.Timestamp("2026-09-10T06:00:00Z")


def _obs(venue: str, symbol: str, bid: float, ask: float, buys: float, sells: float):
    mid = (bid + ask) / 2
    trades = pd.DataFrame([
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=12), "side": "buy", "price": mid, "amount": buys / mid},
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=5), "side": "sell", "price": mid, "amount": sells / mid},
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=2), "side": "buy", "price": mid, "amount": 1.0},
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=1), "side": "sell", "price": mid, "amount": 1.0},
        {"timestamp": OBSERVED_AT, "side": "buy", "price": mid, "amount": 1.0},
    ])
    return observation_from_orderbook_and_trades(
        venue=venue,
        symbol=symbol,
        observed_at=OBSERVED_AT.isoformat(),
        bids=[[bid, 2.0], [bid * 0.999, 3.0]],
        asks=[[ask, 1.5], [ask * 1.001, 2.5]],
        trades=trades,
        source=f"{venue}_public",
        levels=2,
        trade_window_seconds=60,
    )


def test_fixed_trade_window_excludes_old_and_future_rows():
    trades = pd.DataFrame([
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=61), "side": "buy", "price": 100, "amount": 100},
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=30), "side": "buy", "price": 100, "amount": 2},
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=5), "side": "sell", "price": 100, "amount": 1},
        {"timestamp": OBSERVED_AT + pd.Timedelta(milliseconds=1), "side": "buy", "price": 100, "amount": 100},
    ])
    out = summarize_trade_window(trades, observed_at=OBSERVED_AT.isoformat(), window_seconds=60)
    assert out["trade_count"] == 2
    assert out["raw_trade_count"] == 4
    assert out["future_trade_count_excluded"] == 1
    assert out["trade_buy_notional"] == 200
    assert out["trade_sell_notional"] == 100
    assert out["trade_staleness_seconds"] == 5.0
    assert out["trade_side_semantics"] == TRADE_SIDE_SEMANTICS


def test_stale_trade_window_is_rejected():
    cfg = V19MicrostructureConfig(
        min_recent_trade_count=1,
        max_trade_staleness_seconds=10.0,
        trade_window_seconds=60,
    )
    trades = pd.DataFrame([
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=40), "side": "buy", "price": 100, "amount": 1},
    ])
    obs = observation_from_orderbook_and_trades(
        venue="okx",
        symbol="BTC/USDT",
        observed_at=OBSERVED_AT.isoformat(),
        bids=[[100, 1]],
        asks=[[100.1, 1]],
        trades=trades,
        source="okx_public",
        trade_window_seconds=60,
    )
    assert "STALE_RECENT_TRADES" in validate_observation(obs, cfg)


def test_future_trade_is_excluded_not_leaked():
    cfg = V19MicrostructureConfig(min_recent_trade_count=1, max_trade_staleness_seconds=30)
    trades = pd.DataFrame([
        {"timestamp": OBSERVED_AT - pd.Timedelta(seconds=2), "side": "sell", "price": 100, "amount": 1},
        {"timestamp": OBSERVED_AT + pd.Timedelta(seconds=1), "side": "buy", "price": 100, "amount": 1000},
    ])
    obs = observation_from_orderbook_and_trades(
        venue="coinex",
        symbol="BTC/USDT",
        observed_at=OBSERVED_AT.isoformat(),
        bids=[[100, 1]],
        asks=[[100.1, 1]],
        trades=trades,
        source="coinex_public",
        trade_window_seconds=60,
    )
    assert obs.future_trade_count_excluded == 1
    assert obs.trade_imbalance == -1.0
    assert "FUTURE_TRADE_LEAKAGE" not in validate_observation(obs, cfg)


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
    assert out["mean_reported_trade_imbalance"] > 0
    assert out["mean_trade_imbalance"] == out["mean_reported_trade_imbalance"]
    assert out["trade_side_semantics"] == TRADE_SIDE_SEMANTICS
    assert out["trade_window_seconds"] == 60
    assert 0 <= out["trade_sign_agreement"] <= 1


def test_microstructure_fails_closed_on_single_venue():
    cfg = V19MicrostructureConfig(min_venues_per_symbol=2)
    out = aggregate_symbol([_obs("coinex", "ETH/USDT", 200, 200.2, 1000, 1000)], cfg)
    assert out["status"] == "INSUFFICIENT_VENUE_COVERAGE"
    assert out["feature_authorized"] is False


def test_snapshot_never_authorizes_trading_and_declares_measurement_contract():
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
    assert out["measurement_contract"]["primary_trading_horizon"] == "4h"
    assert out["measurement_contract"]["reported_trade_window_seconds"] == 60
    assert out["measurement_contract"]["rest_snapshot_not_event_stream"] is True
    assert len(out["snapshot_sha256"]) == 64
