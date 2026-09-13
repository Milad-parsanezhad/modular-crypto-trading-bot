from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from research_bot.coinex_depth import DepthSnapshot
from research_bot.forward_paper_v14 import ForwardPaperConfig, ForwardPaperRunner
from research_bot.persistence import MemoryPaperStore


def bullish_bars(n=180):
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    base = np.linspace(100.0, 170.0, n) + 0.5 * np.sin(np.arange(n) / 4)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base - 0.2,
            "high": base + 0.8,
            "low": base - 0.8,
            "close": base,
            "volume": np.linspace(1000, 1500, n),
            "value": np.linspace(100000, 200000, n),
            "market": "TESTUSDT",
        }
    )


class FakeMarket:
    def __init__(self, frame, *, depth_timestamp, spread_bps=4.0, book_mid=None):
        self.frame = frame
        self.spread_bps = spread_bps
        self.depth_timestamp = depth_timestamp
        self.book_mid = float(book_mid) if book_mid is not None else float(frame["close"].iloc[-1])

    def klines(self, symbol, period, bars):
        return self.frame.tail(bars).copy()

    def depth(self, symbol):
        px = self.book_mid
        half = self.spread_bps / 20000.0
        bid = px * (1 - half)
        ask = px * (1 + half)
        return DepthSnapshot(
            symbol=symbol,
            timestamp=self.depth_timestamp,
            best_bid=bid,
            best_ask=ask,
            mid=(bid + ask) / 2,
            spread_bps=self.spread_bps,
            bid_depth=100.0,
            ask_depth=100.0,
            imbalance=0.1,
        )


def test_forward_paper_executes_once_per_closed_bar_and_stays_paper_only():
    frame = bullish_bars()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(10_000.0)
    cfg = ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8)
    runner = ForwardPaperRunner(
        store,
        config=cfg,
        market_client=FakeMarket(frame, depth_timestamp=now),
        paper_execution_enabled=True,
    )
    first = runner.run_cycle(now=now)
    assert first["live_execution"] is False
    assert first["results"][0]["status"] == "EXECUTED_PAPER"
    assert first["results"][0]["fill"]["mode"] == "PAPER"
    assert store.summary()["fills"] == 1
    assert store.get_position("BTC/USDT").quantity > 0
    second = runner.run_cycle(now=now)
    assert second["results"][0]["status"] in {"ALREADY_OBSERVED_BAR", "ALREADY_SETTLED_ORDER"}
    assert store.summary()["fills"] == 1


def test_forward_paper_risk_rejects_wide_spread():
    frame = bullish_bars()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(10_000.0)
    cfg = ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8, max_spread_bps=35.0)
    runner = ForwardPaperRunner(
        store,
        config=cfg,
        market_client=FakeMarket(frame, depth_timestamp=now, spread_bps=120.0),
        paper_execution_enabled=True,
    )
    out = runner.run_cycle(now=now)
    assert out["results"][0]["status"] == "RISK_REJECTED"
    assert store.summary()["fills"] == 0


def test_forward_paper_rejects_stale_quote_before_execution():
    frame = bullish_bars()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    stale = now - timedelta(minutes=5)
    store = MemoryPaperStore(10_000.0)
    runner = ForwardPaperRunner(
        store,
        config=ForwardPaperConfig(symbols=("BTC/USDT",), max_quote_age_seconds=30.0),
        market_client=FakeMarket(frame, depth_timestamp=stale),
        paper_execution_enabled=True,
    )
    out = runner.run_cycle(now=now)
    assert out["results"][0]["status"] == "STALE_OR_INVALID_QUOTE"
    assert store.summary()["fills"] == 0


def test_fill_uses_live_executable_quote_not_old_candle_close():
    frame = bullish_bars()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    candle_close = float(frame["close"].iloc[-1])
    store = MemoryPaperStore(10_000.0)
    market = FakeMarket(frame, depth_timestamp=now, spread_bps=4.0, book_mid=candle_close * 1.10)
    runner = ForwardPaperRunner(
        store,
        config=ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8),
        market_client=market,
        paper_execution_enabled=True,
    )
    out = runner.run_cycle(now=now)["results"][0]
    assert out["status"] == "EXECUTED_PAPER"
    expected_ask = market.book_mid * (1 + market.spread_bps / 20000.0)
    expected_fill = expected_ask * (1 + runner.config.slippage_bps / 10000.0)
    assert abs(out["fill"]["fill_price"] - expected_fill) < 1e-9
    assert abs(out["fill"]["fill_price"] - candle_close) > candle_close * 0.05
