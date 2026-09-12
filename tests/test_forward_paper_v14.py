from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from research_bot.coinex_depth import DepthSnapshot
from research_bot.forward_paper_v14 import ForwardPaperConfig, ForwardPaperRunner
from research_bot.persistence import MemoryPaperStore, PaperAccount, PaperPosition


def bullish_bars(n=180):
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    base = np.linspace(100.0, 170.0, n) + 0.5 * np.sin(np.arange(n) / 4)
    return pd.DataFrame({"timestamp": ts, "open": base - 0.2, "high": base + 0.8, "low": base - 0.8, "close": base, "volume": np.linspace(1000, 1500, n), "value": np.linspace(100000, 200000, n), "market": "TESTUSDT"})


class FakeMarket:
    def __init__(self, frame, spread_bps=4.0, bid_depth=100.0, ask_depth=100.0):
        self.frame = frame; self.spread_bps = spread_bps
        self.bid_depth = bid_depth; self.ask_depth = ask_depth
    def klines(self, symbol, period, bars): return self.frame.tail(bars).copy()
    def depth(self, symbol):
        px = float(self.frame["close"].iloc[-1]); half = self.spread_bps / 20000.0; bid = px * (1 - half); ask = px * (1 + half)
        return DepthSnapshot(symbol=symbol, timestamp=pd.Timestamp(self.frame["timestamp"].iloc[-1]).to_pydatetime(), best_bid=bid, best_ask=ask, mid=(bid + ask) / 2, spread_bps=self.spread_bps, bid_depth=self.bid_depth, ask_depth=self.ask_depth, imbalance=0.1)


def test_forward_paper_defaults_to_observation_only():
    frame = bullish_bars(); now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(10_000.0)
    runner = ForwardPaperRunner(store, market_client=FakeMarket(frame))
    out = runner.run_cycle(now=now)
    assert out["paper_execution_enabled"] is False
    assert out["results"][0]["status"] == "OBSERVED_ONLY"
    assert store.summary()["fills"] == 0


def test_forward_paper_executes_once_per_closed_bar_and_stays_paper_only():
    frame = bullish_bars(); now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5); store = MemoryPaperStore(10_000.0); cfg = ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8); runner = ForwardPaperRunner(store, config=cfg, market_client=FakeMarket(frame), paper_execution_enabled=True)
    first = runner.run_cycle(now=now)
    assert first["live_execution"] is False
    assert first["results"][0]["status"] == "EXECUTED_PAPER"
    assert first["results"][0]["fill"]["mode"] == "PAPER"
    assert store.summary()["fills"] == 1
    assert store.get_position("BTC/USDT").quantity > 0
    second = runner.run_cycle(now=now)
    assert second["results"][0]["status"] == "ALREADY_OBSERVED_BAR"
    assert store.summary()["fills"] == 1


def test_forward_paper_risk_rejects_wide_spread():
    frame = bullish_bars(); now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5); store = MemoryPaperStore(10_000.0); cfg = ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8, max_spread_bps=35.0); runner = ForwardPaperRunner(store, config=cfg, market_client=FakeMarket(frame, spread_bps=120.0), paper_execution_enabled=True)
    out = runner.run_cycle(now=now)
    assert out["results"][0]["status"] == "RISK_REJECTED"
    assert store.summary()["fills"] == 0


def test_zero_visible_depth_never_creates_a_phantom_fill():
    frame = bullish_bars(); now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(10_000.0)
    runner = ForwardPaperRunner(
        store,
        config=ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8),
        market_client=FakeMarket(frame, ask_depth=0.0),
        paper_execution_enabled=True,
    )
    out = runner.run_cycle(now=now)
    result = out["results"][0]
    assert result["status"] == "NO_LIQUIDITY"
    assert result["fill"]["filled_quantity"] == 0.0
    assert store.get_position("BTC/USDT").quantity == 0.0


def test_risk_reducing_exit_is_not_blocked_by_drawdown_gate():
    frame = bullish_bars().copy()
    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].iloc[::-1].to_numpy()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(8_000.0)
    store.set_account(PaperAccount(cash=8_000.0, equity=8_100.0, peak_equity=10_000.0))
    store.set_position(PaperPosition(symbol="BTC/USDT", quantity=1.0, avg_price=150.0))
    runner = ForwardPaperRunner(
        store,
        config=ForwardPaperConfig(symbols=("BTC/USDT",), entry_rule_score=0.8),
        market_client=FakeMarket(frame),
        paper_execution_enabled=True,
    )
    out = runner.run_cycle(now=now)
    result = out["results"][0]
    assert result["signal"]["action"] == "EXIT"
    assert result["status"] == "EXECUTED_PAPER"
    assert store.get_position("BTC/USDT").quantity == 0.0


def test_large_exit_is_reduced_in_safe_notional_chunks():
    frame = bullish_bars().copy()
    frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].iloc[::-1].to_numpy()
    now = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime() + timedelta(hours=5)
    store = MemoryPaperStore(5_000.0)
    store.set_position(PaperPosition(symbol="BTC/USDT", quantity=50.0, avg_price=150.0))
    cfg = ForwardPaperConfig(symbols=("BTC/USDT",), max_order_notional=2_000.0)
    runner = ForwardPaperRunner(store, config=cfg, market_client=FakeMarket(frame), paper_execution_enabled=True)

    result = runner.run_cycle(now=now)["results"][0]

    assert result["status"] == "EXECUTED_PAPER"
    assert result["fill"]["requested_quantity"] * result["signal"]["reference_price"] <= 2_000.0
    assert 0.0 < store.get_position("BTC/USDT").quantity < 50.0
