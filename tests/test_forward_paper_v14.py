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
    return pd.DataFrame({"timestamp": ts, "open": base - 0.2, "high": base + 0.8, "low": base - 0.8, "close": base, "volume": np.linspace(1000, 1500, n), "value": np.linspace(100000, 200000, n), "market": "TESTUSDT"})


class FakeMarket:
    def __init__(self, frame, spread_bps=4.0): self.frame = frame; self.spread_bps = spread_bps
    def klines(self, symbol, period, bars): return self.frame.tail(bars).copy()
    def depth(self, symbol):
        px = float(self.frame["close"].iloc[-1]); half = self.spread_bps / 20000.0; bid = px * (1 - half); ask = px * (1 + half)
        return DepthSnapshot(symbol=symbol, timestamp=pd.Timestamp(self.frame["timestamp"].iloc[-1]).to_pydatetime(), best_bid=bid, best_ask=ask, mid=(bid + ask) / 2, spread_bps=self.spread_bps, bid_depth=100.0, ask_depth=100.0, imbalance=0.1)


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
