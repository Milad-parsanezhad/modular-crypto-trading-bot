from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from research_bot.coinex_depth import DepthSnapshot
from research_bot.contracts import ExecutionMode
from research_bot.execution import ExecutionFill, OrderSide
from research_bot.execution_hardening_v52 import (
    DuplicateSettlementError,
    StaleMarketDataError,
    commit_fill_atomic,
    executable_reference_price,
    settle_fill,
    validate_executable_quote,
)
from research_bot.persistence import MemoryPaperStore, PaperAccount, PaperPosition


def _depth(ts, bid=99.0, ask=101.0):
    return DepthSnapshot(
        symbol="BTC/USDT",
        timestamp=ts,
        best_bid=bid,
        best_ask=ask,
        mid=(bid + ask) / 2,
        spread_bps=(ask - bid) / ((ask + bid) / 2) * 10_000,
        bid_depth=10.0,
        ask_depth=10.0,
        imbalance=0.0,
    )


def _fill(*, side=OrderSide.BUY, qty=1.0, px=100.0, fee=0.1, cid="order-1"):
    return ExecutionFill(
        client_order_id=cid,
        symbol="BTC/USDT",
        side=side,
        requested_quantity=qty,
        filled_quantity=qty,
        fill_price=px,
        fee_paid=fee,
        slippage_paid=0.0,
        mode=ExecutionMode.PAPER,
        status="FILLED",
        timestamp=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    )


def test_fresh_quote_guard_rejects_old_and_future_quotes():
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
    assert validate_executable_quote(_depth(now - timedelta(seconds=5)), now=now, max_age_seconds=30) == 5.0
    with pytest.raises(StaleMarketDataError, match="STALE_QUOTE"):
        validate_executable_quote(_depth(now - timedelta(seconds=31)), now=now, max_age_seconds=30)
    with pytest.raises(StaleMarketDataError, match="QUOTE_TIMESTAMP_IN_FUTURE"):
        validate_executable_quote(_depth(now + timedelta(seconds=6)), now=now, max_future_skew_seconds=5)


def test_executable_reference_uses_ask_for_buy_and_bid_for_sell():
    d = _depth(datetime.now(timezone.utc), bid=98.5, ask=101.5)
    assert executable_reference_price(d, OrderSide.BUY) == 101.5
    assert executable_reference_price(d, OrderSide.SELL) == 98.5


def test_settlement_conserves_cash_and_position():
    account = PaperAccount(1000.0, 1000.0, 1000.0)
    pos = PaperPosition("BTC/USDT", 0.0, 0.0)
    a1, p1 = settle_fill(account, pos, _fill(qty=2.0, px=100.0, fee=0.2))
    assert abs(a1.cash - 799.8) < 1e-12
    assert p1.quantity == 2.0
    assert p1.avg_price == 100.0
    a2, p2 = settle_fill(a1, p1, _fill(side=OrderSide.SELL, qty=0.5, px=110.0, fee=0.055, cid="order-2"))
    assert abs(a2.cash - (799.8 + 55.0 - 0.055)) < 1e-12
    assert p2.quantity == 1.5


def test_duplicate_settlement_is_idempotent_at_store_boundary():
    store = MemoryPaperStore(1000.0)
    fill = _fill(qty=1.0, px=100.0, fee=0.1)
    commit_fill_atomic(store, symbol="BTC/USDT", fill=fill, strategy_version="TEST", metadata={})
    cash_after = store.get_account().cash
    qty_after = store.get_position("BTC/USDT").quantity
    with pytest.raises(DuplicateSettlementError):
        commit_fill_atomic(store, symbol="BTC/USDT", fill=fill, strategy_version="TEST", metadata={})
    assert store.get_account().cash == cash_after
    assert store.get_position("BTC/USDT").quantity == qty_after
    assert store.summary()["fills"] == 1


class ExplodingFillStore(MemoryPaperStore):
    def record_fill(self, row: dict) -> None:
        raise RuntimeError("simulated crash before ledger append")


def test_memory_atomic_settlement_rolls_back_on_ledger_failure():
    store = ExplodingFillStore(1000.0)
    before = store.get_account()
    with pytest.raises(RuntimeError, match="simulated crash"):
        commit_fill_atomic(
            store,
            symbol="BTC/USDT",
            fill=_fill(qty=1.0, px=100.0, fee=0.1),
            strategy_version="TEST",
            metadata={},
        )
    assert store.get_account() == before
    assert store.get_position("BTC/USDT").quantity == 0.0
    assert store.summary()["fills"] == 0
