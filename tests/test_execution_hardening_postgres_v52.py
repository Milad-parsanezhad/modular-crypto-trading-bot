from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest

from research_bot.contracts import ExecutionMode
from research_bot.execution import ExecutionFill, OrderSide
from research_bot.execution_hardening_v52 import DuplicateSettlementError, commit_fill_atomic
from research_bot.persistence import PostgresPaperStore


DB_URL = os.getenv("TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not configured")


def _fill(cid="pg-order-1"):
    return ExecutionFill(
        client_order_id=cid,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        requested_quantity=1.0,
        filled_quantity=1.0,
        fill_price=100.0,
        fee_paid=0.1,
        slippage_paid=0.0,
        mode=ExecutionMode.PAPER,
        status="FILLED",
        timestamp=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    )


def _reset(store):
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_equity")
            cur.execute("DELETE FROM paper_fills")
            cur.execute("DELETE FROM paper_observations")
            cur.execute("DELETE FROM paper_positions")
            cur.execute("DELETE FROM paper_account")
    store.ensure_schema()


def test_postgres_settlement_rolls_back_and_is_idempotent():
    store = PostgresPaperStore(DB_URL, initial_cash=1000.0)
    store.ensure_schema()
    _reset(store)

    # Force the final fill INSERT to fail after account/position SQL has already
    # executed. PostgreSQL must roll the entire transaction back.
    with pytest.raises(Exception):
        commit_fill_atomic(
            store,
            symbol="BTC/USDT",
            fill=_fill(),
            strategy_version=None,  # violates paper_fills.strategy_version NOT NULL
            metadata={},
        )
    assert store.get_account().cash == 1000.0
    assert store.get_position("BTC/USDT").quantity == 0.0
    assert store.summary()["fills"] == 0

    commit_fill_atomic(
        store,
        symbol="BTC/USDT",
        fill=_fill(),
        strategy_version="V52_TEST",
        metadata={"paper_only": True},
    )
    cash_after = store.get_account().cash
    qty_after = store.get_position("BTC/USDT").quantity
    assert abs(cash_after - 899.9) < 1e-9
    assert qty_after == 1.0
    assert store.summary()["fills"] == 1

    with pytest.raises(DuplicateSettlementError):
        commit_fill_atomic(
            store,
            symbol="BTC/USDT",
            fill=_fill(),
            strategy_version="V52_TEST",
            metadata={},
        )
    assert store.get_account().cash == cash_after
    assert store.get_position("BTC/USDT").quantity == qty_after
    assert store.summary()["fills"] == 1
