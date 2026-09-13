from __future__ import annotations

"""Execution-integrity primitives for PAPER research.

This module closes four failure seams without enabling LIVE trading:
1) execution must use a fresh executable quote, never a stale candle close;
2) cash, position and fill-ledger persistence must commit atomically;
3) a repeated client_order_id after restart must be idempotent at the store;
4) restart recovery must distinguish an observed signal from a settled order.
"""

from datetime import datetime, timezone
import json
import math
from typing import Any

from .execution import ExecutionFill, OrderSide
from .persistence import MemoryPaperStore, PaperAccount, PaperPosition, PostgresPaperStore


class StaleMarketDataError(RuntimeError):
    pass


class DuplicateSettlementError(RuntimeError):
    pass


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("market-data timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def validate_executable_quote(
    depth,
    *,
    now: datetime,
    max_age_seconds: float = 30.0,
    max_future_skew_seconds: float = 5.0,
) -> float:
    """Validate top-of-book integrity and return quote age in seconds."""
    now = _utc(now)
    ts = _utc(depth.timestamp)
    bid = float(depth.best_bid)
    ask = float(depth.best_ask)
    mid = float(depth.mid)
    if not all(math.isfinite(x) and x > 0.0 for x in (bid, ask, mid)):
        raise StaleMarketDataError("NONFINITE_OR_NONPOSITIVE_QUOTE")
    if ask < bid:
        raise StaleMarketDataError("CROSSED_BOOK")
    age = float((now - ts).total_seconds())
    if age < -float(max_future_skew_seconds):
        raise StaleMarketDataError("QUOTE_TIMESTAMP_IN_FUTURE")
    if age > float(max_age_seconds):
        raise StaleMarketDataError(f"STALE_QUOTE age_seconds={age:.3f}")
    return max(0.0, age)


def executable_reference_price(depth, side: OrderSide) -> float:
    """Use the executable side of the book, not a candle close."""
    px = float(depth.best_ask if side is OrderSide.BUY else depth.best_bid)
    if not math.isfinite(px) or px <= 0.0:
        raise StaleMarketDataError("INVALID_EXECUTABLE_PRICE")
    return px


def settlement_exists(store, client_order_id: str) -> bool:
    """Persistent idempotency check used after process restart."""
    if isinstance(store, MemoryPaperStore):
        return any(
            str(x.get("client_order_id")) == str(client_order_id)
            for x in store.recent_fills(limit=100_000)
        )
    if isinstance(store, PostgresPaperStore):
        with store._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM paper_fills WHERE client_order_id=%s LIMIT 1",
                    (str(client_order_id),),
                )
                return cur.fetchone() is not None
    raise TypeError("store must support persistent v0.52 settlement lookup")


def settle_fill(
    account: PaperAccount,
    position: PaperPosition,
    fill: ExecutionFill,
) -> tuple[PaperAccount, PaperPosition]:
    q = float(fill.filled_quantity)
    if not math.isfinite(q) or q < 0.0:
        raise ValueError("invalid filled quantity")
    if q == 0.0:
        return account, position
    px = float(fill.fill_price)
    fee = float(fill.fee_paid)
    if not math.isfinite(px) or px <= 0.0 or not math.isfinite(fee) or fee < 0.0:
        raise ValueError("invalid fill economics")

    if fill.side is OrderSide.BUY:
        cost = q * px + fee
        if cost > float(account.cash) + 1e-9:
            raise RuntimeError("PAPER_INSUFFICIENT_CASH")
        new_qty = float(position.quantity + q)
        old_cost = float(position.quantity * position.avg_price)
        avg = (old_cost + q * px) / new_qty if new_qty > 0 else 0.0
        new_position = PaperPosition(position.symbol, new_qty, float(avg))
        cash = float(account.cash - cost)
    else:
        if q > float(position.quantity) + 1e-12:
            raise RuntimeError("PAPER_SELL_EXCEEDS_POSITION")
        proceeds = q * px - fee
        new_qty = max(0.0, float(position.quantity - q))
        new_position = PaperPosition(
            position.symbol,
            new_qty,
            position.avg_price if new_qty > 0 else 0.0,
        )
        cash = float(account.cash + proceeds)

    return (
        PaperAccount(cash=cash, equity=account.equity, peak_equity=account.peak_equity),
        new_position,
    )


def fill_row(
    fill: ExecutionFill,
    *,
    strategy_version: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": fill.timestamp.isoformat(),
        "client_order_id": fill.client_order_id,
        "symbol": fill.symbol,
        "side": fill.side.value,
        "requested_quantity": float(fill.requested_quantity),
        "filled_quantity": float(fill.filled_quantity),
        "fill_price": float(fill.fill_price),
        "fee_paid": float(fill.fee_paid),
        "slippage_paid": float(fill.slippage_paid),
        "status": fill.status,
        "strategy_version": strategy_version,
        "metadata": dict(metadata),
    }


def _memory_commit(
    store: MemoryPaperStore,
    symbol: str,
    fill: ExecutionFill,
    row: dict[str, Any],
) -> PaperAccount:
    if settlement_exists(store, fill.client_order_id):
        raise DuplicateSettlementError(fill.client_order_id)
    account_before = store.get_account()
    position_before = store.get_position(symbol)
    account_after, position_after = settle_fill(account_before, position_before, fill)
    fills_len = len(store._fills)  # type: ignore[attr-defined]
    try:
        store.set_position(position_after)
        store.set_account(account_after)
        store.record_fill(row)
    except Exception:
        store.set_position(position_before)
        store.set_account(account_before)
        del store._fills[fills_len:]  # type: ignore[attr-defined]
        raise
    return account_after


def _postgres_commit(
    store: PostgresPaperStore,
    symbol: str,
    fill: ExecutionFill,
    row: dict[str, Any],
) -> PaperAccount:
    now = datetime.now(timezone.utc)
    with store._connect() as conn:  # one ACID transaction; rollback on exception
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM paper_fills WHERE client_order_id=%s",
                (fill.client_order_id,),
            )
            if cur.fetchone() is not None:
                raise DuplicateSettlementError(fill.client_order_id)

            cur.execute(
                "SELECT cash,equity,peak_equity FROM paper_account WHERE id=1 FOR UPDATE"
            )
            a = cur.fetchone()
            if a is None:
                raise RuntimeError("paper_account row missing")
            account_before = PaperAccount(float(a[0]), float(a[1]), float(a[2]))

            cur.execute(
                "SELECT quantity,avg_price FROM paper_positions WHERE symbol=%s FOR UPDATE",
                (symbol,),
            )
            p = cur.fetchone()
            position_before = (
                PaperPosition(symbol, 0.0, 0.0)
                if p is None
                else PaperPosition(symbol, float(p[0]), float(p[1]))
            )
            account_after, position_after = settle_fill(
                account_before, position_before, fill
            )

            if abs(position_after.quantity) < 1e-15:
                cur.execute("DELETE FROM paper_positions WHERE symbol=%s", (symbol,))
            else:
                cur.execute(
                    """
                    INSERT INTO paper_positions(symbol,quantity,avg_price,updated_at)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (symbol) DO UPDATE SET
                      quantity=EXCLUDED.quantity,avg_price=EXCLUDED.avg_price,
                      updated_at=EXCLUDED.updated_at
                    """,
                    (symbol, position_after.quantity, position_after.avg_price, now),
                )

            cur.execute(
                "UPDATE paper_account SET cash=%s,updated_at=%s WHERE id=1",
                (account_after.cash, now),
            )
            cur.execute(
                """
                INSERT INTO paper_fills(
                  timestamp,client_order_id,symbol,side,requested_quantity,filled_quantity,
                  fill_price,fee_paid,slippage_paid,status,strategy_version,metadata
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                """,
                (
                    store._dt(row["timestamp"]),
                    row["client_order_id"],
                    row["symbol"],
                    row["side"],
                    row["requested_quantity"],
                    row["filled_quantity"],
                    row["fill_price"],
                    row["fee_paid"],
                    row["slippage_paid"],
                    row["status"],
                    row["strategy_version"],
                    json.dumps(row.get("metadata", {})),
                ),
            )
    return account_after


def commit_fill_atomic(
    store,
    *,
    symbol: str,
    fill: ExecutionFill,
    strategy_version: str,
    metadata: dict[str, Any],
) -> PaperAccount:
    """Commit cash + position + fill record as one logical settlement."""
    row = fill_row(fill, strategy_version=strategy_version, metadata=metadata)
    if isinstance(store, MemoryPaperStore):
        return _memory_commit(store, symbol, fill, row)
    if isinstance(store, PostgresPaperStore):
        return _postgres_commit(store, symbol, fill, row)
    raise TypeError("store must support crash-consistent v0.52 settlement")
