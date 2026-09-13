from __future__ import annotations

"""Pure reconciliation/state-machine logic for future TESTNET/private adapters.

No network calls or secrets live in this module. It exists so ambiguous submit
outcomes, partial fills, reconnects and venue-reported fees can be tested before
any private exchange execution is enabled.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from typing import Any, Iterable


class OrderLifecycle(str, Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    UNKNOWN_PENDING_RECONCILIATION = "UNKNOWN_PENDING_RECONCILIATION"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


TERMINAL_STATES = {
    OrderLifecycle.FILLED,
    OrderLifecycle.CANCELLED,
    OrderLifecycle.REJECTED,
}


@dataclass(frozen=True)
class ReconciledOrder:
    client_order_id: str
    symbol: str
    side: str
    requested_quantity: float
    state: OrderLifecycle = OrderLifecycle.CREATED
    venue_order_id: str | None = None
    matched_quantity: float = 0.0
    average_price: float | None = None
    fee_reported: float = 0.0
    fee_currency: str | None = None
    updated_at: datetime = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class RealizedCostBreakdown:
    filled_quantity: float
    vwap: float | None
    gross_quote_notional: float
    explicit_fee_reported: float
    explicit_fee_currency: str | None
    explicit_fee_quote: float | None
    implementation_shortfall_quote: float | None
    total_cost_quote: float | None
    complete_for_quote_cost: bool


def deterministic_client_order_id(
    *,
    strategy_version: str,
    symbol: str,
    decision_epoch: int,
    side: str,
    max_length: int = 32,
) -> str:
    """Stable <=32-char identity suitable for Nobitex/CoinEx client IDs."""
    raw = f"{strategy_version}|{symbol}|{int(decision_epoch)}|{side.upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    prefix = "mbot-"
    result = f"{prefix}{digest}"
    if len(result) > max_length:
        result = result[:max_length]
    return result


def mark_submit_started(order: ReconciledOrder, *, now: datetime | None = None) -> ReconciledOrder:
    if order.state is not OrderLifecycle.CREATED:
        raise ValueError("submit may start only from CREATED")
    return replace(order, state=OrderLifecycle.SUBMITTING, updated_at=now or datetime.now(timezone.utc))


def mark_submit_ambiguous(order: ReconciledOrder, *, now: datetime | None = None) -> ReconciledOrder:
    if order.state is not OrderLifecycle.SUBMITTING:
        raise ValueError("ambiguous submit requires SUBMITTING state")
    return replace(
        order,
        state=OrderLifecycle.UNKNOWN_PENDING_RECONCILIATION,
        updated_at=now or datetime.now(timezone.utc),
    )


def retry_submission_allowed(order: ReconciledOrder) -> bool:
    """Fail closed: never blind-retry an ambiguous/private submit."""
    return order.state is OrderLifecycle.CREATED


def _finite_nonnegative(value: Any, name: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"invalid {name}")
    return x


def reconcile_nobitex_order(
    local: ReconciledOrder,
    venue_order: dict[str, Any],
    *,
    now: datetime | None = None,
) -> ReconciledOrder:
    """Reconcile one Nobitex order/status/list object into the local state.

    Nobitex documentation currently describes clientOrderId as experimental.
    The adapter must therefore persist both clientOrderId and numeric order id.
    """
    if not isinstance(venue_order, dict):
        raise ValueError("venue order must be an object")
    venue_client = venue_order.get("clientOrderId")
    if venue_client not in (None, "", local.client_order_id):
        raise ValueError("clientOrderId mismatch")

    venue_id = venue_order.get("id")
    if local.venue_order_id and venue_id is not None and str(venue_id) != str(local.venue_order_id):
        raise ValueError("venue order id mismatch")

    amount = _finite_nonnegative(venue_order.get("amount", local.requested_quantity), "amount")
    matched = _finite_nonnegative(venue_order.get("matchedAmount", 0.0), "matchedAmount")
    if amount <= 0.0 or matched > amount + 1e-12:
        raise ValueError("invalid fill quantities")
    if abs(amount - float(local.requested_quantity)) > max(1e-12, abs(local.requested_quantity) * 1e-9):
        raise ValueError("requested quantity mismatch")

    raw_status = str(venue_order.get("status", "")).strip().lower()
    if raw_status in {"new", "inactive"}:
        state = OrderLifecycle.ACKNOWLEDGED
    elif raw_status == "active":
        state = OrderLifecycle.PARTIAL if matched > 0.0 else OrderLifecycle.ACKNOWLEDGED
    elif raw_status == "done":
        if abs(matched - amount) > max(1e-12, amount * 1e-9):
            raise ValueError("Done order is not fully matched")
        state = OrderLifecycle.FILLED
    elif raw_status == "canceled":
        state = OrderLifecycle.CANCELLED
    elif raw_status in {"rejected", "failed"}:
        state = OrderLifecycle.REJECTED
    else:
        raise ValueError(f"unsupported Nobitex order status: {raw_status!r}")

    avg_raw = venue_order.get("averagePrice")
    avg_price = None
    if avg_raw not in (None, "", "0", 0, 0.0):
        avg_price = _finite_nonnegative(avg_raw, "averagePrice")
        if avg_price <= 0.0:
            avg_price = None
    if matched > 0.0 and avg_price is None:
        # Status/list evidence is insufficient to price the fill; user trades
        # must be queried before economic reconciliation is considered complete.
        avg_price = local.average_price

    fee = _finite_nonnegative(venue_order.get("fee", local.fee_reported), "fee")
    return ReconciledOrder(
        client_order_id=local.client_order_id,
        symbol=local.symbol,
        side=local.side,
        requested_quantity=local.requested_quantity,
        state=state,
        venue_order_id=str(venue_id) if venue_id is not None else local.venue_order_id,
        matched_quantity=matched,
        average_price=avg_price,
        fee_reported=fee,
        # Nobitex Order/Trade examples expose a fee value but the cited object
        # schema does not make fee denomination explicit. Do not guess it.
        fee_currency=local.fee_currency,
        updated_at=now or datetime.now(timezone.utc),
    )


def realized_cost_from_trades(
    trades: Iterable[dict[str, Any]],
    *,
    side: str,
    decision_reference_price: float,
    fee_currency: str | None = None,
    fee_to_quote_rate: float | None = None,
) -> RealizedCostBreakdown:
    """Compute execution shortfall from actual venue trades without guessing fee FX."""
    ref = float(decision_reference_price)
    if not math.isfinite(ref) or ref <= 0.0:
        raise ValueError("invalid decision reference price")
    side_norm = str(side).lower()
    if side_norm not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")

    qty = 0.0
    gross = 0.0
    fee_total = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            raise ValueError("trade must be an object")
        t_qty = _finite_nonnegative(trade.get("amount", 0.0), "trade amount")
        px = _finite_nonnegative(trade.get("price", 0.0), "trade price")
        if t_qty <= 0.0 or px <= 0.0:
            raise ValueError("trade amount/price must be positive")
        qty += t_qty
        gross += t_qty * px
        fee_total += _finite_nonnegative(trade.get("fee", 0.0), "trade fee")

    vwap = gross / qty if qty > 0.0 else None
    if qty <= 0.0:
        return RealizedCostBreakdown(0.0, None, 0.0, fee_total, fee_currency, None, None, None, False)

    direction = 1.0 if side_norm == "buy" else -1.0
    shortfall = direction * (float(vwap) - ref) * qty

    fee_quote = None
    if fee_total == 0.0:
        fee_quote = 0.0
    elif fee_to_quote_rate is not None:
        rate = float(fee_to_quote_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("invalid fee_to_quote_rate")
        fee_quote = fee_total * rate

    complete = fee_quote is not None
    total = shortfall + fee_quote if complete else None
    return RealizedCostBreakdown(
        filled_quantity=qty,
        vwap=float(vwap),
        gross_quote_notional=gross,
        explicit_fee_reported=fee_total,
        explicit_fee_currency=fee_currency,
        explicit_fee_quote=fee_quote,
        implementation_shortfall_quote=shortfall,
        total_cost_quote=total,
        complete_for_quote_cost=complete,
    )
