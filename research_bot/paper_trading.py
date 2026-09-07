"""Live paper trading simulator with order book and slippage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a single order."""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str = "MARKET"
    status: str = "FILLED"
    slippage_pct: float = 0.0
    commission_pct: float = 0.0
    
    def cost(self) -> float:
        """Total cost including commission."""
        base_cost = self.quantity * self.price
        commission = base_cost * self.commission_pct
        slippage = base_cost * self.slippage_pct
        return base_cost + commission + slippage


class PaperTradingEngine:
    """Simulate live trading without real capital."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 2.0,
    ):
        """Initialize paper trading engine."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.orders: List[Order] = []
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_bps / 10000.0
        self.equity_history = [initial_capital]

    def place_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Order:
        """Simulate market order execution."""
        order = Order(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )
        
        # Update cash
        if side == "BUY":
            cost = order.cost()
            if cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol} order")
                order.status = "REJECTED"
                return order
            self.cash -= cost
        elif side == "SELL":
            if self.positions.get(symbol, 0) < quantity:
                logger.warning(f"Insufficient position for {symbol} sell")
                order.status = "REJECTED"
                return order
            self.cash += quantity * price * (1 - self.commission_pct - self.slippage_pct)
        
        # Update position
        current_pos = self.positions.get(symbol, 0)
        if side == "BUY":
            self.positions[symbol] = current_pos + quantity
        else:
            self.positions[symbol] = current_pos - quantity
        
        self.orders.append(order)
        logger.info(f"Order filled: {side} {quantity} {symbol} @ {price:.2f}")
        return order

    def get_portfolio_value(self, prices: dict) -> float:
        """Calculate current portfolio value."""
        position_value = sum(
            qty * prices.get(symbol, 0)
            for symbol, qty in self.positions.items()
            if qty > 0
        )
        return self.cash + position_value

    def get_performance_metrics(self) -> dict:
        """Calculate trading performance."""
        total_trades = len([o for o in self.orders if o.status == "FILLED"])
        winning_trades = len([o for o in self.orders if o.side == "SELL"])  # Simplified
        
        return {
            "total_trades": total_trades,
            "cash_remaining": self.cash,
            "positions": dict(self.positions),
            "commissions_paid": sum(o.cost() * o.commission_pct for o in self.orders),
        }
