"""Risk control: Kelly Criterion, drawdown limits, position sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Position sizing and portfolio risk control."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_drawdown_pct: float = 0.20,
        kelly_fraction: float = 0.25,
        max_leverage: float = 1.0,
    ):
        """Initialize risk parameters.
        
        Args:
            initial_capital: Starting portfolio value
            max_drawdown_pct: Circuit breaker on cumulative drawdown (0-1)
            kelly_fraction: Fraction of Kelly Criterion for position sizing
            max_leverage: Maximum position leverage
        """
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.equity_curve = [initial_capital]

    def kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
    ) -> float:
        """Calculate Kelly Criterion position size.
        
        f* = (bp - q) / b
        where p = win_rate, q = 1-p, b = win_loss_ratio
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        p = win_rate
        q = 1 - p
        b = max(win_loss_ratio, 0.1)  # Avoid division by zero
        
        kelly = (b * p - q) / b
        # Fractional Kelly for safety
        return max(0.0, min(kelly * self.kelly_fraction, self.max_leverage))

    def calculate_position_size(
        self,
        current_equity: float,
        win_rate: float,
        win_loss_ratio: float,
        volatility: float,
        base_position: float = 1.0,
    ) -> float:
        """Adjust position size based on Kelly and volatility."""
        kelly_size = self.kelly_criterion(win_rate, win_loss_ratio)
        vol_adjustment = 1.0 / max(volatility, 0.01)
        
        position = base_position * kelly_size * vol_adjustment
        return min(position, self.max_leverage)

    def check_drawdown_circuit_breaker(self, equity_curve: pd.Series) -> bool:
        """Return True if circuit breaker is triggered (max drawdown exceeded)."""
        if len(equity_curve) < 2:
            return False
        
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()
        
        is_breached = max_dd < -self.max_drawdown_pct
        if is_breached:
            logger.warning(f"Drawdown circuit breaker triggered: {max_dd:.2%}")
        
        return is_breached

    def scale_positions(
        self,
        positions: pd.Series,
        equity_curve: pd.Series,
    ) -> pd.Series:
        """Scale positions based on current equity and risk limits."""
        if self.check_drawdown_circuit_breaker(equity_curve):
            return pd.Series(0.0, index=positions.index)
        
        current_equity = equity_curve.iloc[-1]
        equity_ratio = current_equity / self.initial_capital
        
        # Scale down if equity has declined
        scale_factor = min(equity_ratio, 1.0)
        
        return positions * scale_factor
