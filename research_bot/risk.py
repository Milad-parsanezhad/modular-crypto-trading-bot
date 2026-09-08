from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_drawdown: float = 0.12
    max_gross_exposure: float = 1.0
    max_asset_weight: float = 0.35
    max_turnover_per_step: float = 0.50
    max_spread_bps: float = 35.0
    max_slippage_bps: float = 25.0
    max_cvar_95: float = 0.035
    min_cash_buffer: float = 0.05


@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    peak_equity: float
    gross_exposure: float
    asset_weight: float
    turnover: float
    spread_bps: float
    slippage_bps: float
    recent_returns: tuple[float, ...] = ()

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    kill_switch: bool
    reasons: tuple[str, ...]
    cvar_95: float | None


def historical_cvar(returns: Iterable[float], alpha: float = 0.95) -> float | None:
    """Positive loss number for historical Expected Shortfall/CVaR.

    Returns ``None`` when there are too few observations for a meaningful tail
    estimate.  The function intentionally does not backfill missing risk data.
    """

    r = pd.Series(list(returns), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 20:
        return None
    loss = -r
    var = float(loss.quantile(alpha))
    tail = loss[loss >= var]
    if tail.empty:
        return var
    return float(tail.mean())


class RiskEngine:
    """Independent pre-trade gate.

    The predictive model never bypasses this layer.  A signal can be rejected
    even when model confidence is high if portfolio, liquidity or tail-risk
    limits are breached.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def evaluate(self, snapshot: RiskSnapshot) -> RiskDecision:
        reasons: list[str] = []
        hard = False

        if snapshot.drawdown >= self.limits.max_drawdown:
            reasons.append("MAX_DRAWDOWN_BREACH")
            hard = True
        if snapshot.gross_exposure > self.limits.max_gross_exposure:
            reasons.append("MAX_GROSS_EXPOSURE_BREACH")
            hard = True
        if snapshot.asset_weight > self.limits.max_asset_weight:
            reasons.append("MAX_ASSET_WEIGHT_BREACH")
        if snapshot.turnover > self.limits.max_turnover_per_step:
            reasons.append("TURNOVER_BUDGET_BREACH")
        if snapshot.spread_bps > self.limits.max_spread_bps:
            reasons.append("SPREAD_TOO_WIDE")
        if snapshot.slippage_bps > self.limits.max_slippage_bps:
            reasons.append("SLIPPAGE_TOO_HIGH")

        cvar = historical_cvar(snapshot.recent_returns, alpha=0.95)
        if cvar is not None and cvar > self.limits.max_cvar_95:
            reasons.append("CVAR_95_BREACH")
            hard = True

        return RiskDecision(
            approved=not reasons,
            kill_switch=hard,
            reasons=tuple(reasons),
            cvar_95=cvar,
        )

    def position_size_from_risk(
        self,
        *,
        equity: float,
        stop_distance_fraction: float,
        risk_fraction: float = 0.01,
        volatility_scale: float = 1.0,
    ) -> float:
        """Return notional size in account currency.

        The result is capped by max asset weight and gross exposure.  It is a
        deterministic sizing baseline for research; more advanced portfolio
        optimizers must beat it out-of-sample before replacing it.
        """

        if equity <= 0:
            return 0.0
        if stop_distance_fraction <= 0:
            raise ValueError("stop_distance_fraction must be positive")
        if not 0 < risk_fraction <= 1:
            raise ValueError("risk_fraction must be in (0, 1]")
        scale = float(np.clip(volatility_scale, 0.0, 1.0))
        raw = equity * risk_fraction / stop_distance_fraction * scale
        cap = equity * min(self.limits.max_asset_weight, self.limits.max_gross_exposure)
        return float(max(0.0, min(raw, cap)))
