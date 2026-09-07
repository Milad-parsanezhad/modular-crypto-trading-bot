from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def periods_per_year(timeframe: str) -> float:
    """Convert timeframe to periods per year."""
    mapping = {
        "1m": 60 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "30m": 2 * 24 * 365,
        "1h": 24 * 365,
        "4h": 6 * 365,
        "1d": 365,
    }
    return float(mapping.get(timeframe, 365))


def performance_metrics(
    returns: pd.Series,
    timeframe: str = "4h",
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    r = pd.Series(returns).dropna().astype(float)
    
    if r.empty:
        return {"n": 0, "total_return": 0.0, "sharpe": np.nan}
    
    ppy = periods_per_year(timeframe)
    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(r) / ppy, 1 / ppy)
    
    # Annualized return
    ann_return = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    
    # Sharpe ratio
    mean_ret = r.mean()
    vol = r.std(ddof=1)
    sharpe = float(mean_ret / vol * math.sqrt(ppy)) if vol > 0 and np.isfinite(vol) else np.nan
    
    # Sortino ratio (downside volatility)
    downside_ret = r[r < 0]
    downside_vol = downside_ret.std(ddof=1) if len(downside_ret) > 1 else 0
    sortino = float(mean_ret / downside_vol * math.sqrt(ppy)) if downside_vol > 0 and np.isfinite(downside_vol) else np.nan
    
    # Drawdown metrics
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    calmar = float(ann_return / abs(mdd)) if mdd < 0 and np.isfinite(mdd) else np.nan
    
    return {
        "n": int(len(r)),
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "mean_period_return": float(mean_ret),
        "annual_volatility": float(vol * math.sqrt(ppy)),
    }


def backtest_positions(
    future_returns: pd.Series,
    positions: pd.Series,
    timeframe: str = "4h",
    fee_bps: float = 10.0,
    slippage_bps: float = 2.0,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Backtest positions with costs."""
    fr = pd.Series(future_returns).astype(float)
    pos = pd.Series(positions, index=fr.index).fillna(0.0).clip(-1, 1)
    
    # Turnover and friction
    turnover = pos.diff().abs().fillna(pos.abs())
    friction = turnover * (fee_bps + slippage_bps) / 10000.0
    
    # Strategy returns net of costs
    strategy_returns = pos * fr - friction
    
    # Metrics
    metrics = performance_metrics(strategy_returns, timeframe=timeframe)
    metrics.update({
        "turnover_sum": float(turnover.sum()),
        "avg_abs_position": float(pos.abs().mean()),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
    })
    
    return strategy_returns, metrics
