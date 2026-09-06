from __future__ import annotations

import math
import numpy as np
import pandas as pd


def periods_per_year(timeframe: str) -> float:
    return float({"1m":60*24*365,"5m":12*24*365,"15m":4*24*365,"30m":2*24*365,"1h":24*365,"4h":6*365,"1d":365}.get(timeframe,365))


def performance_metrics(returns: pd.Series, timeframe: str = "4h") -> dict:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty: return {"n":0}
    ppy = periods_per_year(timeframe); equity=(1+r).cumprod(); total_return=float(equity.iloc[-1]-1); years=max(len(r)/ppy,1/ppy)
    ann_return=float(equity.iloc[-1]**(1/years)-1) if equity.iloc[-1] > 0 else -1.0
    vol=r.std(ddof=1); sharpe=float(r.mean()/vol*math.sqrt(ppy)) if vol and np.isfinite(vol) else np.nan
    downside=r[r<0].std(ddof=1); sortino=float(r.mean()/downside*math.sqrt(ppy)) if downside and np.isfinite(downside) else np.nan
    dd=equity/equity.cummax()-1; mdd=float(dd.min()); calmar=float(ann_return/abs(mdd)) if mdd<0 else np.nan
    return {"n":int(len(r)),"total_return":total_return,"annualized_return":ann_return,"sharpe":sharpe,"sortino":sortino,"max_drawdown":mdd,"calmar":calmar,"mean_period_return":float(r.mean()),"period_volatility":float(vol) if np.isfinite(vol) else np.nan}


def backtest_positions(future_returns: pd.Series, positions: pd.Series, timeframe: str = "4h", fee_bps: float = 10.0, slippage_bps: float = 2.0):
    fr=pd.Series(future_returns).astype(float); pos=pd.Series(positions,index=fr.index).fillna(0.0).clip(-1,1)
    turnover=pos.diff().abs().fillna(pos.abs()); friction=turnover*(fee_bps+slippage_bps)/10000.0
    strategy_returns=pos*fr-friction; metrics=performance_metrics(strategy_returns,timeframe=timeframe)
    metrics.update({"turnover_sum":float(turnover.sum()),"avg_abs_position":float(pos.abs().mean()),"fee_bps":float(fee_bps),"slippage_bps":float(slippage_bps)})
    return strategy_returns, metrics
