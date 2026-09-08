from __future__ import annotations

import math
import numpy as np
import pandas as pd


def periods_per_year(timeframe: str) -> float:
    return float({"1m":60*24*365,"5m":12*24*365,"15m":4*24*365,"30m":2*24*365,"1h":24*365,"4h":6*365,"8h":3*365,"1d":365}.get(timeframe,365))


def performance_metrics(returns: pd.Series, timeframe: str = "4h") -> dict:
    """Risk-adjusted performance metrics for net strategy returns.

    The function intentionally reports both central and tail-risk diagnostics.
    Profit factor and win rate here are return-period diagnostics; execution
    reports should additionally compute trade-level metrics from fills/orders.
    """

    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if r.empty:
        return {"n":0}

    ppy = periods_per_year(timeframe)
    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(r) / ppy, 1 / ppy)
    ann_return = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1)
    sharpe = float(r.mean() / vol * math.sqrt(ppy)) if vol and np.isfinite(vol) else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * math.sqrt(ppy)) if downside and np.isfinite(downside) else np.nan
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else np.nan

    positive = r[r > 0]
    negative = r[r < 0]
    gross_profit = float(positive.sum())
    gross_loss = float(-negative.sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)
    win_rate = float((r > 0).mean())
    expectancy = float(r.mean())

    loss = -r
    var_95_loss = float(loss.quantile(0.95))
    tail = loss[loss >= var_95_loss]
    cvar_95_loss = float(tail.mean()) if not tail.empty else var_95_loss

    return {
        "n": int(len(r)),
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "mean_period_return": float(r.mean()),
        "period_volatility": float(vol) if np.isfinite(vol) else np.nan,
        "win_rate_periods": win_rate,
        "profit_factor_periods": profit_factor,
        "expectancy_period": expectancy,
        "var_95_loss": var_95_loss,
        "cvar_95_loss": cvar_95_loss,
        "gross_positive_returns": gross_profit,
        "gross_negative_returns": gross_loss,
    }


def backtest_positions(
    future_returns: pd.Series,
    positions: pd.Series,
    timeframe: str = "4h",
    fee_bps: float = 10.0,
    slippage_bps: float = 2.0,
    funding_returns: pd.Series | None = None,
):
    """Cost-aware position backtest.

    ``future_returns`` must already be aligned to the position decision time.
    ``funding_returns`` is optional and should represent the signed return drag
    (positive values are costs) attributable to the held position for the same
    interval.  No funding series is fabricated when it is unavailable.
    """

    fr = pd.Series(future_returns).astype(float)
    pos = pd.Series(positions, index=fr.index).fillna(0.0).clip(-1, 1)
    turnover = pos.diff().abs().fillna(pos.abs())
    friction = turnover * (fee_bps + slippage_bps) / 10000.0

    if funding_returns is None:
        funding_drag = pd.Series(0.0, index=fr.index)
        funding_available = False
    else:
        funding_drag = pd.Series(funding_returns, index=fr.index).fillna(0.0).astype(float) * pos.abs()
        funding_available = True

    gross_strategy_returns = pos * fr
    strategy_returns = gross_strategy_returns - friction - funding_drag
    metrics = performance_metrics(strategy_returns, timeframe=timeframe)
    metrics.update(
        {
            "turnover_sum": float(turnover.sum()),
            "trade_events": int((turnover > 0).sum()),
            "avg_abs_position": float(pos.abs().mean()),
            "exposure_fraction": float((pos != 0).mean()),
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "total_explicit_cost": float(friction.sum()),
            "total_funding_drag": float(funding_drag.sum()),
            "funding_available": bool(funding_available),
            "gross_strategy_return_sum": float(gross_strategy_returns.sum()),
            "net_strategy_return_sum": float(strategy_returns.sum()),
        }
    )
    return strategy_returns, metrics
