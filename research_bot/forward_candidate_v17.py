from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .strategy_lab import StrategyLabConfig, build_strategy_features


STRATEGY_VERSION = "S6_BREAKOUT_V17"


@dataclass(frozen=True)
class ForwardCandidateLimits:
    """Pre-registered paper limits. They do not authorize live execution."""

    risk_per_trade: float = 0.0025
    max_asset_weight: float = 0.35
    max_portfolio_gross: float = 0.70
    max_drawdown: float = 0.05
    max_spread_bps: float = 35.0
    max_slippage_bps: float = 25.0
    cooling_bars: int = 42


@dataclass(frozen=True)
class ForwardCandidateSignal:
    strategy_version: str
    timestamp: str
    action: str
    reference_price: float
    stop_reference: float
    breakout_reference: float
    risk_weight: float
    reasons: tuple[str, ...]
    limits: dict
    paper_only: bool = True
    live_execution: bool = False


def s6_candidate_signal(
    frame: pd.DataFrame,
    *,
    currently_long: bool,
    config: StrategyLabConfig | None = None,
    limits: ForwardCandidateLimits | None = None,
) -> ForwardCandidateSignal:
    """Return the latest causal S6 paper decision without placing an order.

    A decision made after close[t] is eligible only for a paper fill at
    open[t+1]. The function deliberately has no exchange execution dependency.
    """

    cfg = config or StrategyLabConfig()
    risk = limits or ForwardCandidateLimits()
    if len(frame) < max(cfg.ema_window + 6, cfg.breakout_window + 1):
        raise ValueError("insufficient closed 4h bars for S6")
    x = build_strategy_features(frame, cfg)
    prior_high = x["high"].shift(1).rolling(cfg.breakout_window).max()
    prior_low = x["low"].shift(1).rolling(cfg.exit_window).min()
    row = x.iloc[-1]
    breakout = float(prior_high.iloc[-1])
    stop = float(prior_low.iloc[-1])
    close = float(row["close"])
    trend_positive = bool(float(row["ema_200_slope"]) > 0.0)
    breakout_now = bool(close > breakout and trend_positive)
    exit_now = bool(close < stop)

    if currently_long and exit_now:
        action = "EXIT_CANDIDATE"
        reasons = ("CLOSE_BELOW_PRIOR_10_BAR_LOW",)
    elif currently_long:
        action = "HOLD"
        reasons = ("S6_POSITION_ACTIVE",)
    elif breakout_now:
        action = "BUY_CANDIDATE"
        reasons = ("CLOSE_ABOVE_PRIOR_20_BAR_HIGH", "EMA200_SLOPE_POSITIVE")
    else:
        action = "NO_TRADE"
        reasons = tuple(
            reason
            for condition, reason in (
                (not close > breakout, "NO_20_BAR_BREAKOUT"),
                (not trend_positive, "EMA200_SLOPE_NOT_POSITIVE"),
            )
            if condition
        )

    atr_pct = float(row["atr_pct"])
    proposed = cfg.risk_per_trade / (cfg.stop_atr * atr_pct) if np.isfinite(atr_pct) and atr_pct > 0 else 0.0
    weight = float(np.clip(proposed, 0.0, min(risk.max_asset_weight, risk.max_portfolio_gross)))
    return ForwardCandidateSignal(
        strategy_version=STRATEGY_VERSION,
        timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
        action=action,
        reference_price=close,
        stop_reference=stop,
        breakout_reference=breakout,
        risk_weight=weight,
        reasons=reasons,
        limits=asdict(risk),
    )
