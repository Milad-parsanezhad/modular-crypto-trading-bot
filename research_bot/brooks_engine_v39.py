from __future__ import annotations

"""Objective Al Brooks-inspired price-action research engine for v0.39.

This module formalizes auditable proxies for Always-In direction, trend/range,
breakout/failed breakout, H1/H2/L1/L2, three-push wedge, micro double top/bottom,
signal bar, follow-through and measured move. It does not claim to reproduce
Brooks' discretionary method in full. All features are closed-bar causal.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrooksPolicyV39:
    efficiency_window: int = 20
    trend_efficiency: float = 0.35
    range_efficiency: float = 0.20
    body_fraction: float = 0.55
    close_location: float = 0.65
    micro_double_window: int = 8
    micro_double_tolerance_atr: float = 0.20
    wedge_lookback_bars: int = 30


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    direction = close.diff(window).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return (direction / path.replace(0.0, np.nan)).clip(0.0, 1.0)


def _three_push(
    high: pd.Series,
    low: pd.Series,
    *,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Return causal wedge-top and wedge-bottom proxies.

    A pivot at t-1 is only known on t, so the event is stamped on t.
    """

    n = len(high)
    wedge_top = np.zeros(n, dtype=np.int8)
    wedge_bottom = np.zeros(n, dtype=np.int8)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    h = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)

    for i in range(2, n):
        pivot_i = i - 1
        if np.isfinite(h[i - 2 : i + 1]).all() and h[pivot_i] > h[i - 2] and h[pivot_i] >= h[i]:
            highs.append((i, h[pivot_i]))
        if np.isfinite(l[i - 2 : i + 1]).all() and l[pivot_i] < l[i - 2] and l[pivot_i] <= l[i]:
            lows.append((i, l[pivot_i]))

        highs = [(j, v) for j, v in highs if i - j <= lookback]
        lows = [(j, v) for j, v in lows if i - j <= lookback]

        if len(highs) >= 3:
            v = [z[1] for z in highs[-3:]]
            if v[0] < v[1] < v[2]:
                wedge_top[i] = 1
        if len(lows) >= 3:
            v = [z[1] for z in lows[-3:]]
            if v[0] > v[1] > v[2]:
                wedge_bottom[i] = 1

    return (
        pd.Series(wedge_top, index=high.index, dtype="int8"),
        pd.Series(wedge_bottom, index=high.index, dtype="int8"),
    )


def _h1_h2_l1_l2(
    x: pd.DataFrame,
    always_long: pd.Series,
    always_short: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Auditable H1/H2/L1/L2 pullback-attempt proxy."""

    n = len(x)
    h1 = np.zeros(n, dtype=np.int8)
    h2 = np.zeros(n, dtype=np.int8)
    l1 = np.zeros(n, dtype=np.int8)
    l2 = np.zeros(n, dtype=np.int8)

    high = x["high"].to_numpy(dtype=float)
    low = x["low"].to_numpy(dtype=float)
    open_ = x["open"].to_numpy(dtype=float)
    close = x["close"].to_numpy(dtype=float)
    prior_hi20 = x["prior_high20"].to_numpy(dtype=float)
    prior_lo20 = x["prior_low20"].to_numpy(dtype=float)
    bull = always_long.to_numpy(dtype=bool)
    bear = always_short.to_numpy(dtype=bool)

    bull_attempts = 0
    bear_attempts = 0
    bull_armed = False
    bear_armed = False

    for i in range(1, n):
        if not bull[i]:
            bull_attempts = 0
            bull_armed = False
        else:
            if low[i] < low[i - 1] or close[i] < close[i - 1]:
                bull_armed = True
            if bull_armed and high[i] > high[i - 1] and close[i] > open_[i]:
                bull_attempts += 1
                if bull_attempts == 1:
                    h1[i] = 1
                else:
                    h2[i] = 1
                    bull_attempts = 0
                bull_armed = False
            if np.isfinite(prior_hi20[i]) and close[i] > prior_hi20[i]:
                bull_attempts = 0
                bull_armed = False

        if not bear[i]:
            bear_attempts = 0
            bear_armed = False
        else:
            if high[i] > high[i - 1] or close[i] > close[i - 1]:
                bear_armed = True
            if bear_armed and low[i] < low[i - 1] and close[i] < open_[i]:
                bear_attempts += 1
                if bear_attempts == 1:
                    l1[i] = 1
                else:
                    l2[i] = 1
                    bear_attempts = 0
                bear_armed = False
            if np.isfinite(prior_lo20[i]) and close[i] < prior_lo20[i]:
                bear_attempts = 0
                bear_armed = False

    idx = x.index
    return (
        pd.Series(h1, index=idx, dtype="int8"),
        pd.Series(h2, index=idx, dtype="int8"),
        pd.Series(l1, index=idx, dtype="int8"),
        pd.Series(l2, index=idx, dtype="int8"),
    )


def add_brooks_features_v39(
    features: pd.DataFrame,
    policy: BrooksPolicyV39 | None = None,
) -> pd.DataFrame:
    p = policy or BrooksPolicyV39()
    required = {
        "open", "high", "low", "close", "atr", "ema20", "ema50",
        "prior_high20", "prior_low20",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"missing Brooks feature columns: {sorted(missing)}")

    x = features.copy()
    rng = (x["high"] - x["low"]).replace(0.0, np.nan)
    body = (x["close"] - x["open"]).abs()
    body_fraction = body / rng
    close_loc = (x["close"] - x["low"]) / rng

    x["trend_efficiency_v39"] = _efficiency_ratio(x["close"], p.efficiency_window)
    x["brooks_trend_regime_v39"] = x["trend_efficiency_v39"].ge(p.trend_efficiency).astype("int8")
    x["brooks_range_regime_v39"] = x["trend_efficiency_v39"].le(p.range_efficiency).astype("int8")

    ema20_slope = x["ema20"].diff(3)
    x["brooks_always_in_long_v39"] = (
        (x["ema20"] > x["ema50"])
        & (x["close"] > x["ema20"])
        & (ema20_slope > 0)
        & x["trend_efficiency_v39"].ge(p.range_efficiency)
    ).astype("int8")
    x["brooks_always_in_short_v39"] = (
        (x["ema20"] < x["ema50"])
        & (x["close"] < x["ema20"])
        & (ema20_slope < 0)
        & x["trend_efficiency_v39"].ge(p.range_efficiency)
    ).astype("int8")

    x["brooks_breakout_up_v39"] = (
        (x["close"] > x["prior_high20"])
        & (x["close"].shift(1) <= x["prior_high20"].shift(1))
    ).astype("int8")
    x["brooks_breakout_down_v39"] = (
        (x["close"] < x["prior_low20"])
        & (x["close"].shift(1) >= x["prior_low20"].shift(1))
    ).astype("int8")
    x["brooks_failed_breakdown_v39"] = (
        (x["low"] < x["prior_low20"]) & (x["close"] > x["prior_low20"])
    ).astype("int8")
    x["brooks_failed_breakout_v39"] = (
        (x["high"] > x["prior_high20"]) & (x["close"] < x["prior_high20"])
    ).astype("int8")

    x["brooks_signal_bull_v39"] = (
        (x["close"] > x["open"])
        & body_fraction.ge(p.body_fraction)
        & close_loc.ge(p.close_location)
    ).astype("int8")
    x["brooks_signal_bear_v39"] = (
        (x["close"] < x["open"])
        & body_fraction.ge(p.body_fraction)
        & close_loc.le(1.0 - p.close_location)
    ).astype("int8")

    x["brooks_follow_through_bull_v39"] = (
        (
            x["brooks_breakout_up_v39"].shift(1).fillna(0).eq(1)
            | x["brooks_signal_bull_v39"].shift(1).fillna(0).eq(1)
        )
        & (x["close"] > x["high"].shift(1))
    ).astype("int8")
    x["brooks_follow_through_bear_v39"] = (
        (
            x["brooks_breakout_down_v39"].shift(1).fillna(0).eq(1)
            | x["brooks_signal_bear_v39"].shift(1).fillna(0).eq(1)
        )
        & (x["close"] < x["low"].shift(1))
    ).astype("int8")

    h1, h2, l1, l2 = _h1_h2_l1_l2(
        x,
        x["brooks_always_in_long_v39"].eq(1),
        x["brooks_always_in_short_v39"].eq(1),
    )
    x["brooks_h1_v39"] = h1
    x["brooks_h2_v39"] = h2
    x["brooks_l1_v39"] = l1
    x["brooks_l2_v39"] = l2

    wedge_top, wedge_bottom = _three_push(x["high"], x["low"], lookback=p.wedge_lookback_bars)
    x["brooks_wedge_top_v39"] = wedge_top
    x["brooks_wedge_bottom_v39"] = wedge_bottom

    prev_low = x["low"].shift(2).rolling(p.micro_double_window, min_periods=3).min()
    prev_high = x["high"].shift(2).rolling(p.micro_double_window, min_periods=3).max()
    tol = p.micro_double_tolerance_atr * x["atr"]
    x["brooks_micro_double_bottom_v39"] = (
        ((x["low"] - prev_low).abs() <= tol) & (x["close"] > x["open"])
    ).astype("int8")
    x["brooks_micro_double_top_v39"] = (
        ((x["high"] - prev_high).abs() <= tol) & (x["close"] < x["open"])
    ).astype("int8")

    prior_range = (x["prior_high20"] - x["prior_low20"]).replace(0.0, np.nan)
    x["brooks_measured_move_progress_long_v39"] = (
        (x["close"] - x["prior_high20"]) / prior_range
    ).clip(-2.0, 2.0)
    x["brooks_measured_move_progress_short_v39"] = (
        (x["prior_low20"] - x["close"]) / prior_range
    ).clip(-2.0, 2.0)
    x["brooks_measured_move_target_long_v39"] = x["prior_high20"] + prior_range
    x["brooks_measured_move_target_short_v39"] = x["prior_low20"] - prior_range

    long_cols = [
        "brooks_always_in_long_v39", "brooks_breakout_up_v39",
        "brooks_failed_breakdown_v39", "brooks_h1_v39", "brooks_h2_v39",
        "brooks_wedge_bottom_v39", "brooks_micro_double_bottom_v39",
        "brooks_signal_bull_v39", "brooks_follow_through_bull_v39",
    ]
    short_cols = [
        "brooks_always_in_short_v39", "brooks_breakout_down_v39",
        "brooks_failed_breakout_v39", "brooks_l1_v39", "brooks_l2_v39",
        "brooks_wedge_top_v39", "brooks_micro_double_top_v39",
        "brooks_signal_bear_v39", "brooks_follow_through_bear_v39",
    ]
    x["brooks_score_v39"] = (
        x[long_cols].sum(axis=1) - x[short_cols].sum(axis=1)
    ) / float(len(long_cols))
    return x.replace([np.inf, -np.inf], np.nan)
