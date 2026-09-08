from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IchimokuOpportunityConfig:
    triangle_window: int = 20
    below_cloud_fraction: float = 0.70
    volume_window: int = 20
    volume_expansion: float = 1.20
    atr_window: int = 14
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.08
    breakout_buffer_atr: float = 0.10


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope over trailing windows; uses current/past values only."""

    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))

    def slope(values: np.ndarray) -> float:
        if len(values) != window or np.isnan(values).any():
            return np.nan
        y = values.astype(float)
        y = y - y.mean()
        return float(np.sum(x * y) / denom)

    return series.rolling(window).apply(slope, raw=True)


def add_ichimoku_state(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe Ichimoku state for modelling.

    Senkou values are represented at the decision timestamp from information
    known at that timestamp.  We do not shift them backward to manufacture a
    future feature, and Chikou is intentionally excluded from model inputs.
    """

    required = {"timestamp", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    x = df.copy()
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    close = x["close"].astype(float)

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a_now = (tenkan + kijun) / 2.0
    span_b_now = (high.rolling(52).max() + low.rolling(52).min()) / 2.0
    cloud_top = pd.concat([span_a_now, span_b_now], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a_now, span_b_now], axis=1).min(axis=1)

    x["ichi_tenkan"] = tenkan
    x["ichi_kijun"] = kijun
    x["ichi_span_a_now"] = span_a_now
    x["ichi_span_b_now"] = span_b_now
    x["ichi_cloud_top_now"] = cloud_top
    x["ichi_cloud_bottom_now"] = cloud_bottom
    x["ichi_tk_distance_pct"] = (tenkan - kijun) / close
    x["ichi_price_kijun_pct"] = (close - kijun) / close
    x["ichi_price_cloud_top_pct"] = (close - cloud_top) / close
    x["ichi_cloud_width_pct"] = (cloud_top - cloud_bottom) / close
    x["ichi_cloud_bullish"] = (span_a_now > span_b_now).astype(float)
    x["ichi_tk_bullish"] = (tenkan > kijun).astype(float)
    x["ichi_price_above_cloud"] = (close > cloud_top).astype(float)
    x["ichi_price_below_cloud"] = (close < cloud_bottom).astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def detect_kumo_triangle_breakout(
    df: pd.DataFrame,
    config: IchimokuOpportunityConfig | None = None,
) -> pd.DataFrame:
    """Algorithmic candidate detector for the project's triangle-under-Kumo idea.

    The detector is deliberately a *candidate* generator, not an entry rule.
    Every condition uses only information available at or before each row.
    The current close is compared with the *previous* rolling resistance to
    avoid using the breakout candle to define its own threshold.
    """

    cfg = config or IchimokuOpportunityConfig()
    x = add_ichimoku_state(df)
    w = cfg.triangle_window

    high_slope = _rolling_slope(x["high"], w)
    low_slope = _rolling_slope(x["low"], w)
    prior_resistance = x["high"].shift(1).rolling(w).max()
    prior_support = x["low"].shift(1).rolling(w).min()
    pattern_width = (prior_resistance - prior_support) / x["close"]
    prior_width = pattern_width.shift(max(2, w // 4))

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / cfg.atr_window, adjust=False).mean()
    atr_pct = atr / x["close"]

    below_cloud_fraction = (
        x["ichi_price_below_cloud"].shift(1).rolling(w).mean()
    )
    triangle_geometry = (high_slope < 0) & (low_slope > 0) & (pattern_width < prior_width)
    was_below_kumo = below_cloud_fraction >= cfg.below_cloud_fraction
    breakout_level = prior_resistance + cfg.breakout_buffer_atr * atr
    confirmed_breakout = x["close"] > breakout_level
    tk_confirmation = x["ichi_tk_bullish"].eq(1.0)
    kumo_confirmation = x["close"] > x["ichi_cloud_bottom_now"]
    volume_baseline = x["volume"].shift(1).rolling(cfg.volume_window).median()
    volume_confirmation = x["volume"] >= cfg.volume_expansion * volume_baseline
    volatility_ok = atr_pct.between(cfg.min_atr_pct, cfg.max_atr_pct)

    x["triangle_high_slope"] = high_slope
    x["triangle_low_slope"] = low_slope
    x["triangle_width_pct"] = pattern_width
    x["triangle_below_kumo_fraction"] = below_cloud_fraction
    x["triangle_geometry"] = triangle_geometry.astype(float)
    x["triangle_was_below_kumo"] = was_below_kumo.astype(float)
    x["triangle_breakout_confirmed"] = confirmed_breakout.astype(float)
    x["triangle_tk_confirmed"] = tk_confirmation.astype(float)
    x["triangle_kumo_confirmed"] = kumo_confirmation.astype(float)
    x["triangle_volume_confirmed"] = volume_confirmation.astype(float)
    x["triangle_volatility_ok"] = volatility_ok.astype(float)
    x["triangle_candidate"] = (
        triangle_geometry
        & was_below_kumo
        & confirmed_breakout
        & tk_confirmation
        & kumo_confirmation
        & volume_confirmation
        & volatility_ok
    ).astype(float)
    return x.replace([np.inf, -np.inf], np.nan)
