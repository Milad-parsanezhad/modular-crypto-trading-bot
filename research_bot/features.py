from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI with proper NaN handling."""
    if len(close) < window + 1:
        return pd.Series(np.nan, index=close.index)
    
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate ATR with proper NaN handling."""
    if len(df) < window + 1:
        return pd.Series(np.nan, index=df.index)
    
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features using only current/past bars.
    
    CRITICAL: No look-ahead bias. All features use only data available at bar close.
    """
    x = df.copy()
    close = x["close"]
    ret = close.pct_change()
    
    # Momentum features (point-in-time)
    x["ret_1"] = ret
    x["log_ret_1"] = np.log(close / close.shift(1))
    
    for w in (3, 6, 12, 24, 48):
        x[f"mom_{w}"] = close.pct_change(w)
    
    # Volatility features
    for w in (6, 12, 24, 48):
        x[f"vol_{w}"] = ret.rolling(w).std()
    
    # Technical indicators
    x["rsi_14"] = _rsi(close, 14) / 100.0
    x["atr_14_pct"] = _atr(x, 14) / close.replace(0, np.nan)
    x["range_pct"] = (x["high"] - x["low"]) / close.replace(0, np.nan)
    
    # Volume features
    x["volume_log"] = np.log1p(x["volume"])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_std = x["volume"].rolling(24).std()
        x["volume_z_24"] = (x["volume"] - x["volume"].rolling(24).mean()) / volume_std.replace(0, np.nan)
    
    # Amihud illiquidity
    with np.errstate(divide='ignore', invalid='ignore'):
        x["amihud_24"] = (ret.abs() / x["volume"].replace(0, np.nan)).rolling(24).mean()
    
    # Moving average ratios
    for w in (12, 24, 48, 96):
        ma = close.rolling(w).mean()
        x[f"price_to_ma_{w}"] = close / ma.replace(0, np.nan) - 1.0
    
    # Ichimoku components (NO FUTURE SHIFT)
    tenkan = (x["high"].rolling(9).max() + x["low"].rolling(9).min()) / 2
    kijun = (x["high"].rolling(26).max() + x["low"].rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (x["high"].rolling(52).max() + x["low"].rolling(52).min()) / 2
    
    x["ichi_tenkan_kijun"] = (tenkan - kijun) / close.replace(0, np.nan)
    x["ichi_price_kijun"] = (close - kijun) / close.replace(0, np.nan)
    x["ichi_cloud_width"] = (span_a - span_b).abs() / close.replace(0, np.nan)
    
    # Temporal features
    ts = pd.to_datetime(x["timestamp"], utc=True)
    x["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    x["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    x["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    
    # Replace infinities
    x = x.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Features computed: {len(x)} bars, {x.isnull().sum().sum()} NaNs total")
    return x


FEATURE_COLUMNS = [
    "ret_1", "log_ret_1",
    "mom_3", "mom_6", "mom_12", "mom_24", "mom_48",
    "vol_6", "vol_12", "vol_24", "vol_48",
    "rsi_14", "atr_14_pct", "range_pct",
    "volume_log", "volume_z_24", "amihud_24",
    "price_to_ma_12", "price_to_ma_24", "price_to_ma_48", "price_to_ma_96",
    "ichi_tenkan_kijun", "ichi_price_kijun", "ichi_cloud_width",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
