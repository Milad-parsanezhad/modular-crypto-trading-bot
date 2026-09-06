from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-prev_close).abs(), (df["low"]-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features using only current/past bars."""
    x = df.copy()
    close = x["close"]
    ret = close.pct_change()
    x["ret_1"] = ret
    x["log_ret_1"] = np.log(close).diff()
    for w in (3, 6, 12, 24, 48): x[f"mom_{w}"] = close.pct_change(w)
    for w in (6, 12, 24, 48): x[f"vol_{w}"] = ret.rolling(w).std()
    x["rsi_14"] = _rsi(close, 14) / 100.0
    x["atr_14_pct"] = _atr(x, 14) / close
    x["range_pct"] = (x["high"] - x["low"]) / close
    x["volume_log"] = np.log1p(x["volume"])
    x["volume_z_24"] = (x["volume"]-x["volume"].rolling(24).mean()) / x["volume"].rolling(24).std().replace(0, np.nan)
    x["amihud_24"] = (ret.abs()/x["volume"].replace(0, np.nan)).rolling(24).mean()
    for w in (12, 24, 48, 96):
        x[f"price_to_ma_{w}"] = close / close.rolling(w).mean() - 1.0
    tenkan = (x["high"].rolling(9).max()+x["low"].rolling(9).min())/2
    kijun = (x["high"].rolling(26).max()+x["low"].rolling(26).min())/2
    span_b_now = (x["high"].rolling(52).max()+x["low"].rolling(52).min())/2
    span_a_now = (tenkan+kijun)/2
    x["ichi_tenkan_kijun"] = (tenkan-kijun)/close
    x["ichi_price_kijun"] = (close-kijun)/close
    x["ichi_cloud_width"] = (span_a_now-span_b_now).abs()/close
    ts = pd.to_datetime(x["timestamp"], utc=True)
    x["hour_sin"] = np.sin(2*np.pi*ts.dt.hour/24); x["hour_cos"] = np.cos(2*np.pi*ts.dt.hour/24)
    x["dow_sin"] = np.sin(2*np.pi*ts.dt.dayofweek/7); x["dow_cos"] = np.cos(2*np.pi*ts.dt.dayofweek/7)
    return x.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = ["ret_1","log_ret_1","mom_3","mom_6","mom_12","mom_24","mom_48","vol_6","vol_12","vol_24","vol_48","rsi_14","atr_14_pct","range_pct","volume_log","volume_z_24","amihud_24","price_to_ma_12","price_to_ma_24","price_to_ma_48","price_to_ma_96","ichi_tenkan_kijun","ichi_price_kijun","ichi_cloud_width","hour_sin","hour_cos","dow_sin","dow_cos"]
