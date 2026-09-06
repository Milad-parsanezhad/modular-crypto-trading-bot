"""Causal feature engineering; no negative shifts in predictor columns."""
import numpy as np
import pandas as pd


def make_features(df):
    out = df.copy()
    close, high, low = df.close, df.high, df.low
    def midpoint(window):
        return (high.rolling(window).max() + low.rolling(window).min()) / 2
    tenkan, kijun = midpoint(9), midpoint(26)
    # Values drawn at today's timestamp were computed 26 bars ago.
    cloud_a = ((tenkan + kijun) / 2).shift(26)
    cloud_b = midpoint(52).shift(26)
    out["tenkan"], out["kijun"] = tenkan, kijun
    out["cloud_top"] = pd.concat([cloud_a, cloud_b], axis=1).max(axis=1, skipna=False)
    out["cloud_bottom"] = pd.concat([cloud_a, cloud_b], axis=1).min(axis=1, skipna=False)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    returns = np.log(close / close.shift())
    out["feature_return_1"] = returns
    out["feature_return_6"] = np.log(close / close.shift(6))
    out["feature_volatility"] = returns.rolling(24).std()
    out["feature_range"] = (high-low) / close
    out["feature_atr"] = out.atr / close
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out["feature_rsi"] = (up / (up+down)).where(up+down > 0, 0.5)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12-ema26
    out["feature_macd"] = macd / close
    out["feature_macd_hist"] = (macd-macd.ewm(span=9, adjust=False).mean()) / close
    volume_mean = df.volume.rolling(24).mean()
    out["feature_volume"] = (df.volume / volume_mean).where(volume_mean > 0, 0.0)
    out["feature_ichi_tenkan"] = tenkan / close - 1
    out["feature_ichi_kijun"] = kijun / close - 1
    out["feature_ichi_cross"] = (tenkan-kijun) / close
    out["feature_ichi_cloud_a"] = cloud_a / close - 1
    out["feature_ichi_cloud_b"] = cloud_b / close - 1
    out["feature_ichi_future_cloud_width"] = ((tenkan+kijun)/2-midpoint(52)) / close
    # Chikou comparison known now; never use close.shift(-26) as an input.
    out["feature_ichi_chikou_comparison"] = close / close.shift(26) - 1
    out["feature_ichi_kijun_slope"] = kijun.diff(3) / close
    trailing = close.pct_change(48)
    threshold = out.feature_volatility * np.sqrt(48)
    out["regime"] = np.where(trailing > threshold, "bull", np.where(trailing < -threshold, "bear", "sideways"))
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def feature_columns(frame, ichimoku=True):
    return [c for c in frame if c.startswith("feature_") and (ichimoku or not c.startswith("feature_ichi_"))]


def labels(frame, horizon=1, cost=0.002):
    """Decision after close t; target is open(t+1) to open(t+1+h)."""
    future = frame.open.shift(-(horizon+1)) / frame.open.shift(-1) - 1
    return (future > cost).astype(float).where(future.notna())


def ichimoku_signal(row):
    return int(row.close > row.cloud_top and row.tenkan > row.kijun)
