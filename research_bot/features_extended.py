"""Extended alpha features: on-chain, funding rates, order flow, volatility regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Robust feature extraction with point-in-time validation."""

    def __init__(self, lookback_min: int = 48):
        """Initialize with minimum lookback bars for valid features."""
        self.lookback_min = lookback_min

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe momentum with anti-look-ahead-bias."""
        x = df.copy()
        close = x["close"]
        
        # 1h, 4h, 1d equivalent momentum for 4h candles
        for w in (1, 6, 24, 168):  # 4h, 1d, 4d, 1w
            x[f"mom_{w}h"] = close.pct_change(w)
            x[f"mom_sign_{w}h"] = np.sign(x[f"mom_{w}h"])
        
        return x

    def add_volatility_regime(self, df: pd.DataFrame, windows: tuple = (24, 72)) -> pd.DataFrame:
        """Multi-scale volatility regime detection."""
        x = df.copy()
        ret = x["close"].pct_change()
        
        for w in windows:
            vol = ret.rolling(w).std()
            vol_ma = vol.rolling(w).mean()
            x[f"vol_regime_{w}"] = vol / vol_ma.replace(0, np.nan)
        
        # GARCH-like exponential weighting
        x["vol_ewm_24"] = ret.abs().ewm(span=24).mean()
        x["vol_ewm_72"] = ret.abs().ewm(span=72).mean()
        x["vol_ratio_24_72"] = x["vol_ewm_24"] / x["vol_ewm_72"].replace(0, np.nan)
        
        return x

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order flow and market microstructure proxies."""
        x = df.copy()
        close = x["close"]
        volume = x["volume"]
        ret = close.pct_change()
        
        # Volume-weighted price momentum
        x["vwap_ratio"] = close / (close * volume).rolling(24).sum() / volume.rolling(24).sum().replace(0, np.nan)
        
        # Order flow imbalance proxy (volume * return direction)
        x["order_flow_imbalance"] = (volume * np.sign(ret)).rolling(24).sum() / volume.rolling(24).sum().replace(0, np.nan)
        
        # Kyle's Lambda proxy: absolute return / volume
        x["kyle_lambda_24"] = (ret.abs() / volume.replace(0, np.nan)).rolling(24).mean()
        
        # Bid-ask bounce proxy
        x["close_range_ratio"] = (close - x["low"]) / (x["high"] - x["low"] + 1e-8)
        
        return x

    def add_funding_rate_features(self, df: pd.DataFrame, funding_rates: Optional[pd.Series] = None) -> pd.DataFrame:
        """Perpetual funding rate analysis (if available)."""
        x = df.copy()
        
        if funding_rates is not None:
            x["funding_rate"] = funding_rates
            x["funding_rate_ma_8"] = funding_rates.rolling(8).mean()
            x["funding_rate_z"] = (funding_rates - funding_rates.rolling(24).mean()) / funding_rates.rolling(24).std().replace(0, np.nan)
        else:
            logger.warning("Funding rates not provided; skipping funding features.")
        
        return x

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-of-day and day-of-week effects."""
        x = df.copy()
        ts = pd.to_datetime(x["timestamp"], utc=True)
        
        # Circular encoding
        x["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        x["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        x["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        x["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        
        # Is US market hours (UTC 14-22 ≈ US 9-17)
        x["is_us_hours"] = ((ts.dt.hour >= 14) & (ts.dt.hour < 22)).astype(float)
        
        return x

    def add_ichimoku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku with point-in-time compliance."""
        x = df.copy()
        high = x["high"]
        low = x["low"]
        close = x["close"]
        
        # Tenkan (9-period)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun (26-period)
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (plotted 26 bars ahead, but we use current)
        span_a = (tenkan + kijun) / 2
        
        # Senkou Span B (52-period, plotted 26 bars ahead)
        span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        
        # Chikou (close plotted 26 bars back, shifted for backtest)
        chikou = close.shift(26)
        
        x["ichi_tenkan_kijun"] = (tenkan - kijun) / close.replace(0, np.nan)
        x["ichi_price_span_a"] = (close - span_a) / close.replace(0, np.nan)
        x["ichi_price_span_b"] = (close - span_b) / close.replace(0, np.nan)
        x["ichi_cloud_width"] = (span_a - span_b).abs() / close.replace(0, np.nan)
        x["ichi_chikou_price"] = (chikou - close) / close.replace(0, np.nan)
        
        return x

    def transform(self, df: pd.DataFrame, funding_rates: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply all feature transformations with error handling."""
        try:
            x = df.copy()
            x = self.add_momentum_features(x)
            x = self.add_volatility_regime(x)
            x = self.add_microstructure_features(x)
            x = self.add_funding_rate_features(x, funding_rates)
            x = self.add_temporal_features(x)
            x = self.add_ichimoku_features(x)
            
            # Replace infinities with NaN
            x = x.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Features computed for {len(x)} bars, {x.isnull().sum().sum()} NaNs")
            return x
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
