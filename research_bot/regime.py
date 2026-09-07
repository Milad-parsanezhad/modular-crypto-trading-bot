from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime detection features."""
    x = df.copy()
    ret = x["close"].pct_change()
    
    # Rolling volatility
    x["vol_24"] = ret.rolling(24).std()
    x["vol_72"] = ret.rolling(72).std()
    
    # Volatility regime
    x["vol_ratio"] = x["vol_24"] / x["vol_72"].replace(0, np.nan)
    
    # Returns skew
    x["skew_24"] = ret.rolling(24).skew()
    
    # Trend (using moving averages)
    sma_12 = x["close"].rolling(12).mean()
    sma_48 = x["close"].rolling(48).mean()
    x["trend_12_48"] = (sma_12 - sma_48) / sma_48.replace(0, np.nan)
    
    return x.replace([np.inf, -np.inf], np.nan)


REGIME_COLUMNS = [
    "vol_24", "vol_72", "vol_ratio",
    "skew_24",
    "trend_12_48",
]
