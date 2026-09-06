from __future__ import annotations

import pandas as pd


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ret = x["close"].pct_change(); trend = x["close"].pct_change(24); vol = ret.rolling(24).std()
    vol_hi = vol.rolling(180, min_periods=60).quantile(0.70).shift(1)
    trend_abs_ref = trend.abs().rolling(180, min_periods=60).median().shift(1)
    x["regime_high_vol"] = (vol > vol_hi).astype(float)
    x["regime_trend_up"] = ((trend > trend_abs_ref) & trend.notna()).astype(float)
    x["regime_trend_down"] = ((trend < -trend_abs_ref) & trend.notna()).astype(float)
    x["regime_range"] = ((x["regime_trend_up"] == 0) & (x["regime_trend_down"] == 0)).astype(float)
    return x

REGIME_COLUMNS = ["regime_high_vol","regime_trend_up","regime_trend_down","regime_range"]
