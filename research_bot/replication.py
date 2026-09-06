from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility_threshold(close: pd.Series, span: int = 50, multiplier: float = 1.0, min_periods: int = 20) -> pd.Series:
    """Lagged point-in-time EWMA volatility threshold."""
    c = pd.Series(close, copy=False).astype(float)
    log_ret = np.log(c).diff()
    vol = log_ret.ewm(span=span, min_periods=min_periods, adjust=False).std(bias=False)
    return (vol * float(multiplier)).shift(1)


def symmetric_cusum_events(close: pd.Series, threshold: float | pd.Series) -> pd.Index:
    """Symmetric CUSUM event filter on log returns; never fills future values."""
    c = pd.Series(close, copy=False).astype(float)
    log_ret = np.log(c).diff()
    thr = pd.Series(float(threshold), index=c.index) if np.isscalar(threshold) else pd.Series(threshold, index=c.index).astype(float)
    s_pos = 0.0
    s_neg = 0.0
    events = []
    for idx in c.index[1:]:
        r, h = log_ret.loc[idx], thr.loc[idx]
        if not np.isfinite(r) or not np.isfinite(h) or h <= 0:
            continue
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos > h:
            s_pos = 0.0
            events.append(idx)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(idx)
    return pd.Index(events)


def triple_barrier_events(df: pd.DataFrame, event_index: pd.Index, volatility: pd.Series, horizon_bars: int = 12, profit_take_mult: float = 1.5, stop_loss_mult: float = 1.0, min_volatility: float = 1e-6, allow_overlap: bool = False) -> pd.DataFrame:
    """Long-side triple-barrier outcomes with ambiguity/no-overlap controls."""
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    index_values = list(df.index)
    pos = {idx: i for i, idx in enumerate(index_values)}
    vol = pd.Series(volatility, index=df.index).astype(float)
    records = []
    blocked_until = -1
    for event in event_index:
        if event not in pos:
            continue
        i = pos[event]
        if not allow_overlap and i <= blocked_until:
            continue
        if i >= len(df) - 1:
            continue
        v = vol.loc[event]
        if not np.isfinite(v) or v < min_volatility:
            continue
        entry = float(df.iloc[i]["close"])
        pt = entry * (1.0 + profit_take_mult * v)
        sl = entry * (1.0 - stop_loss_mult * v)
        last = min(i + horizon_bars, len(df) - 1)
        label, exit_i, exit_price, barrier, ambiguous = 0, last, float(df.iloc[last]["close"]), "vertical", False
        for j in range(i + 1, last + 1):
            high, low = float(df.iloc[j]["high"]), float(df.iloc[j]["low"])
            hit_pt, hit_sl = high >= pt, low <= sl
            if hit_pt and hit_sl:
                label, exit_i, exit_price, barrier, ambiguous = 0, j, float(df.iloc[j]["close"]), "ambiguous", True
                break
            if hit_pt:
                label, exit_i, exit_price, barrier = 1, j, pt, "profit"
                break
            if hit_sl:
                label, exit_i, exit_price, barrier = -1, j, sl, "stop"
                break
        if not allow_overlap:
            blocked_until = exit_i
        records.append({
            "event_index": event,
            "entry_position": i,
            "exit_position": exit_i,
            "exit_index": df.index[exit_i],
            "entry_price": entry,
            "exit_price": float(exit_price),
            "volatility": float(v),
            "profit_barrier": float(pt),
            "stop_barrier": float(sl),
            "label": int(label),
            "barrier": barrier,
            "ambiguous": bool(ambiguous),
            "event_return": float(exit_price / entry - 1.0),
            "holding_bars": int(exit_i - i),
        })
    return pd.DataFrame.from_records(records)


class PurgedWalkForwardSplit:
    """Expanding walk-forward splitter with purge+embargo gap."""
    def __init__(self, n_splits: int = 5, min_train_size: int = 300, test_size: int | None = None, purge_bars: int = 12, embargo_bars: int = 3):
        self.n_splits = int(n_splits)
        self.min_train_size = int(min_train_size)
        self.test_size = None if test_size is None else int(test_size)
        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)

    def split(self, X):
        n = len(X)
        gap = self.purge_bars + self.embargo_bars
        available = n - self.min_train_size - gap
        if available <= self.n_splits:
            raise ValueError("Not enough observations for requested walk-forward split")
        test_size = self.test_size or max(1, available // self.n_splits)
        produced, train_end = 0, self.min_train_size
        while produced < self.n_splits:
            test_start = train_end + gap
            test_end = min(test_start + test_size, n)
            if test_end <= test_start:
                break
            yield np.arange(0, train_end, dtype=int), np.arange(test_start, test_end, dtype=int)
            produced += 1
            train_end = test_end
            if train_end + gap >= n:
                break
