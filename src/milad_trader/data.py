"""OHLCV ingestion. Index timestamps are UTC candle OPEN times."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def validate_candles(df, timeframe="4h", now=None):
    df = df.copy()
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df.pop("timestamp"), utc=True, errors="raise")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Require timezone-aware UTC candle-open timestamps")
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    if not set(OHLCV) <= set(df):
        raise ValueError("Missing OHLCV columns")
    if df.index.hasnans or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be increasing, unique and non-null")
    df = df[OHLCV].astype(float)
    if len(df) < 2 or not np.isfinite(df.values).all():
        raise ValueError("Require at least two finite candles")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    delta = pd.Timedelta(timeframe)
    if (df.index.asi8 % delta.value != 0).any():
        raise ValueError("Candle timestamps are not aligned to the timeframe")
    if not (df.index.to_series().diff().iloc[1:] == delta).all():
        raise ValueError("Missing or irregular candles; do not silently forward-fill")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone aware")
    if (df.index + delta > current).any():
        raise ValueError("Input includes unfinished or future candles")
    return df


def read_csv(path, timeframe="4h"):
    return validate_candles(pd.read_csv(path), timeframe)


def save_candles(df, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp")
    manifest = dict(metadata, rows=len(df), first_open=df.index[0].isoformat(),
                    last_open=df.index[-1].isoformat(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def synthetic_candles(rows=1600, timeframe="4h", seed=314):
    """Deterministic software fixture, explicitly not historical market data."""
    rng = np.random.default_rng(seed)
    regime = np.sin(np.arange(rows) / 100) * 0.001
    close = 30000 * np.exp(np.cumsum(regime + rng.normal(0, 0.007, rows)))
    opening = np.r_[close[0], close[:-1]]
    width = rng.uniform(0.002, 0.01, rows)
    frame = pd.DataFrame({"open": opening, "close": close,
                          "high": np.maximum(opening, close) * (1 + width),
                          "low": np.minimum(opening, close) * (1 - width),
                          "volume": rng.lognormal(6, 0.5, rows)},
                         index=pd.date_range("2020-01-01", periods=rows, freq=timeframe, tz="UTC"))
    return validate_candles(frame, timeframe)


def fetch_public(exchange_id, symbol, timeframe, since, until=None, max_pages=1000):
    """Only public CCXT methods; credentials and order endpoints are never used."""
    import ccxt
    if exchange_id not in {"coinex", "binance", "kraken"}:
        raise ValueError("Supported public adapters: coinex, binance, kraken")
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000})
    start = pd.Timestamp(since)
    end = pd.Timestamp.now(tz="UTC") if until is None else pd.Timestamp(until)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Require timezone-aware since < until")
    step = int(pd.Timedelta(timeframe).total_seconds() * 1000)
    cursor, end_ms = int(start.timestamp()*1000), int(end.timestamp()*1000)
    rows = []
    for _ in range(max_pages):
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
                break
            except (ccxt.NetworkError, ccxt.RateLimitExceeded):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            break
        fresh = [r for r in batch if r[0] >= cursor]
        if not fresh:
            raise ValueError("Exchange pagination made no progress")
        rows.extend(fresh)
        cursor = max(r[0] for r in fresh) + step
        if cursor >= end_ms:
            break
    else:
        raise ValueError("Download exceeded max_pages; request a smaller range")
    if not rows:
        raise ValueError("Exchange returned no data")
    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.loc[(frame.index >= start) & (frame.index + pd.Timedelta(timeframe) <= end)]
    frame = validate_candles(frame, timeframe)
    if frame.index[0] != start.ceil(timeframe):
        raise ValueError("Exchange did not return requested historical start")
    if frame.index[-1] + pd.Timedelta(timeframe) != end.floor(timeframe):
        raise ValueError("Exchange returned an incomplete historical range")
    return frame
