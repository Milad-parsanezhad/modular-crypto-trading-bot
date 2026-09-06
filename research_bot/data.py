from __future__ import annotations

from typing import Iterable, Optional
import time

import ccxt
import pandas as pd


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _timeframe_to_milliseconds(timeframe: str) -> int:
    mapping = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe for historical pagination: {timeframe}")
    return mapping[timeframe]


def _make_exchange(exchange_id: str):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown CCXT exchange: {exchange_id}")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"enableRateLimit": True})


def fetch_ohlcv(
    exchange_id: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    limit: int = 1500,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch public OHLCV without API keys.

    Pagination is conservative and deduplicates timestamps. `limit` is the
    desired number of final bars, not necessarily a single exchange request.
    """
    ex = _make_exchange(exchange_id)
    ex.load_markets()
    if symbol not in ex.markets:
        raise ValueError(f"{symbol} is not listed on {exchange_id}")

    batch_size = min(1000, max(100, limit))
    rows = []
    cursor = since
    if cursor is None and limit > batch_size:
        cursor = ex.milliseconds() - int(limit * _timeframe_to_milliseconds(timeframe) * 1.05)

    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch_size)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + 1
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < batch_size:
            break
        if ex.rateLimit:
            time.sleep(ex.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"No OHLCV returned by {exchange_id} for {symbol} {timeframe}")

    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.tail(limit).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_with_fallback(
    exchanges: Iterable[str],
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    limit: int = 1500,
) -> tuple[pd.DataFrame, str]:
    errors = {}
    for exchange_id in exchanges:
        try:
            return fetch_ohlcv(exchange_id, symbol=symbol, timeframe=timeframe, limit=limit), exchange_id
        except Exception as exc:
            errors[exchange_id] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"All public data sources failed: {errors}")
