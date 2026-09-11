from __future__ import annotations

import time
import pandas as pd


def fetch_ccxt_spot_ohlcv(
    exchange_id: str,
    symbol: str,
    *,
    timeframe: str = "4h",
    start_ms: int,
    end_ms: int,
    max_bars: int = 5000,
    page_limit: int = 300,
    max_pages: int = 30,
) -> pd.DataFrame:
    """Fetch a bounded public spot OHLCV window through CCXT.

    This helper is used only as a pre-registered technical source fallback after
    a primary external venue is blocked. It never changes the symbol universe or
    model based on outcomes. The returned frame is sorted and clipped to the
    exact requested calendar interval.
    """
    import ccxt

    cls = getattr(ccxt, exchange_id)
    exchange = cls({"enableRateLimit": True, "timeout": 20_000})
    try:
        markets = exchange.load_markets()
        if symbol not in markets:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "source"])
        market = markets[symbol]
        if market.get("spot") is False:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "source"])
        cursor = int(start_ms)
        rows: list[list[float]] = []
        seen: set[int] = set()
        for _ in range(max_pages):
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=page_limit)
            if not batch:
                break
            progressed = False
            last_ts = cursor
            for row in batch:
                ts = int(row[0]); last_ts = max(last_ts, ts)
                if ts > int(end_ms):
                    continue
                if ts >= int(start_ms) and ts not in seen:
                    seen.add(ts); rows.append(row); progressed = True
            if last_ts >= int(end_ms) or len(rows) >= int(max_bars):
                break
            next_cursor = last_ts + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if not progressed and batch[-1][0] < start_ms:
                break
            time.sleep(max(float(getattr(exchange, "rateLimit", 0)) / 1000.0, 0.01))
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "source"])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source"] = f"ccxt_{exchange_id}_public_spot"
    return df.drop_duplicates("timestamp").sort_values("timestamp").tail(int(max_bars)).reset_index(drop=True)
