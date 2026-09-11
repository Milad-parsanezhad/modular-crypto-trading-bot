from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BYBIT_BASE_URL = "https://api.bybit.com"

BYBIT_INTERVALS = {
    "1min": "1",
    "3min": "3",
    "5min": "5",
    "15min": "15",
    "30min": "30",
    "1hour": "60",
    "2hour": "120",
    "4hour": "240",
    "6hour": "360",
    "12hour": "720",
    "1day": "D",
    "1week": "W",
}

PERIOD_MS = {
    "1min": 60_000,
    "3min": 180_000,
    "5min": 300_000,
    "15min": 900_000,
    "30min": 1_800_000,
    "1hour": 3_600_000,
    "2hour": 7_200_000,
    "4hour": 14_400_000,
    "6hour": 21_600_000,
    "12hour": 43_200_000,
    "1day": 86_400_000,
    "1week": 604_800_000,
}


def _get_json(path: str, params: dict, timeout: int = 20) -> dict:
    url = f"{BYBIT_BASE_URL}{path}?{urlencode({k:v for k,v in params.items() if v is not None})}"
    req = Request(url, headers={"User-Agent": "modular-crypto-research-bot/0.24c"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error for {path}: {payload}")
    return payload


def _symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def fetch_bybit_spot_ohlcv(
    symbol: str = "BTC/USDT",
    period: str = "4hour",
    start_ms: int | None = None,
    end_ms: int | None = None,
    bars: int = 3600,
    limit: int = 1000,
    max_pages: int = 20,
) -> pd.DataFrame:
    """Fetch Bybit public spot klines using deterministic backward pagination.

    The function is intended for external-venue research replication. It returns
    only the exchange OHLCV/turnover fields and never fabricates order-flow data.
    Bybit returns klines newest-first; this function normalizes them to ascending
    UTC timestamps and removes duplicates.
    """
    if period not in BYBIT_INTERVALS:
        raise ValueError(f"Unsupported Bybit period: {period}")
    if bars <= 0:
        raise ValueError("bars must be positive")
    step = PERIOD_MS[period]
    end_cursor = int(end_ms if end_ms is not None else pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    target_start = int(start_ms) if start_ms is not None else end_cursor - int(bars * step * 1.05)
    rows: list[list[str]] = []
    seen: set[int] = set()

    for _ in range(max_pages):
        payload = _get_json(
            "/v5/market/kline",
            {
                "category": "spot",
                "symbol": _symbol(symbol),
                "interval": BYBIT_INTERVALS[period],
                "start": target_start,
                "end": end_cursor,
                "limit": min(int(limit), 1000),
            },
            20,
        )
        data = (payload.get("result") or {}).get("list") or []
        if not data:
            break
        earliest = None
        for row in data:
            ts = int(row[0])
            earliest = ts if earliest is None else min(earliest, ts)
            if ts not in seen:
                seen.add(ts)
                rows.append(row)
        if earliest is None or earliest <= target_start or len(rows) >= bars:
            break
        next_end = earliest - 1
        if next_end >= end_cursor:
            break
        end_cursor = next_end
        time.sleep(0.04)

    columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    if not rows:
        return pd.DataFrame(columns=columns + ["market", "source"])
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market"] = _symbol(symbol)
    df["source"] = "bybit_public_spot"
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if start_ms is not None:
        df = df[df["timestamp"] >= pd.to_datetime(int(start_ms), unit="ms", utc=True)]
    if end_ms is not None:
        df = df[df["timestamp"] <= pd.to_datetime(int(end_ms), unit="ms", utc=True)]
    return df.tail(int(bars)).reset_index(drop=True)


def audit_bybit_ohlcv(df: pd.DataFrame, period: str) -> dict:
    if period not in PERIOD_MS:
        raise ValueError(f"Unsupported Bybit period: {period}")
    if df.empty:
        return {
            "rows": 0,
            "gap_count": 0,
            "estimated_missing_bars": 0,
            "coverage_start": None,
            "coverage_end": None,
        }
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values().drop_duplicates()
    step = pd.Timedelta(milliseconds=PERIOD_MS[period])
    delta = ts.diff().dropna()
    gaps = delta > step * 1.5
    missing = int(sum(max(0, round(d / step) - 1) for d in delta[gaps]))
    return {
        "rows": int(len(ts)),
        "gap_count": int(gaps.sum()),
        "estimated_missing_bars": missing,
        "coverage_start": ts.iloc[0].isoformat(),
        "coverage_end": ts.iloc[-1].isoformat(),
        "median_spacing_seconds": float(delta.median().total_seconds()) if len(delta) else None,
    }


def fetch_bybit_open_interest_history(
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 200,
    max_pages: int = 150,
) -> pd.DataFrame:
    """Fetch public Bybit linear-perpetual OI history with cursor pagination.

    This is an explicitly cross-venue feature. It must never be relabeled as
    CoinEx OI. Bybit documents history availability back to symbol launch.
    """
    rows = []
    cursor = None
    seen = set()
    for _ in range(max_pages):
        payload = _get_json(
            "/v5/market/open-interest",
            {
                "category": "linear",
                "symbol": symbol.upper(),
                "intervalTime": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(int(limit), 200),
                "cursor": cursor,
            },
            20,
        )
        result = payload.get("result") or {}
        data = result.get("list") or []
        for row in data:
            key = str(row.get("timestamp"))
            if key not in seen:
                seen.add(key)
                rows.append(row)
        next_cursor = result.get("nextPageCursor") or ""
        if not data or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.03)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open_interest", "single_open_interest", "source"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    if "singleOpenInterest" in df.columns:
        df["single_open_interest"] = pd.to_numeric(df["singleOpenInterest"], errors="coerce")
    else:
        df["single_open_interest"] = pd.NA
    df["source"] = "bybit_linear_public"
    return df[["timestamp", "open_interest", "single_open_interest", "source"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
