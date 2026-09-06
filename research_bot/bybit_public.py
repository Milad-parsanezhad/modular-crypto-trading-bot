from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BYBIT_BASE_URL = "https://api.bybit.com"


def _get_json(path: str, params: dict, timeout: int = 20) -> dict:
    url = f"{BYBIT_BASE_URL}{path}?{urlencode({k:v for k,v in params.items() if v is not None})}"
    req = Request(url, headers={"User-Agent": "modular-crypto-research-bot/0.3"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error for {path}: {payload}")
    return payload


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
