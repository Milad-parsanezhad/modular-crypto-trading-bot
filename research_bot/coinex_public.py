from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


COINEX_BASE_URL = "https://api.coinex.com/v2"

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
    "3day": 259_200_000,
    "1week": 604_800_000,
}


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _get_json(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    url = f"{COINEX_BASE_URL}{path}" + (f"?{query}" if query else "")
    req = Request(url, headers={"User-Agent": "modular-crypto-research-bot/0.3"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("code") != 0:
        raise RuntimeError(f"CoinEx API error for {path}: {payload}")
    return payload


def _to_market(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def fetch_coinex_klines(
    symbol: str = "BTC/USDT",
    period: str = "4hour",
    market_type: str = "spot",
    start_ms: int | None = None,
    end_ms: int | None = None,
    bars: int = 3000,
    request_fn: Callable[[str, dict | None, int], dict] = _get_json,
) -> pd.DataFrame:
    """Fetch long-history CoinEx klines using explicit time windows."""
    if period not in PERIOD_MS:
        raise ValueError(f"Unsupported CoinEx period: {period}")
    if market_type not in {"spot", "futures"}:
        raise ValueError("market_type must be 'spot' or 'futures'")

    step = PERIOD_MS[period]
    end_ms = int(end_ms or utc_now_ms())
    if start_ms is None:
        start_ms = end_ms - int(max(1, bars) * step * 1.02)
    start_ms = int(start_ms)
    path = "/spot/kline" if market_type == "spot" else "/futures/kline"
    market = _to_market(symbol)

    rows: list[dict] = []
    cursor = start_ms
    chunk_bars = 900
    max_loops = max(5, int(np.ceil(max(1, bars) / chunk_bars)) + 5)
    loops = 0
    while cursor <= end_ms and loops < max_loops:
        loops += 1
        chunk_end = min(end_ms, cursor + (chunk_bars - 1) * step)
        payload = request_fn(
            path,
            {
                "market": market,
                "period": period,
                "limit": 1000,
                "start_time": cursor,
                "end_time": chunk_end,
            },
            20,
        )
        data = payload.get("data") or []
        rows.extend(data)
        if data:
            last_ts = max(int(r["created_at"]) for r in data)
            cursor = max(chunk_end + 1, last_ts + step)
        else:
            cursor = chunk_end + 1
        time.sleep(0.03)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "value", "market"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["timestamp", "open", "high", "low", "close", "volume", "value", "market"]
    df = df[keep].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if bars:
        df = df.tail(int(bars)).reset_index(drop=True)
    return df


def audit_kline_gaps(df: pd.DataFrame, period: str) -> dict:
    if df.empty:
        return {"rows": 0, "gap_count": 0, "coverage_start": None, "coverage_end": None}
    step = pd.Timedelta(milliseconds=PERIOD_MS[period])
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    deltas = ts.diff().dropna()
    gap_mask = deltas > step * 1.5
    missing_est = int(sum(max(0, round(d / step) - 1) for d in deltas[gap_mask]))
    return {
        "rows": int(len(df)),
        "gap_count": int(gap_mask.sum()),
        "estimated_missing_bars": missing_est,
        "coverage_start": ts.iloc[0].isoformat(),
        "coverage_end": ts.iloc[-1].isoformat(),
        "median_spacing_seconds": float(deltas.median().total_seconds()) if len(deltas) else None,
    }


def fetch_coinex_funding_history(
    symbol: str = "BTC/USDT",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 100,
    max_pages: int = 100,
    request_fn: Callable[[str, dict | None, int], dict] = _get_json,
) -> pd.DataFrame:
    market = _to_market(symbol)
    end_ms = int(end_ms or utc_now_ms())
    start_ms = int(start_ms or (end_ms - 365 * 86_400_000))
    rows = []
    page = 1
    while page <= max_pages:
        payload = request_fn(
            "/futures/funding-rate-history",
            {
                "market": market,
                "start_time": start_ms,
                "end_time": end_ms,
                "page": page,
                "limit": int(limit),
            },
            20,
        )
        data = payload.get("data") or []
        rows.extend(data)
        has_next = bool((payload.get("pagination") or {}).get("has_next"))
        if not has_next or not data:
            break
        page += 1
        time.sleep(0.03)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "actual_funding_rate", "theoretical_funding_rate", "market"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    for col in ["actual_funding_rate", "theoretical_funding_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["timestamp", "actual_funding_rate", "theoretical_funding_rate", "market"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_coinex_basis_history(
    symbol: str = "BTC/USDT",
    start_ms: int | None = None,
    end_ms: int | None = None,
    window_days: int = 7,
    request_fn: Callable[[str, dict | None, int], dict] = _get_json,
) -> pd.DataFrame:
    market = _to_market(symbol)
    end_ms = int(end_ms or utc_now_ms())
    start_ms = int(start_ms or (end_ms - 30 * 86_400_000))
    rows = []
    cursor = start_ms
    window_ms = int(window_days * 86_400_000)
    while cursor <= end_ms:
        chunk_end = min(end_ms, cursor + window_ms - 1)
        payload = request_fn(
            "/futures/basis-history",
            {"market": market, "start_time": cursor, "end_time": chunk_end},
            20,
        )
        rows.extend(payload.get("data") or [])
        cursor = chunk_end + 1
        time.sleep(0.03)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "basis_rate", "market"])
    df = pd.DataFrame(rows)
    ts_col = "created_at" if "created_at" in df.columns else "funding_time"
    df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df["basis_rate"] = pd.to_numeric(df["basis_rate"], errors="coerce")
    return df[["timestamp", "basis_rate", "market"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_coinex_open_interest_snapshot(
    symbol: str = "BTC/USDT",
    request_fn: Callable[[str, dict | None, int], dict] = _get_json,
) -> dict:
    market = _to_market(symbol)
    payload = request_fn("/futures/market", {"market": market}, 20)
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"No CoinEx futures market data for {market}")
    row = data[0]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "open_interest_volume": float(row.get("open_interest_volume") or np.nan),
        "maker_fee_rate": float(row.get("maker_fee_rate") or np.nan),
        "taker_fee_rate": float(row.get("taker_fee_rate") or np.nan),
        "contract_type": row.get("contract_type"),
        "source": "coinex_public_snapshot",
    }


def fetch_coinex_market_deals(
    symbol: str = "BTC/USDT",
    market_type: str = "spot",
    pages: int = 5,
    limit: int = 1000,
    request_fn: Callable[[str, dict | None, int], dict] = _get_json,
) -> pd.DataFrame:
    """Fetch recent public trades and page backward by deal id when supported."""
    if market_type not in {"spot", "futures"}:
        raise ValueError("market_type must be 'spot' or 'futures'")
    market = _to_market(symbol)
    path = "/spot/deals" if market_type == "spot" else "/futures/deals"
    rows = []
    last_id = 0
    seen: set[int] = set()

    for _ in range(max(1, int(pages))):
        payload = request_fn(path, {"market": market, "limit": min(int(limit), 1000), "last_id": last_id}, 20)
        data = payload.get("data") or []
        fresh = []
        for r in data:
            deal_id = int(r["deal_id"])
            if deal_id not in seen:
                seen.add(deal_id)
                fresh.append(r)
        rows.extend(fresh)
        if not fresh:
            break
        ids = [int(r["deal_id"]) for r in fresh]
        next_last_id = min(ids) - 1
        if next_last_id <= 0 or next_last_id == last_id:
            break
        last_id = next_last_id
        time.sleep(0.03)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "deal_id", "side", "price", "amount", "notional", "market"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
    for col in ["price", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["notional"] = df["price"] * df["amount"]
    return df[["timestamp", "deal_id", "side", "price", "amount", "notional", "market"]].drop_duplicates("deal_id").sort_values("timestamp").reset_index(drop=True)
