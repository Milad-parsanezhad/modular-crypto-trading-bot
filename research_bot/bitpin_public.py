from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_BITPIN_API_BASE_URL = "https://api.bitpin.market/api/v1"
BITPIN_API_BASE_URL = os.getenv("BITPIN_API_BASE_URL", DEFAULT_BITPIN_API_BASE_URL).rstrip("/")

RequestFn = Callable[[str, int], Any]


class BitpinAPIError(RuntimeError):
    """Raised when Bitpin returns an HTTP/API/contract error."""


def _request_json(path: str, timeout: int = 20) -> Any:
    """GET a public Bitpin endpoint without credentials.

    The API host is configurable because Bitpin has used more than one public
    API hostname. No Authorization header is ever attached by this module.
    """
    url = f"{BITPIN_API_BASE_URL}/{path.lstrip('/')}"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "modular-crypto-research-bot/bitpin-readonly-1",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        raise BitpinAPIError(f"Bitpin HTTP {exc.code} for {path}") from exc
    except URLError as exc:
        raise BitpinAPIError(f"Bitpin network error for {path}: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BitpinAPIError(f"Bitpin returned invalid JSON for {path}") from exc


def _to_market(symbol: str) -> str:
    """Normalize BTC/USDT, BTC-USDT or BTC_USDT to Bitpin's BTC_USDT form."""
    cleaned = symbol.strip().upper().replace("-", "_").replace("/", "_")
    parts = [p for p in cleaned.split("_") if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid Bitpin symbol: {symbol!r}")
    return f"{parts[0]}_{parts[1]}"


def _as_rows(payload: Any, *, endpoint: str) -> list[dict[str, Any]]:
    """Accept common list/paginated response envelopes and fail closed otherwise."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("results", "data", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise BitpinAPIError(f"Unexpected Bitpin response contract for {endpoint}")
    else:
        raise BitpinAPIError(f"Unexpected Bitpin response type for {endpoint}")

    if not all(isinstance(row, dict) for row in rows):
        raise BitpinAPIError(f"Non-object row in Bitpin response for {endpoint}")
    return rows


def _finite_positive(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BitpinAPIError(f"Invalid numeric {field}: {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise BitpinAPIError(f"Non-positive/non-finite {field}: {value!r}")
    return out


def _timestamp_utc(value: Any) -> datetime:
    """Parse unix seconds/ms or ISO timestamps; receipt time is a last-resort fallback."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        x = float(value)
        if x > 1e12:
            x /= 1000.0
        return datetime.fromtimestamp(x, tz=timezone.utc)
    text = str(value).strip()
    if text:
        try:
            num = float(text)
        except ValueError:
            num = None
        if num is not None:
            if num > 1e12:
                num /= 1000.0
            return datetime.fromtimestamp(num, tz=timezone.utc)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    raise BitpinAPIError(f"Unparseable Bitpin timestamp: {value!r}")


def fetch_bitpin_currencies(*, request_fn: RequestFn = _request_json) -> pd.DataFrame:
    endpoint = "/mkt/currencies/"
    rows = _as_rows(request_fn(endpoint, 20), endpoint=endpoint)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_bitpin_markets(*, request_fn: RequestFn = _request_json) -> pd.DataFrame:
    endpoint = "/mkt/markets/"
    rows = _as_rows(request_fn(endpoint, 20), endpoint=endpoint)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def fetch_bitpin_tickers(*, request_fn: RequestFn = _request_json) -> pd.DataFrame:
    endpoint = "/mkt/tickers/"
    rows = _as_rows(request_fn(endpoint, 20), endpoint=endpoint)
    if not rows:
        return pd.DataFrame(columns=["symbol", "price", "low", "high", "timestamp"])

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns or "price" not in df.columns:
        raise BitpinAPIError("Bitpin ticker response missing symbol/price")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    for col in ("price", "low", "high", "daily_change_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if (~df["price"].map(lambda x: math.isfinite(float(x)) and float(x) > 0 if pd.notna(x) else False)).any():
        raise BitpinAPIError("Bitpin ticker response contains invalid prices")
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].map(_timestamp_utc)
    return df


def _level_pair(level: Any, *, side: str) -> tuple[float, float]:
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        price, qty = level[0], level[1]
    elif isinstance(level, dict):
        price = level.get("price")
        qty = level.get("amount", level.get("quantity", level.get("volume")))
    else:
        raise BitpinAPIError(f"Malformed {side} orderbook level")
    return _finite_positive(price, field=f"{side}.price"), _finite_positive(qty, field=f"{side}.quantity")


def fetch_bitpin_orderbook(
    symbol: str = "BTC/USDT",
    *,
    depth: int = 20,
    request_fn: RequestFn = _request_json,
) -> dict[str, Any]:
    """Fetch and validate a public Bitpin order book.

    Endpoint is intentionally isolated here so a future Bitpin deprecation can be
    changed without touching execution logic.
    """
    if int(depth) <= 0:
        raise ValueError("depth must be positive")
    market = _to_market(symbol)
    endpoint = f"/mth/orderbook/{quote(market, safe='_')}/"
    payload = request_fn(endpoint, 20)
    if not isinstance(payload, dict):
        raise BitpinAPIError("Bitpin orderbook response must be an object")
    book = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    bids_raw = (book or {}).get("bids") or []
    asks_raw = (book or {}).get("asks") or []
    if not bids_raw or not asks_raw:
        raise BitpinAPIError(f"Bitpin orderbook unavailable for {market}")

    bids = [_level_pair(x, side="bid") for x in bids_raw]
    asks = [_level_pair(x, side="ask") for x in asks_raw]
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    bids = bids[: int(depth)]
    asks = asks[: int(depth)]
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_ask < best_bid:
        raise BitpinAPIError(f"Crossed Bitpin book for {market}: bid={best_bid} ask={best_ask}")

    ts_value = None
    if isinstance(book, dict):
        ts_value = book.get("updated_at", book.get("timestamp", book.get("created_at")))
    return {
        "symbol": symbol,
        "market": market,
        "timestamp": _timestamp_utc(ts_value),
        "bids": bids,
        "asks": asks,
    }


def fetch_bitpin_recent_trades(
    symbol: str = "BTC/USDT",
    *,
    request_fn: RequestFn = _request_json,
) -> pd.DataFrame:
    market = _to_market(symbol)
    endpoint = f"/mth/matches/{quote(market, safe='_')}/"
    rows = _as_rows(request_fn(endpoint, 20), endpoint=endpoint)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "id", "symbol", "side", "price", "base_amount", "quote_amount"])
    df = pd.DataFrame(rows)
    if "price" not in df.columns:
        raise BitpinAPIError("Bitpin trade response missing price")
    for col in ("price", "base_amount", "quote_amount", "commission"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    ts_col = "created_at" if "created_at" in df.columns else "timestamp" if "timestamp" in df.columns else None
    if ts_col is not None:
        df["timestamp"] = df[ts_col].map(_timestamp_utc)
    else:
        df["timestamp"] = datetime.now(timezone.utc)
    if "symbol" not in df.columns:
        df["symbol"] = market
    return df.sort_values("timestamp").reset_index(drop=True)
