from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

from .coinex_public import _get_json, _to_market


@dataclass(frozen=True)
class DepthSnapshot:
    symbol: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float
    bid_depth: float
    ask_depth: float
    imbalance: float


def fetch_coinex_depth(symbol: str, limit: int = 5, interval: str = "0.01") -> DepthSnapshot:
    market = _to_market(symbol)
    payload = _get_json(
        "/spot/depth",
        {"market": market, "limit": int(limit), "interval": interval},
        20,
    )
    data = payload.get("data") or {}
    depth = data.get("depth") or {}
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    if not bids or not asks:
        raise RuntimeError(f"CoinEx depth unavailable for {market}")
    bid = float(bids[0][0])
    ask = float(asks[0][0])
    if bid <= 0 or ask <= 0 or ask < bid:
        raise RuntimeError(f"Invalid CoinEx depth for {market}: bid={bid} ask={ask}")
    mid = (bid + ask) / 2.0
    bid_depth = float(sum(float(x[1]) for x in bids))
    ask_depth = float(sum(float(x[1]) for x in asks))
    denom = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / denom if denom > 0 else 0.0
    ts_ms = (depth.get("updated_at") or data.get("updated_at"))
    ts = (
        datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        if ts_ms is not None
        else datetime.now(timezone.utc)
    )
    return DepthSnapshot(
        symbol=symbol,
        timestamp=ts,
        best_bid=bid,
        best_ask=ask,
        mid=mid,
        spread_bps=float((ask - bid) / mid * 10_000.0),
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        imbalance=float(np.clip(imbalance, -1.0, 1.0)),
    )
