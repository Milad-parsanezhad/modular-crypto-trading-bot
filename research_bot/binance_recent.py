from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def fetch_binance_recent_open_interest(
    symbol: str = "BTCUSDT",
    period: str = "4h",
    limit: int = 180,
    timeout: int = 20,
) -> pd.DataFrame:
    """Optional cross-venue recent OI series from Binance USD-M.

    This series is explicitly labeled as Binance data and is never relabeled as
    CoinEx open interest. Binance's REST historical-OI endpoint is recent-history
    only, so it is an auxiliary feature rather than a long-history solution.
    """
    params = urlencode({"symbol": symbol, "period": period, "limit": min(int(limit), 500)})
    url = f"https://fapi.binance.com/futures/data/openInterestHist?{params}"
    req = Request(url, headers={"User-Agent": "modular-crypto-research-bot/0.3"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance OI error: {payload}")
    if not payload:
        return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value", "source"])
    df = pd.DataFrame(payload)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["open_interest_value"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["source"] = "binance_usdm_recent"
    return df[["timestamp", "open_interest", "open_interest_value", "source"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
