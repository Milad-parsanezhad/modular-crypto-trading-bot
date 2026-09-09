from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import ccxt
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.cost_regime_v18 import V18Config, run_cost_aware_conversion, run_external_regime_replication


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= now].reset_index(drop=True)


def fetch_coinex(symbols: list[str], bars: int) -> dict[str, pd.DataFrame]:
    out = {}
    end_ms = utc_now_ms()
    for symbol in symbols:
        df = fetch_coinex_klines(symbol=symbol, period="4hour", market_type="spot", end_ms=end_ms, bars=bars)
        df = _drop_incomplete(df, PERIOD_MS["4hour"])
        if not df.empty:
            out[symbol] = df
    return out


def fetch_ccxt_external(exchange_id: str, symbols: list[str], bars: int, timeframe: str = "4h") -> dict[str, pd.DataFrame]:
    klass = getattr(ccxt, exchange_id)
    exchange = klass({"enableRateLimit": True})
    exchange.load_markets()
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    now_ms = exchange.milliseconds()
    since = now_ms - int((bars + 50) * step_ms)
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in exchange.markets:
            continue
        rows = []
        cursor = since
        loops = 0
        while cursor < now_ms and len(rows) < bars + 50 and loops < 20:
            loops += 1
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=min(300, bars + 50 - len(rows)))
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1][0])
            nxt = last + step_ms
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(exchange.rateLimit / 1000.0 if exchange.rateLimit else 0.05)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").sort_values("timestamp")
        df = _drop_incomplete(df, step_ms).tail(bars).reset_index(drop=True)
        if not df.empty:
            out[symbol] = df
    exchange.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/v18/v18_cost_regime_results.json")
    p.add_argument("--bars", type=int, default=1800)
    p.add_argument("--external-exchange", default="okx")
    args = p.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT"]
    cfg = V18Config()
    coinex = fetch_coinex(symbols, args.bars)
    if len(coinex) < 2:
        raise RuntimeError(f"CoinEx dataset incomplete: {sorted(coinex)}")
    exp_a = run_cost_aware_conversion(coinex, cfg)

    external_exchange = args.external_exchange
    try:
        external = fetch_ccxt_external(external_exchange, symbols, args.bars)
        if len(external) < 2:
            raise RuntimeError(f"incomplete external dataset: {sorted(external)}")
    except Exception as first_error:
        fallback = "kucoin" if external_exchange != "kucoin" else "okx"
        external = fetch_ccxt_external(fallback, symbols, args.bars)
        if len(external) < 2:
            raise RuntimeError(f"external venue fetch failed: primary={first_error}; fallback={sorted(external)}")
        external_exchange = fallback

    exp_b = run_external_regime_replication(external, cfg)
    payload = {
        "version": "v0.18",
        "research_status": "COST_AWARE_AND_EXTERNAL_REGIME_REPLICATION_NOT_LIVE_SIGNAL",
        "experiment_a_source": "CoinEx public spot 4h",
        "experiment_b_external_source": f"{external_exchange} public spot 4h via ccxt",
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "combined_decision": {
            "cost_aware_supported": exp_a["decision"] == "COST_AWARE_CONVERSION_SUPPORTED",
            "external_regime_supported": exp_b["decision"] == "EXTERNAL_REGIME_REPLICATION_SUPPORTED",
            "live_execution_authorized": False,
            "paper_strategy_promotion_authorized": False,
            "next_gate": "search-aware audit plus prospective replication if either hypothesis is supported",
        },
        "limitations": [
            "Experiment A freezes a simple Ridge forecast within this protocol; it is not the v0.10 Logistic candidate.",
            "Experiment B is an external-venue replication, but crypto venue prices are economically correlated and therefore not a fully independent asset class.",
            "No result authorizes real-money execution.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "experiment_a_decision": exp_a["decision"],
        "experiment_b_decision": exp_b["decision"],
        "external_exchange": external_exchange,
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
