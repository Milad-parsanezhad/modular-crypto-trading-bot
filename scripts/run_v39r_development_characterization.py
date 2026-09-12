from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from research_bot.canonical_strategy_v39r import CanonicalConfig, canonical_score


VENUES = ("coinex", "okx", "kucoin")
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT")
TIMEFRAME = "4h"
START = "2025-01-01T00:00:00Z"
END = "2026-09-10T23:59:59Z"
BASE_ROUNDTRIP_BPS = 24.0
STRESS_ROUNDTRIP_BPS = 36.0
TARGET_R = 3.0
MAX_HOLD_BARS = 30
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_BLOCK = 20
SEED = 314


def _to_ms(ts: str) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _exchange(name: str):
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True, "timeout": 30_000})


def fetch_fixed_ohlcv(exchange, symbol: str) -> pd.DataFrame:
    start_ms = _to_ms(START)
    end_ms = _to_ms(END)
    step_ms = 4 * 60 * 60 * 1000
    since = start_ms
    rows: list[list[float]] = []
    stalls = 0
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1][0])
        nxt = last + step_ms
        if nxt <= since:
            stalls += 1
            if stalls >= 2:
                raise RuntimeError(f"pagination stalled: {exchange.id} {symbol} since={since}")
            nxt = since + step_ms
        else:
            stalls = 0
        since = nxt
        if last >= end_ms:
            break
        time.sleep(max(float(getattr(exchange, "rateLimit", 0)) / 1000.0, 0.01))

    if not rows:
        raise RuntimeError(f"no OHLCV: {exchange.id} {symbol}")
    x = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x["timestamp"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    start = pd.Timestamp(START)
    end = pd.Timestamp(END)
    x = x[(x["timestamp"] >= start) & (x["timestamp"] <= end)].reset_index(drop=True)
    if len(x) < 300:
        raise RuntimeError(f"insufficient bars: {exchange.id} {symbol}: {len(x)}")
    return x


def _realize_trades(scored: pd.DataFrame, venue: str, symbol: str, cfg: CanonicalConfig) -> pd.DataFrame:
    rows: list[dict] = []
    blocked_through = -1
    n = len(scored)
    for t in range(n - 1):
        if t <= blocked_through:
            continue
        d = int(scored.at[t, "canonical_direction"])
        if d == 0:
            continue
        entry_i = t + 1
        entry = float(scored.at[entry_i, "open"])
        atr = float(scored.at[t, "atr"])
        if not (np.isfinite(entry) and np.isfinite(atr) and entry > 0 and atr > 0):
            continue
        stop_dist = float(cfg.stop_atr * atr)
        stop = entry - d * stop_dist
        target = entry + d * TARGET_R * stop_dist
        last_i = min(n - 1, entry_i + MAX_HOLD_BARS - 1)
        exit_i = last_i
        exit_price = float(scored.at[last_i, "close"])
        outcome = "TIME"
        gross_r = d * (exit_price - entry) / stop_dist

        for j in range(entry_i, last_i + 1):
            low = float(scored.at[j, "low"])
            high = float(scored.at[j, "high"])
            hit_stop = low <= stop if d > 0 else high >= stop
            hit_target = high >= target if d > 0 else low <= target
            if hit_stop:  # conservative same-bar ordering
                exit_i = j
                exit_price = stop
                gross_r = -1.0
                outcome = "STOP"
                break
            if hit_target:
                exit_i = j
                exit_price = target
                gross_r = TARGET_R
                outcome = "TARGET"
                break

        base_cost_r = (BASE_ROUNDTRIP_BPS / 10_000.0) * entry / stop_dist
        stress_cost_r = (STRESS_ROUNDTRIP_BPS / 10_000.0) * entry / stop_dist
        rows.append(
            {
                "venue": venue,
                "symbol": symbol,
                "signal_time": scored.at[t, "timestamp"],
                "entry_time": scored.at[entry_i, "timestamp"],
                "exit_time": scored.at[exit_i, "timestamp"],
                "direction": d,
                "entry": entry,
                "stop_distance": stop_dist,
                "outcome": outcome,
                "gross_r": float(gross_r),
                "base_cost_r": float(base_cost_r),
                "stress_cost_r": float(stress_cost_r),
                "net_r": float(gross_r - base_cost_r),
                "stress_net_r": float(gross_r - stress_cost_r),
                "long_score": int(scored.at[t, "canonical_long_score"]),
                "short_score": int(scored.at[t, "canonical_short_score"]),
            }
        )
        blocked_through = exit_i
    return pd.DataFrame(rows)


def _profit_factor(r: pd.Series) -> float | None:
    x = pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else None
    return gains / losses


def _moving_block_ci_low(values: pd.Series) -> float | None:
    x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(x)
    if n < 2:
        return None
    block = min(BOOTSTRAP_BLOCK, n)
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for b in range(BOOTSTRAP_SAMPLES):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            idx = (start + np.arange(block)) % n
            sampled.extend(x[idx].tolist())
        means[b] = float(np.mean(sampled[:n]))
    return float(np.quantile(means, 0.025))


def _metrics(trades: pd.DataFrame, required_symbols: tuple[str, ...] = SYMBOLS) -> dict:
    if trades.empty:
        return {
            "n": 0,
            "profit_factor": None,
            "expectancy_r": None,
            "positive_asset_fraction": 0.0,
            "block_ci_low": None,
            "positive_quarter_fraction": 0.0,
            "stress_profit_factor": None,
        }
    x = trades.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
    asset_means = x.groupby("symbol")["net_r"].mean().to_dict()
    positive_assets = sum(float(asset_means.get(s, -np.inf)) > 0.0 for s in required_symbols)
    quarters = pd.to_datetime(x["exit_time"], utc=True).dt.to_period("Q")
    qsum = x.groupby(quarters)["net_r"].sum()
    return {
        "n": int(len(x)),
        "profit_factor": _profit_factor(x["net_r"]),
        "expectancy_r": float(x["net_r"].mean()),
        "positive_asset_fraction": float(positive_assets / len(required_symbols)),
        "block_ci_low": _moving_block_ci_low(x["net_r"]),
        "positive_quarter_fraction": float((qsum > 0).mean()) if len(qsum) else 0.0,
        "stress_profit_factor": _profit_factor(x["stress_net_r"]),
        "quarters": int(len(qsum)),
    }


def _ge(value, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) >= threshold


def _gt(value, threshold: float) -> bool:
    return value is not None and np.isfinite(value) and float(value) > threshold


def _venue_gate(metrics: dict) -> dict:
    pf = metrics.get("profit_factor")
    stress_pf = metrics.get("stress_profit_factor")
    # Positive infinity is valid for PF when there are gains and no losses.
    pf_ok = pf is not None and (math.isinf(float(pf)) or float(pf) >= 1.05)
    stress_ok = stress_pf is not None and (math.isinf(float(stress_pf)) or float(stress_pf) >= 1.00)
    gates = {
        "n_ge_200": int(metrics.get("n", 0)) >= 200,
        "pf_ge_1_05": pf_ok,
        "expectancy_gt_0": _gt(metrics.get("expectancy_r"), 0.0),
        "breadth_ge_0_60": _ge(metrics.get("positive_asset_fraction"), 0.60),
        "block_ci_low_gt_0": _gt(metrics.get("block_ci_low"), 0.0),
        "positive_quarter_fraction_ge_0_60": _ge(metrics.get("positive_quarter_fraction"), 0.60),
        "stress_pf_ge_1_00": stress_ok,
    }
    return {**gates, "venue_pass": bool(all(gates.values()))}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return "Infinity" if math.isinf(obj) and obj > 0 else ("-Infinity" if math.isinf(obj) else None)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = CanonicalConfig()
    all_trades: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    venue_results: dict[str, dict] = {}
    data_manifest: list[dict] = []

    for venue in VENUES:
        ex = _exchange(venue)
        markets = ex.load_markets()
        missing = [s for s in SYMBOLS if s not in markets]
        if missing:
            raise RuntimeError(f"required symbols missing on {venue}: {missing}")
        venue_trades: list[pd.DataFrame] = []
        for symbol in SYMBOLS:
            frame = fetch_fixed_ohlcv(ex, symbol)
            scored = canonical_score(frame, cfg)
            trades = _realize_trades(scored, venue, symbol, cfg)
            if not trades.empty:
                venue_trades.append(trades)
                all_trades.append(trades)
            diagnostics.append(
                {
                    "venue": venue,
                    "symbol": symbol,
                    "bars": int(len(scored)),
                    "first_bar": scored["timestamp"].iloc[0],
                    "last_bar": scored["timestamp"].iloc[-1],
                    "cusum_events": int(scored["canonical_cusum_event"].sum()),
                    "long_signals": int((scored["canonical_direction"] == 1).sum()),
                    "short_signals": int((scored["canonical_direction"] == -1).sum()),
                    "max_long_score": int(scored["canonical_long_score"].max()),
                    "max_short_score": int(scored["canonical_short_score"].max()),
                    "realized_trades": int(len(trades)),
                }
            )
            data_manifest.append(
                {
                    "venue": venue,
                    "symbol": symbol,
                    "timeframe": TIMEFRAME,
                    "start": START,
                    "end": END,
                    "bars": int(len(frame)),
                }
            )
        vt = pd.concat(venue_trades, ignore_index=True) if venue_trades else pd.DataFrame()
        metrics = _metrics(vt)
        gates = _venue_gate(metrics)
        venue_results[venue] = {"metrics": metrics, "gates": gates}

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades_all.empty:
        trades_all.to_csv(outdir / "trades_v39r.csv.gz", index=False, compression="gzip")
    else:
        pd.DataFrame(columns=["venue", "symbol", "signal_time", "entry_time", "exit_time", "direction", "net_r"]).to_csv(
            outdir / "trades_v39r.csv.gz", index=False, compression="gzip"
        )
    pd.DataFrame(diagnostics).to_csv(outdir / "score_diagnostics_v39r.csv", index=False)
    pd.DataFrame(data_manifest).to_csv(outdir / "data_manifest_v39r.csv", index=False)

    overall_pass = all(bool(venue_results[v]["gates"]["venue_pass"]) for v in VENUES)
    decision = {
        "version": "v0.39R",
        "decision": "V39R_DEVELOPMENT_PASS" if overall_pass else "V39R_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "venues": list(VENUES),
        "symbols": list(SYMBOLS),
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "base_roundtrip_bps": BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": STRESS_ROUNDTRIP_BPS,
        "target_r": TARGET_R,
        "max_hold_bars": MAX_HOLD_BARS,
        "config": cfg.__dict__,
        "venue_results": venue_results,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "parameters_changed_after_results": False,
    }
    (outdir / "decision_v39r.json").write_text(json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
