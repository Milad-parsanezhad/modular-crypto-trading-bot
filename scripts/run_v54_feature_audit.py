from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.candle_time_v53 import CandleTimeContractV53, closed_bar_snapshot_v53
from research_bot.coinex_public import fetch_coinex_klines
from research_bot.feature_audit_v54 import V54AuditConfig, audit_feature_families_v54
from research_bot.multitimeframe_v53 import build_multitimeframe_feature_frame_v53


DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return pd.Timestamp(obj).isoformat()
    return obj


def build_symbol_frame(symbol: str, *, now: datetime, one_hour_bars: int, four_hour_bars: int) -> pd.DataFrame:
    one = fetch_coinex_klines(symbol, period="1hour", market_type="spot", bars=one_hour_bars)
    four = fetch_coinex_klines(symbol, period="4hour", market_type="spot", bars=four_hour_bars)
    one = closed_bar_snapshot_v53(one, decision_time=pd.Timestamp(now), contract=CandleTimeContractV53("1h"))
    four = closed_bar_snapshot_v53(four, decision_time=pd.Timestamp(now), contract=CandleTimeContractV53("4h"))
    if len(one) < 900 or len(four) < 150:
        raise ValueError(f"insufficient closed history for {symbol}: 1h={len(one)}, 4h={len(four)}")
    one["asset"] = symbol
    four["asset"] = symbol
    return build_multitimeframe_feature_frame_v53({"1h": one, "4h": four}, decision_timeframe="1h")


def run_universe(symbols: tuple[str, ...], *, now: datetime, config: V54AuditConfig, one_hour_bars: int, four_hour_bars: int) -> dict:
    per_symbol: dict[str, dict] = {}
    blocked: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = build_symbol_frame(symbol, now=now, one_hour_bars=one_hour_bars, four_hour_bars=four_hour_bars)
            per_symbol[symbol] = audit_feature_families_v54(frame, config)
        except Exception as exc:
            blocked[symbol] = f"{type(exc).__name__}: {exc}"

    family_rows: dict[str, list[dict]] = {}
    for symbol, report in per_symbol.items():
        for family, values in report.get("family_value", {}).items():
            family_rows.setdefault(family, []).append({"symbol": symbol, **values})

    family_summary: dict[str, dict] = {}
    for family, rows in sorted(family_rows.items()):
        increments = np.asarray([float(r["all_minus_drop_sharpe"]) for r in rows], dtype=float)
        only = np.asarray([float(r["only_family_sharpe"]) for r in rows], dtype=float)
        family_summary[family] = {
            "symbols": int(len(rows)),
            "positive_increment_symbols": int(sum(bool(r["positive_increment"]) for r in rows)),
            "median_all_minus_drop_sharpe": float(np.median(increments)),
            "median_only_family_sharpe": float(np.median(only)),
            "promotion_candidate": bool(len(rows) >= 3 and np.median(increments) > 0 and sum(bool(r["positive_increment"]) for r in rows) >= 3),
        }

    return {
        "experiment": "V54_REAL_COINEX_FEATURE_AUDIT",
        "generated_at": pd.Timestamp(now).isoformat(),
        "symbols_requested": list(symbols),
        "symbols_completed": sorted(per_symbol),
        "blocked": blocked,
        "per_symbol": per_symbol,
        "family_summary": family_summary,
        "promotion_rule": "candidate only if >=3 symbols, median ALL-minus-DROP Sharpe > 0, and positive increment on >=3 symbols; still no execution authorization",
        "paper_execution": False,
        "live_execution": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--one-hour-bars", type=int, default=3000)
    p.add_argument("--four-hour-bars", type=int, default=1200)
    p.add_argument("--min-train", type=int, default=1200)
    p.add_argument("--test-rows", type=int, default=240)
    p.add_argument("--step-rows", type=int, default=240)
    p.add_argument("--output", type=Path, default=Path("artifacts/v54/feature_audit.json"))
    args = p.parse_args()

    cfg = V54AuditConfig(
        min_train_rows=args.min_train,
        test_rows=args.test_rows,
        step_rows=args.step_rows,
        purge_bars=1,
        horizon_bars=1,
    )
    report = run_universe(
        tuple(args.symbols),
        now=datetime.now(timezone.utc),
        config=cfg,
        one_hour_bars=args.one_hour_bars,
        four_hour_bars=args.four_hour_bars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
