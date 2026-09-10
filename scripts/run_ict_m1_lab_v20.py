from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from research_bot.binance_spot_archive import load_monthly_spot_archives
from research_bot.ict_m1_v20 import (
    IctM1Config,
    ORIGIN_VARIANTS,
    evaluate_ict_m1_variants,
    summarize_ict_m1_trades,
)
from research_bot.reproducibility import dataframe_fingerprint


PERIODS = (
    ("development", "2020-01-01 00:00:00+00:00", "2023-12-31 23:59:59+00:00"),
    ("validation", "2024-01-01 00:00:00+00:00", "2024-12-31 23:59:59+00:00"),
    ("final_test", "2025-01-01 00:00:00+00:00", "2025-12-31 23:59:59+00:00"),
)


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the causal v0.20 ICT/M1 origin-definition lab")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path)
    source.add_argument("--archive-cache", type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--bars", type=int, default=20_000)
    parser.add_argument("--ichimoku-gate", choices=("off", "trend"), default="trend")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v20-ict-m1-lab"))
    args = parser.parse_args()

    if args.input_csv:
        frame = _load_csv(args.input_csv).tail(args.bars).reset_index(drop=True)
        provenance = {"kind": "user_csv", "path": str(args.input_csv)}
    else:
        frame = load_monthly_spot_archives(
            args.archive_cache,
            args.symbol,
            timeframe=args.timeframe,
        ).tail(args.bars).reset_index(drop=True)
        provenance = {
            "kind": "official_binance_vision_monthly_spot_archives",
            "path": str(args.archive_cache),
        }

    cfg = IctM1Config(timeframe=args.timeframe, ichimoku_gate=args.ichimoku_gate)
    summary, trades, setups = evaluate_ict_m1_variants(frame, cfg)
    period_rows: list[dict] = []
    for period, start, end in PERIODS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        period_setups = setups[
            setups["confirm_time"].between(start_ts, end_ts, inclusive="both")
        ] if not setups.empty else setups
        for variant in ORIGIN_VARIANTS:
            period_trades = trades[
                trades["variant"].eq(variant)
                & trades["fill_time"].between(start_ts, end_ts, inclusive="both")
            ] if not trades.empty else trades
            row = summarize_ict_m1_trades(variant, period_setups, period_trades)
            row["period"] = period
            period_rows.append(row)
    period_summary = pd.DataFrame(period_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    period_summary.to_csv(args.output_dir / "period_summary.csv", index=False)
    setups.to_csv(args.output_dir / "setups.csv", index=False)
    trades.to_csv(args.output_dir / "trades.csv", index=False)
    payload = {
        "version": "0.20",
        "research_status": "HYPOTHESIS_NOT_PROMOTED",
        "strategy": "ICT_M1_CAUSAL_ORIGIN_ABLATION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "rows": len(frame),
        "start": frame["timestamp"].min(),
        "end": frame["timestamp"].max(),
        "dataset_sha256": dataframe_fingerprint(frame),
        "provenance": provenance,
        "config": asdict(cfg),
        "summary": summary.to_dict("records"),
        "period_summary": period_summary.to_dict("records"),
        "promotion_decision": "RESEARCH_ONLY_REQUIRES_OOS_AND_FORWARD_EVIDENCE",
        "paper_only": True,
        "live_execution": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
