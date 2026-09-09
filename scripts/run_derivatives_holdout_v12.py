from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.binance_vision import fetch_um_daily_metrics
from research_bot.cross_sectional_v06 import build_symbol_panel
from research_bot.derivatives_ablation_v12 import V12Config, run_v12_ablation


DEFAULT_COHORT = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "AVAXUSDT",
]


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _merge_oi(frame: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    if oi.empty:
        x["binance_open_interest"] = np.nan
        x["binance_open_interest_value"] = np.nan
        return x
    o = oi.copy()
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    keep = [c for c in ["timestamp", "binance_open_interest", "binance_open_interest_value"] if c in o]
    return pd.merge_asof(
        x.sort_values("timestamp"),
        o[keep].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).reset_index(drop=True)


def _load_symbol(symbol: str, start_month: str, end_month: str | None, oi_days: int):
    try:
        frame, archive_meta = build_symbol_panel(symbol, start_month, end_month, True)
        if frame.empty:
            return symbol, pd.DataFrame(), archive_meta, {}, "EMPTY_ARCHIVE_PANEL"
        coverage_end = pd.to_datetime(frame["timestamp"], utc=True).max().normalize().date()
        coverage_start = max(
            pd.to_datetime(frame["timestamp"], utc=True).min().normalize().date(),
            (pd.Timestamp(coverage_end) - pd.Timedelta(days=max(60, oi_days - 1))).date(),
        )
        try:
            oi, oi_meta = fetch_um_daily_metrics(
                symbol=symbol,
                start_date=coverage_start,
                end_date=coverage_end,
                max_workers=10,
                verify_checksum=False,
            )
        except Exception as exc:
            oi = pd.DataFrame()
            oi_meta = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        frame = _merge_oi(frame, oi)
        return symbol, frame, archive_meta, oi_meta, None
    except Exception as exc:
        return symbol, pd.DataFrame(), {}, {}, f"{type(exc).__name__}: {str(exc)[:600]}"


def main() -> None:
    p = argparse.ArgumentParser(description="v0.12 derivatives/order-flow external holdout")
    p.add_argument("--symbols", default=",".join(DEFAULT_COHORT))
    p.add_argument("--start-month", default="2025-09")
    p.add_argument("--end-month", default=None)
    p.add_argument("--oi-days", type=int, default=300)
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--holdout-fraction", type=float, default=0.25)
    p.add_argument("--top-quantile", type=float, default=0.25)
    p.add_argument("--cost-bps", type=float, default=8.0)
    p.add_argument("--min-assets", type=int, default=6)
    p.add_argument("--seeds", default="11,42,101")
    p.add_argument("--bootstrap-samples", type=int, default=300)
    p.add_argument("--block-length", type=int, default=9)
    p.add_argument("--output", default="artifacts/v12/derivatives_external_holdout.json")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if len(symbols) < args.min_assets:
        raise ValueError("Requested cohort smaller than min-assets gate")
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    panels: list[pd.DataFrame] = []
    archive_meta: dict[str, dict] = {}
    oi_meta: dict[str, dict] = {}
    failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(symbols)))) as ex:
        futures = [ex.submit(_load_symbol, s, args.start_month, args.end_month, args.oi_days) for s in symbols]
        for fut in as_completed(futures):
            symbol, frame, ameta, ometa, error = fut.result()
            archive_meta[symbol] = ameta
            oi_meta[symbol] = ometa
            if error is not None:
                failures[symbol] = error
                continue
            if frame.empty:
                failures[symbol] = "EMPTY_PANEL_AFTER_OI_MERGE"
                continue
            panels.append(frame)

    if len(panels) < args.min_assets:
        raise RuntimeError(f"Only {len(panels)} usable symbols; need {args.min_assets}; failures={failures}")

    panel = pd.concat(panels, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    cfg = V12Config(
        timeframe="8h",
        holdout_fraction=args.holdout_fraction,
        development_folds=3,
        top_quantile=args.top_quantile,
        one_way_cost_bps=args.cost_bps,
        min_assets_per_timestamp=args.min_assets,
        min_development_timestamps=240,
        min_holdout_timestamps=90,
        seeds=seeds,
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
        alpha=0.05,
        fdr_alpha=0.10,
        max_allowed_drawdown=-0.35,
        min_regime_periods=25,
        random_state=12012,
    )
    result = run_v12_ablation(panel, cfg)

    report = {
        "research_status": "V12_EXTERNAL_DERIVATIVES_HOLDOUT_NOT_TRADING_SIGNAL",
        "protocol": "docs/V12_DERIVATIVES_EXTERNAL_HOLDOUT_PROTOCOL.md",
        "data_contract": {
            "provider": "Binance Vision public USD-M archive",
            "bar_clock": "completed 8h perpetual bars",
            "premium_interpretation": "basis/crowding proxy, not reconstructed spot-perpetual basis",
            "orderflow_interpretation": "kline taker-buy quote imbalance; coarse, not L2/L3",
            "open_interest_availability": "daily metrics shifted +1 UTC day before backward asof merge",
            "live_execution": False,
        },
        "requested_symbols": symbols,
        "usable_symbols": sorted(panel["symbol"].unique().tolist()),
        "provider_failures": failures,
        "archive_meta": archive_meta,
        "open_interest_meta": oi_meta,
        "panel_raw_rows": int(len(panel)),
        "config": asdict(cfg),
        "ablation": result,
        "decision_contract": (
            "No output in this artifact authorizes BUY/SELL, paper execution, testnet execution, "
            "or live execution. A provisional derivatives edge still requires forward paper replication."
        ),
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")

    # Console output is intentionally compact and evidence-oriented.
    print(json.dumps(_clean({
        "requested": len(symbols),
        "usable": len(report["usable_symbols"]),
        "failures": failures,
        "coverage": [result["coverage_start"], result["coverage_end"]],
        "development": [result["development_start"], result["development_end"]],
        "holdout": [result["final_holdout_start"], result["final_holdout_end"]],
        "promotion": result["promotion_status"],
        "best_full_model": result["provisional_best_full_model"],
        "reasons": result["promotion_reasons"],
        "holdout_summary": result["holdout_summary"],
        "multiple_testing": result["multiple_testing"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
