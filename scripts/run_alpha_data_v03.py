from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.coinex_public import (
    audit_kline_gaps,
    fetch_coinex_basis_history,
    fetch_coinex_funding_history,
    fetch_coinex_klines,
    fetch_coinex_market_deals,
    fetch_coinex_open_interest_snapshot,
    utc_now_ms,
)
from research_bot.orderflow import aggregate_order_flow, order_flow_summary
from research_bot.bybit_public import fetch_bybit_open_interest_history


def _clean(obj):
    if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    return obj


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _series_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    return {
        "n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
        "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
        "nonzero_share": float((x.abs() > 1e-12).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--bars", type=int, default=15000)
    p.add_argument("--period", default="4hour")
    p.add_argument("--funding-days", type=int, default=2500)
    p.add_argument("--basis-days", type=int, default=365)
    p.add_argument("--trade-pages", type=int, default=5)
    p.add_argument("--output-dir", default="artifacts/v03")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    end_ms = utc_now_ms()
    step_ms = {"1hour":3_600_000,"2hour":7_200_000,"4hour":14_400_000,"6hour":21_600_000,"12hour":43_200_000,"1day":86_400_000}.get(args.period)
    if step_ms is None: raise ValueError("v0.3 supports 1h/2h/4h/6h/12h/1d")
    start_ms = end_ms - int(args.bars * step_ms * 1.02)

    spot = fetch_coinex_klines(args.symbol,args.period,"spot",start_ms,end_ms,args.bars)
    futures = fetch_coinex_klines(args.symbol,args.period,"futures",start_ms,end_ms,args.bars)
    funding = fetch_coinex_funding_history(args.symbol,end_ms-args.funding_days*86_400_000,end_ms)
    basis = fetch_coinex_basis_history(args.symbol,end_ms-args.basis_days*86_400_000,end_ms)
    oi_snapshot = fetch_coinex_open_interest_snapshot(args.symbol)

    spot_trades = fetch_coinex_market_deals(args.symbol,"spot",pages=args.trade_pages)
    futures_trades = fetch_coinex_market_deals(args.symbol,"futures",pages=args.trade_pages)
    spot_flow = aggregate_order_flow(spot_trades,"4h")
    futures_flow = aggregate_order_flow(futures_trades,"4h")

    aligned = spot[["timestamp","close"]].rename(columns={"close":"spot_close"}).merge(
        futures[["timestamp","close"]].rename(columns={"close":"futures_close"}),on="timestamp",how="inner")
    if not aligned.empty:
        aligned["realized_basis_rate"]=(aligned["futures_close"]-aligned["spot_close"])/aligned["spot_close"]

    funding_4h=pd.DataFrame()
    if not funding.empty:
        funding_4h=funding.set_index("timestamp")[["actual_funding_rate","theoretical_funding_rate"]].resample("4h").sum(min_count=1).reset_index()
    basis_4h=pd.DataFrame()
    if not basis.empty:
        basis_4h=basis.set_index("timestamp")[["basis_rate"]].resample("4h").mean().reset_index()

    alpha_frame=aligned.copy()
    parts=[funding_4h,basis_4h,
        spot_flow.rename(columns={c:f"spot_of_{c}" for c in spot_flow.columns if c!="timestamp"}),
        futures_flow.rename(columns={c:f"fut_of_{c}" for c in futures_flow.columns if c!="timestamp"})]
    for part in parts:
        if not part.empty: alpha_frame=alpha_frame.merge(part,on="timestamp",how="left")

    bybit_oi=pd.DataFrame(); bybit_oi_error=None
    try:
        bybit_oi=fetch_bybit_open_interest_history("BTCUSDT","4h",start_ms=start_ms,end_ms=end_ms,limit=200,max_pages=150)
        if not bybit_oi.empty and not alpha_frame.empty:
            alpha_frame=alpha_frame.merge(bybit_oi[["timestamp","open_interest"]].rename(columns={"open_interest":"bybit_oi"}),on="timestamp",how="left")
    except Exception as exc:
        bybit_oi_error=f"{type(exc).__name__}: {exc}"

    _write_csv(spot,out/"coinex_spot_klines.csv"); _write_csv(futures,out/"coinex_futures_klines.csv")
    _write_csv(funding,out/"coinex_funding_history.csv"); _write_csv(basis,out/"coinex_basis_history.csv")
    _write_csv(spot_trades,out/"coinex_spot_trades_recent.csv"); _write_csv(futures_trades,out/"coinex_futures_trades_recent.csv")
    _write_csv(spot_flow,out/"coinex_spot_orderflow_4h.csv"); _write_csv(futures_flow,out/"coinex_futures_orderflow_4h.csv")
    _write_csv(alpha_frame,out/"alpha_frame_v03.csv")
    if not bybit_oi.empty: _write_csv(bybit_oi,out/"bybit_open_interest_4h.csv")

    report={
        "research_status":"v0.3_data_alpha_lab_not_production","symbol":args.symbol,"period":args.period,"requested_bars":args.bars,
        "spot_gap_audit":audit_kline_gaps(spot,args.period),"futures_gap_audit":audit_kline_gaps(futures,args.period),
        "aligned_spot_futures_rows":int(len(aligned)),
        "realized_basis_proxy":_series_stats(aligned["realized_basis_rate"] if "realized_basis_rate" in aligned else pd.Series(dtype=float)),
        "funding":{"rows":int(len(funding)),"coverage_start":None if funding.empty else funding["timestamp"].min().isoformat(),"coverage_end":None if funding.empty else funding["timestamp"].max().isoformat(),"actual_rate_stats":_series_stats(funding["actual_funding_rate"] if not funding.empty else pd.Series(dtype=float))},
        "coinex_basis_api":{"rows":int(len(basis)),"coverage_start":None if basis.empty else basis["timestamp"].min().isoformat(),"coverage_end":None if basis.empty else basis["timestamp"].max().isoformat(),"basis_rate_stats":_series_stats(basis["basis_rate"] if not basis.empty else pd.Series(dtype=float)),"interpretation":"Treat as uninformative if nonzero_share is zero; use independently computed spot-perpetual basis proxy instead."},
        "coinex_open_interest_snapshot":oi_snapshot,
        "bybit_cross_venue_open_interest":{"rows":int(len(bybit_oi)),"coverage_start":None if bybit_oi.empty else bybit_oi["timestamp"].min().isoformat(),"coverage_end":None if bybit_oi.empty else bybit_oi["timestamp"].max().isoformat(),"error":bybit_oi_error,"scope_note":"cross-venue public historical OI; never relabeled as CoinEx OI"},
        "spot_order_flow":order_flow_summary(spot_trades),"futures_order_flow":order_flow_summary(futures_trades),
        "alpha_frame_rows":int(len(alpha_frame)),
        "warnings":[
            "v0.3 is data-integrity/alpha-source ingestion, not a profitability claim.",
            "CoinEx OI is a current point-in-time snapshot; historical CoinEx OI will accumulate prospectively unless an archive is sourced.",
            "CoinEx REST recent deals cover a short interval in liquid BTC; full historical order-flow requires CoinEx Historical Market Data transaction archives.",
            "Bybit OI is explicitly cross-venue and must not be confused with CoinEx-specific OI.",
            "No live orders or private API keys are used."
        ]}
    (out/"v03_data_alpha_report.json").write_text(json.dumps(_clean(report),indent=2),encoding="utf-8")
    print("===V03_DATA_ALPHA_REPORT==="); print(json.dumps(_clean(report),indent=2)); print("===END_V03_DATA_ALPHA_REPORT===")

if __name__=="__main__": main()
