from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.microstructure_v13 import (
    MicrostructureForwardConfig,
    build_forward_microstructure_panel,
    diagnostic_correlations,
)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--start-date", default="2026-09-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--interval", default="1h")
    p.add_argument("--output-dir", default="artifacts/v13")
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = MicrostructureForwardConfig(
        interval=a.interval,
        start_date=a.start_date,
        end_date=a.end_date,
    )

    panels = []
    symbol_reports = {}
    for symbol in [s.strip().upper() for s in a.symbols.split(",") if s.strip()]:
        frame, meta = build_forward_microstructure_panel(symbol, cfg)
        symbol_reports[symbol] = {
            "meta": meta,
            "diagnostic_spearman": diagnostic_correlations(frame),
        }
        if not frame.empty:
            frame.to_csv(out / f"{symbol.lower()}_forward_microstructure.csv", index=False)
            panels.append(frame)

    combined = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(out / "forward_microstructure_panel.csv", index=False)

    statuses = [x["meta"].get("status") for x in symbol_reports.values()]
    report = {
        "research_status": "V13_FORWARD_MICROSTRUCTURE_COLLECTION_NOT_TRADING_SIGNAL",
        "protocol_frozen_at": "2026-09-09",
        "symbols": list(symbol_reports),
        "rows": int(len(combined)),
        "symbol_reports": symbol_reports,
        "liquidation_history": "DATA_UNAVAILABLE",
        "promotion_status": "NO_MODEL_PROMOTION_ATTEMPTED",
        "window_status": (
            "FORWARD_WINDOW_READY"
            if statuses and all(s == "FORWARD_WINDOW_READY" for s in statuses)
            else "FORWARD_WINDOW_ACCUMULATING"
        ),
        "method": {
            "true_basis": "synchronized completed spot and USD-M perpetual 1h closes",
            "order_flow": "separate spot/futures 1h taker-buy quote imbalance proxy",
            "open_interest": "Binance Vision daily metrics with conservative +1 UTC day availability shift",
            "diagnostics": "Spearman only; no test-set tuning or profitability claim",
        },
        "warnings": [
            "The post-v0.12 forward window begins 2026-09-01 and is intentionally not backfilled with pre-window outcomes.",
            "Kline taker flow is finer than the v0.12 8h proxy but is still not L2/L3 microstructure.",
            "Liquidation history is recorded as DATA_UNAVAILABLE rather than fabricated.",
            "This artifact authorizes no paper, testnet or live trade.",
        ],
    }
    (out / "v13_forward_microstructure_report.json").write_text(
        json.dumps(clean(report), indent=2), encoding="utf-8"
    )
    print("===V13_FORWARD_MICROSTRUCTURE_REPORT===")
    print(json.dumps(clean(report), indent=2))
    print("===END_V13_FORWARD_MICROSTRUCTURE_REPORT===")


if __name__ == "__main__":
    main()
