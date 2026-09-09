from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from research_bot.binance_spot_archive import load_monthly_spot_archives
from research_bot.strategy_lab import (
    StrategyLabConfig,
    build_strategy_features,
    cost_sensitivity,
    evaluate_irgc_s,
    evaluate_rule_strategies,
)


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen v0.17 4H Ichimoku Strategy Lab")
    parser.add_argument("--btc-csv", type=Path)
    parser.add_argument("--eth-csv", type=Path)
    parser.add_argument("--bars", type=int, default=9000)
    parser.add_argument("--archive-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v17-strategy-lab"))
    args = parser.parse_args()
    cfg = StrategyLabConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {}
    provenance = {}
    for symbol, supplied in (("BTC/USDT", args.btc_csv), ("ETH/USDT", args.eth_csv)):
        if supplied:
            inputs[symbol] = _load_csv(supplied)
            provenance[symbol] = {"source": str(supplied), "kind": "user_csv"}
        elif args.archive_cache:
            archive_symbol = symbol.replace("/", "")
            inputs[symbol] = load_monthly_spot_archives(
                args.archive_cache, archive_symbol, timeframe="4h",
            ).tail(args.bars).reset_index(drop=True)
            provenance[symbol] = {
                "source": str(args.archive_cache),
                "kind": "official_binance_vision_monthly_spot_archives",
            }
        else:
            from research_bot.data import fetch_with_fallback
            frame, exchange = fetch_with_fallback(("coinex", "binance"), symbol, "4h", args.bars)
            inputs[symbol] = frame
            provenance[symbol] = {"source": exchange, "kind": "public_exchange_api"}

    rule_tables = []
    period_tables = []
    feature_map = {}
    all_costs = []
    for symbol, frame in inputs.items():
        summary, positions, features = evaluate_rule_strategies(frame, cfg)
        summary.insert(0, "symbol", symbol)
        rule_tables.append(summary)
        features.to_csv(args.output_dir / f"{symbol.split('/')[0]}_features_returns.csv", index=False)
        positions.to_csv(args.output_dir / f"{symbol.split('/')[0]}_positions.csv", index=False)
        costs = cost_sensitivity(features, positions, cfg)
        costs.insert(0, "symbol", symbol)
        all_costs.append(costs)
        feature_map[symbol] = features
        for period, start, end in (
            ("development", "2020-01-01", "2023-12-31 23:59:59"),
            ("validation", "2024-01-01", "2024-12-31 23:59:59"),
            ("final_test", "2025-01-01", "2025-12-31 23:59:59"),
        ):
            period_summary, _, _ = evaluate_rule_strategies(
                frame, cfg, evaluation_start=start, evaluation_end=end,
            )
            period_summary.insert(0, "period", period)
            period_summary.insert(0, "symbol", symbol)
            period_tables.append(period_summary)

    rules = pd.concat(rule_tables, ignore_index=True)
    rules.to_csv(args.output_dir / "rule_strategy_summary.csv", index=False)
    periods = pd.concat(period_tables, ignore_index=True)
    periods.to_csv(args.output_dir / "rule_strategy_period_summary.csv", index=False)
    pd.concat(all_costs, ignore_index=True).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    events, predictions, irgc = evaluate_irgc_s(feature_map, cfg)
    events.to_csv(args.output_dir / "irgc_s_events.csv", index=False)
    predictions.to_csv(args.output_dir / "irgc_s_oos_predictions.csv", index=False)
    payload = {
        "version": "0.17",
        "status": "RESEARCH_ONLY",
        "timeframe": "4h",
        "provenance": provenance,
        "config": asdict(cfg),
        "best_rule_by_sharpe": rules.sort_values("sharpe", ascending=False).head(1).to_dict("records"),
        "final_test_leaders": periods[periods["period"].eq("final_test")]
        .sort_values(["symbol", "sharpe"], ascending=[True, False])
        .groupby("symbol", as_index=False).head(1).to_dict("records"),
        "irgc_s": irgc,
        "live_execution": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
