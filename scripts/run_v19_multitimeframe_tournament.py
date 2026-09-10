from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.multitimeframe_strategies_v19 import (
    STRATEGY_REGISTRY,
    TournamentConfig,
    choose_provisional_winner,
    evaluate_candidate,
    generate_direction,
    registry_frame,
    simulate_bracket_trades,
)

PERIOD_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_BARS = {"1m": 30000, "5m": 24000, "15m": 16000, "1h": 12000, "4h": 8000, "1d": 3000}
QUICK_BARS = {"1m": 3500, "5m": 3000, "15m": 2600, "1h": 2200, "4h": 1800, "1d": 1200}
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
]


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= now].reset_index(drop=True)


def fetch_timeframe(symbols: list[str], timeframe: str, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    period = PERIOD_MAP[timeframe]
    end_ms = utc_now_ms()
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            df = fetch_coinex_klines(symbol=symbol, period=period, market_type="spot", end_ms=end_ms, bars=bars)
            df = _drop_incomplete(df, PERIOD_MS[period])
            if len(df) >= min(300, bars // 4):
                out[symbol] = df
            else:
                errors[symbol] = f"insufficient rows={len(df)}"
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return out, errors


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    for candidate, frame in frames.items():
        if candidate != symbol:
            return frame
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="v0.19 source-derived multi-timeframe strategy tournament")
    p.add_argument("--output-dir", default="artifacts/v19-multitimeframe")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--quick", action="store_true", help="smaller public-data smoke run; never eligible for promotion")
    p.add_argument("--min-pretest-trades", type=int, default=1000)
    p.add_argument("--min-test-trades", type=int, default=200)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bars_map = QUICK_BARS if args.quick else DEFAULT_BARS
    cfg = TournamentConfig(
        min_pretest_trades=(10**9 if args.quick else args.min_pretest_trades),
        min_test_trades=args.min_test_trades,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registry_frame().to_csv(output / "strategy_registry.csv", index=False)

    all_trades: list[pd.DataFrame] = []
    summaries: list[dict] = []
    provenance: dict[str, dict] = {}

    for timeframe in PERIOD_MAP:
        frames, errors = fetch_timeframe(symbols, timeframe, bars_map[timeframe])
        provenance[timeframe] = {
            "requested_bars_per_symbol": bars_map[timeframe],
            "successful_symbols": sorted(frames),
            "failures": errors,
            "rows": {s: int(len(f)) for s, f in frames.items()},
        }
        for spec in [s for s in STRATEGY_REGISTRY if s.timeframe == timeframe]:
            candidate_parts: list[pd.DataFrame] = []
            for symbol, frame in frames.items():
                peer = peer_for(symbol, frames) if spec.family == "correlation_divergence" else None
                direction, features = generate_direction(spec, frame, peer=peer)
                ledger = simulate_bracket_trades(spec, frame, direction, features, symbol, cfg)
                if not ledger.empty:
                    candidate_parts.append(ledger)
            trades = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
            if not trades.empty:
                trades = trades.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
                all_trades.append(trades)
            metrics = evaluate_candidate(trades, cfg)
            summaries.append({
                "strategy": spec.name,
                "timeframe": spec.timeframe,
                "family": spec.family,
                "source_basis": spec.source_basis,
                **metrics,
            })
            print(json.dumps({
                "strategy": spec.name,
                "timeframe": timeframe,
                "pretest_trades": metrics["pretest_trades"],
                "test_trades": metrics["test_trades"],
                "validation_pf": metrics["validation_profit_factor"],
                "test_pf": metrics["test_profit_factor"],
            }, default=str))

    summary = pd.DataFrame(summaries).sort_values(["pretest_eligible", "validation_score"], ascending=[False, False])
    decision = choose_provisional_winner(summary, cfg)
    if args.quick:
        decision = {
            "decision": "SMOKE_ONLY_NO_PROMOTION",
            "reason": "--quick intentionally disables the 1000+ pre-test trade promotion gate",
            "winner": None,
            "live_execution_authorized": False,
            "paper_replacement_authorized": False,
        }

    summary.to_csv(output / "strategy_summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(output / "trade_ledger.csv", index=False)

    payload = {
        "version": "v0.19",
        "research_status": "SOURCE_DERIVED_MULTITIMEFRAME_TOURNAMENT",
        "source": "CoinEx public spot OHLCV; uploaded books/notes define hypothesis families; code contains causal crypto-normalized proxies",
        "symbols_requested": symbols,
        "timeframes": list(PERIOD_MAP),
        "candidate_count": len(STRATEGY_REGISTRY),
        "candidates_per_timeframe": registry_frame().groupby("timeframe").size().to_dict(),
        "execution_contract": {
            "decision": "closed bar t",
            "entry": "open t+1",
            "intrabar_collision": "stop first",
            "fee_bps_one_way": cfg.fee_bps,
            "slippage_bps_one_way": cfg.slippage_bps,
            "risk_per_trade": cfg.risk_per_trade,
            "max_drawdown_gate": cfg.max_drawdown,
            "no_overlapping_trade_per_symbol_candidate": True,
        },
        "selection_contract": {
            "development_fraction": cfg.development_fraction,
            "validation_fraction": cfg.validation_fraction,
            "test_fraction": 1.0 - cfg.development_fraction - cfg.validation_fraction,
            "min_pretest_trades": cfg.min_pretest_trades,
            "min_test_trades": cfg.min_test_trades,
            "selection_uses_validation_only": True,
            "final_test_is_one_shot_gate": True,
            "bootstrap": f"moving block, {cfg.bootstrap_samples} samples, block={cfg.bootstrap_block}",
        },
        "decision": decision,
        "provenance": provenance,
        "limitations": [
            "Several source rules were written for FX/index futures; crypto adaptations are explicitly research proxies.",
            "The 00/20/50/80 concept is normalized to crypto price magnitude rather than copied as pip levels.",
            "Open-interest/COT concepts are not silently imputed into spot OHLCV; they require a separate derivatives-data experiment.",
            "Thirty candidates create multiple-testing risk; any apparent winner remains forward-paper only until search-aware inference and fresh replication pass.",
            "No result from this runner authorizes real-money execution.",
        ],
    }
    (output / "decision.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
