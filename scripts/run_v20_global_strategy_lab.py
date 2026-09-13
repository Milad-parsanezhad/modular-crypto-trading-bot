from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import ccxt
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    V19_NAMES,
    RiskPsychologyPolicy,
    V20ValidationConfig,
    apply_candidate_level_policy,
    evaluate_external_replication,
    evaluate_v20_candidate,
    generate_direction_v20,
    registry_frame_v20,
    simulate_v20_trades,
)

COINEX_PERIOD = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
CCXT_PERIOD = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
DEFAULT_BARS = {"1m": 30000, "5m": 24000, "15m": 16000, "1h": 12000, "4h": 8000, "1d": 3000}
EXTERNAL_BARS = {"1m": 8000, "5m": 7000, "15m": 6000, "1h": 5000, "4h": 3000, "1d": 1800}
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)


def fetch_coinex_timeframe(symbols: list[str], timeframe: str, bars: int):
    period, end_ms = COINEX_PERIOD[timeframe], utc_now_ms()
    frames, errors = {}, {}
    for symbol in symbols:
        try:
            df = fetch_coinex_klines(symbol=symbol, period=period, market_type="spot", end_ms=end_ms, bars=bars)
            df = _drop_incomplete(df, PERIOD_MS[period])
            if len(df) >= min(300, bars // 4):
                frames[symbol] = df
            else:
                errors[symbol] = f"insufficient rows={len(df)}"
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return frames, errors


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]):
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((f for s, f in frames.items() if s != symbol), None)


def fetch_ccxt_timeframe(exchange_id: str, symbols: list[str], timeframe: str, bars: int):
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    tf = CCXT_PERIOD[timeframe]
    step_ms = int(exchange.parse_timeframe(tf) * 1000)
    now_ms = exchange.milliseconds()
    frames, errors = {}, {}
    try:
        for symbol in symbols:
            if symbol not in exchange.markets:
                errors[symbol] = "market unavailable"
                continue
            try:
                cursor = now_ms - int((bars + 50) * step_ms)
                rows, loops = [], 0
                while cursor < now_ms and len(rows) < bars + 50 and loops < 80:
                    loops += 1
                    batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=min(300, bars + 50 - len(rows)))
                    if not batch:
                        break
                    rows.extend(batch)
                    nxt = int(batch[-1][0]) + step_ms
                    if nxt <= cursor:
                        break
                    cursor = nxt
                    time.sleep(exchange.rateLimit / 1000.0 if exchange.rateLimit else 0.05)
                if not rows:
                    errors[symbol] = "no rows"
                    continue
                df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                df = _drop_incomplete(df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").sort_values("timestamp"), step_ms).tail(bars).reset_index(drop=True)
                if len(df) >= min(300, bars // 4):
                    frames[symbol] = df
                else:
                    errors[symbol] = f"insufficient rows={len(df)}"
            except Exception as exc:
                errors[symbol] = f"{type(exc).__name__}: {exc}"
    finally:
        exchange.close()
    return frames, errors


def run_candidate(spec, frames, tournament, policy):
    parts = []
    for symbol, frame in frames.items():
        peer = peer_for(symbol, frames) if ("CORRELATION" in spec.name or spec.family == "correlation_divergence") else None
        direction, features = generate_direction_v20(spec, frame, peer=peer)
        ledger = simulate_v20_trades(spec, frame, direction, features, symbol, tournament=tournament, policy=policy)
        if not ledger.empty:
            parts.append(ledger)
    attempts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return apply_candidate_level_policy(attempts, spec, policy=policy) if not attempts.empty else attempts


def policy_dict(p: RiskPsychologyPolicy) -> dict:
    return {k: getattr(p, k) for k in p.__dataclass_fields__}


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.20 global-source + books + MTF + risk/psychology strategy lab")
    ap.add_argument("--output-dir", default="artifacts/v20-global-strategy-lab")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--external-exchange", default="okx")
    ap.add_argument("--min-pretest-trades", type=int, default=1000)
    ap.add_argument("--min-validation-trades", type=int, default=200)
    ap.add_argument("--min-external-trades", type=int, default=200)
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    policy = RiskPsychologyPolicy()
    tournament = TournamentConfig(risk_per_trade=policy.base_risk_per_trade, min_pretest_trades=args.min_pretest_trades, min_test_trades=args.min_validation_trades)
    validation = V20ValidationConfig(min_pretest_trades=args.min_pretest_trades, min_validation_trades=args.min_validation_trades, min_external_trades=args.min_external_trades)
    registry = registry_frame_v20()
    registry.to_csv(output / "strategy_registry.csv", index=False)
    total_trials = len(STRATEGY_REGISTRY_V20)
    summaries, all_attempts, provenance = [], [], {}

    for timeframe in COINEX_PERIOD:
        frames, errors = fetch_coinex_timeframe(symbols, timeframe, DEFAULT_BARS[timeframe])
        provenance[timeframe] = {"source": "CoinEx public spot OHLCV", "requested_bars_per_symbol": DEFAULT_BARS[timeframe], "successful_symbols": sorted(frames), "failures": errors, "rows": {s: int(len(f)) for s, f in frames.items()}}
        for spec in [s for s in STRATEGY_REGISTRY_V20 if s.timeframe == timeframe]:
            attempts = run_candidate(spec, frames, tournament, policy)
            if not attempts.empty:
                all_attempts.append(attempts)
            m = evaluate_v20_candidate(attempts, total_trials=total_trials, validation=validation)
            summaries.append({"strategy": spec.name, "timeframe": spec.timeframe, "family": spec.family, "source_basis": spec.source_basis, "generation": "v0.19_baseline" if spec.name in V19_NAMES else "v0.20_new", **m})
            print(json.dumps({"strategy": spec.name, "timeframe": timeframe, "pretest": m["pretest_trades"], "validation": m["validation_trades"], "pf": m["validation_profit_factor"], "exp_r": m["validation_expectancy_r"], "mdd": m["validation_max_drawdown"], "p_adj": m["validation_multiplicity_adjusted_p"], "eligible": m["internal_eligible"]}, default=str))

    summary = pd.DataFrame(summaries).sort_values(["internal_eligible", "validation_score_v20"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    if all_attempts:
        pd.concat(all_attempts, ignore_index=True).to_csv(output / "trade_attempt_ledger.csv", index=False)
    leaders = []
    for tf, group in summary.groupby("timeframe", sort=False):
        leaders.append(group.sort_values(["internal_eligible", "validation_score_v20"], ascending=[False, False]).iloc[0].to_dict())
    pd.DataFrame(leaders).to_csv(output / "timeframe_leaders.csv", index=False)

    eligible = summary[summary["internal_eligible"] == True].copy()  # noqa: E712
    selected = eligible.sort_values("validation_score_v20", ascending=False).iloc[0] if not eligible.empty else None
    external_result, external_provenance = None, None
    if selected is not None:
        spec = next(s for s in STRATEGY_REGISTRY_V20 if s.name == str(selected["strategy"]))
        ext_frames, ext_errors = fetch_ccxt_timeframe(args.external_exchange, symbols, spec.timeframe, EXTERNAL_BARS[spec.timeframe])
        exchange_used = args.external_exchange
        if len(ext_frames) < 5:
            exchange_used = "kucoin" if args.external_exchange != "kucoin" else "okx"
            ext_frames, ext_errors = fetch_ccxt_timeframe(exchange_used, symbols, spec.timeframe, EXTERNAL_BARS[spec.timeframe])
        external_provenance = {"exchange": exchange_used, "timeframe": spec.timeframe, "requested_bars_per_symbol": EXTERNAL_BARS[spec.timeframe], "successful_symbols": sorted(ext_frames), "failures": ext_errors, "rows": {s: int(len(f)) for s, f in ext_frames.items()}}
        ext_attempts = run_candidate(spec, ext_frames, tournament, policy)
        if not ext_attempts.empty:
            ext_attempts.to_csv(output / "external_replication_ledger.csv", index=False)
        external_result = evaluate_external_replication(ext_attempts, validation)
        (output / "external_replication.json").write_text(json.dumps(external_result, indent=2, default=str), encoding="utf-8")

    if selected is None:
        decision = {"decision": "NO_STRATEGY_PROMOTED", "reason": "No candidate cleared the frozen internal 1000+ signal-trade, cost, risk, breadth, block-CI and multiplicity gates.", "winner": None, "forward_paper_candidate_authorized": False, "paper_replacement_authorized": False, "live_execution_authorized": False}
    elif external_result and external_result.get("external_pass"):
        decision = {"decision": "FORWARD_PAPER_CANDIDATE", "reason": "Internal winner also passed external-venue replication; fresh forward PAPER evidence remains mandatory.", "winner": str(selected["strategy"]), "timeframe": str(selected["timeframe"]), "forward_paper_candidate_authorized": True, "paper_replacement_authorized": False, "live_execution_authorized": False}
    else:
        decision = {"decision": "NO_STRATEGY_PROMOTED", "reason": "The internal winner failed or lacked sufficient external-venue replication evidence.", "winner": str(selected["strategy"]), "timeframe": str(selected["timeframe"]), "forward_paper_candidate_authorized": False, "paper_replacement_authorized": False, "live_execution_authorized": False}

    payload = {
        "version": "v0.20", "research_status": "GLOBAL_SOURCE_BOOK_MTF_RISK_PSYCHOLOGY_LAB",
        "candidate_count": total_trials, "candidates_per_timeframe": registry.groupby("timeframe").size().to_dict(),
        "symbols_requested": symbols, "timeframes": list(COINEX_PERIOD),
        "scientific_contract": {"causality": "closed bar signal; next-open fill; completed HTF only; causal Ichimoku", "cost": "10 bps fee + 2 bps slippage each way", "risk": policy_dict(policy), "selection": ">=1000 signal trades; validation only; block CI; multiplicity screen; then external venue", "fresh_forward_required": True},
        "provenance": provenance, "external_provenance": external_provenance, "decision": decision,
        "limitations": [
            "No finite study can literally crawl every educational website; source selection prioritizes official ICT, professional curricula, regulators and academic evidence.",
            "ICT concepts are research hypotheses, not proof of alpha.",
            "CoinEx and a second crypto venue are economically correlated, so venue replication is not a fully independent asset-class holdout.",
            "v0.19 history has already been inspected; v0.20 history is discovery/replication evidence, not a virgin final test.",
            "The multiplicity screen is conservative normal/Bonferroni plus moving-block CI, not a full White Reality Check/SPA/DSR/PBO implementation.",
            "OI/funding/basis/CVD/liquidation/order-flow signals are not fabricated from OHLCV and require a separate point-in-time data track.",
            "No result authorizes live capital; a pass only starts a fresh forward PAPER phase.",
        ],
    }
    (output / "decision.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
