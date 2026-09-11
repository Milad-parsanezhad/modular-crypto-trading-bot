from __future__ import annotations

"""v0.26 locked-winner external replication + strictly fresh temporal OOS monitor.

The v0.25 winner is immutable in this stage.  No parameter search, threshold
relaxation, strategy replacement or live execution is permitted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import ccxt
import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
    V20ValidationConfig,
    generate_direction_v20,
    simulate_v20_trades,
)
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25, apply_portfolio_allocator_v25
from research_bot.replication_monitor_v26 import (
    LOCKED_TIMEFRAME_V26,
    LOCKED_WINNER_V26,
    V25_ARTIFACT_ID,
    V25_SELECTION_LOCK_UTC,
    V25_WORKFLOW_RUN_ID,
    V26QualificationPolicy,
    compact_replication_metrics,
    filter_fresh_temporal_attempts,
    qualification_decision,
)


LOCKED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]
EXTERNAL_EXCHANGE_ORDER = ("okx", "kucoin")
EXTERNAL_BARS = 3000
TEMPORAL_WARMUP_BARS = 3200
MIN_VENUE_SYMBOLS = 15
ROUND_TRIP_COST_FRACTION = 0.0024


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _drop_incomplete(frame: pd.DataFrame, step_ms: int, *, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    cutoff = now if now is not None else pd.Timestamp.now(tz="UTC")
    return x.loc[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= cutoff].reset_index(drop=True)


def _frame_digest(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    x = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).astype(str)
    payload = x.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot(frames: dict[str, pd.DataFrame], path: Path) -> None:
    parts = []
    for symbol, frame in sorted(frames.items()):
        x = frame.copy()
        x.insert(0, "symbol", symbol)
        parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(path, index=False, compression="gzip")


def _provenance(frames: dict[str, pd.DataFrame], errors: dict[str, str], *, source: str, requested_bars: int) -> dict[str, Any]:
    rows = {}
    ranges = {}
    digests = {}
    for symbol, frame in sorted(frames.items()):
        rows[symbol] = int(len(frame))
        if len(frame):
            ts = pd.to_datetime(frame["timestamp"], utc=True)
            ranges[symbol] = {"first": ts.min().isoformat(), "last": ts.max().isoformat()}
        digests[symbol] = _frame_digest(frame)
    return {
        "source": source,
        "requested_bars_per_symbol": requested_bars,
        "successful_symbols": sorted(frames),
        "symbol_count": len(frames),
        "rows": rows,
        "ranges": ranges,
        "sha256_by_symbol": digests,
        "failures": errors,
    }


def fetch_ccxt_frames(exchange_id: str, symbols: list[str], *, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    tf = LOCKED_TIMEFRAME_V26
    step_ms = int(exchange.parse_timeframe(tf) * 1000)
    now_ms = exchange.milliseconds()
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    try:
        for symbol in symbols:
            if symbol not in exchange.markets:
                errors[symbol] = "market unavailable"
                continue
            try:
                cursor = now_ms - int((bars + 80) * step_ms)
                rows: list[list[float]] = []
                loops = 0
                while cursor < now_ms and len(rows) < bars + 80 and loops < 100:
                    loops += 1
                    limit = min(300, bars + 80 - len(rows))
                    batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=limit)
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
                frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
                frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
                frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
                frame = _drop_incomplete(frame, step_ms).tail(bars).reset_index(drop=True)
                if len(frame) >= min(1000, bars // 2):
                    frames[symbol] = frame
                else:
                    errors[symbol] = f"insufficient rows={len(frame)}"
            except Exception as exc:  # pragma: no cover - network/runtime path
                errors[symbol] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    return frames, errors


def fetch_coinex_frames(symbols: list[str], *, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    end_ms = utc_now_ms()
    for symbol in symbols:
        try:
            frame = fetch_coinex_klines(symbol=symbol, period="4hour", market_type="spot", end_ms=end_ms, bars=bars)
            frame = _drop_incomplete(frame, PERIOD_MS["4hour"]).tail(bars).reset_index(drop=True)
            if len(frame) >= min(1000, bars // 2):
                frames[symbol] = frame
            else:
                errors[symbol] = f"insufficient rows={len(frame)}"
        except Exception as exc:  # pragma: no cover - network/runtime path
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return frames, errors


def _raw_attempts(spec: Any, frames: dict[str, pd.DataFrame], tournament: TournamentConfig, policy: RiskPsychologyPolicy) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        direction, features = generate_direction_v20(spec, frame, peer=None)
        ledger = simulate_v20_trades(
            spec,
            frame,
            direction,
            features,
            symbol,
            tournament=tournament,
            policy=policy,
        )
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def _empty_external_eval() -> dict[str, Any]:
    return {
        "external_trades": 0,
        "external_total_return": np.nan,
        "external_profit_factor": np.nan,
        "external_win_rate": np.nan,
        "external_expectancy_r": np.nan,
        "external_median_r": np.nan,
        "external_max_drawdown": np.nan,
        "external_mean_account_return": np.nan,
        "external_settlement_batches": 0,
        "external_positive_asset_fraction": 0.0,
        "external_block_ci_low": np.nan,
        "external_block_ci_high": np.nan,
        "external_pass_v25": False,
    }


def _evaluate(allocated: pd.DataFrame, validation: V20ValidationConfig) -> dict[str, Any]:
    if allocated.empty:
        return _empty_external_eval()
    return evaluate_external_replication_v25(allocated, validation=validation)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.26 locked-winner fresh replication monitor")
    ap.add_argument("--output-dir", default="artifacts/v26-fresh-replication-monitor")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = next(s for s in STRATEGY_REGISTRY_V20 if s.name == LOCKED_WINNER_V26)
    if spec.timeframe != LOCKED_TIMEFRAME_V26:
        raise RuntimeError("locked winner timeframe drift")

    policy = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    qualification = V26QualificationPolicy()
    tournament = TournamentConfig(
        risk_per_trade=policy.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=qualification.min_temporal_trades,
    )
    validation = V20ValidationConfig(
        min_external_trades=qualification.min_external_trades,
        min_validation_trades=qualification.min_temporal_trades,
    )

    external_frames: dict[str, pd.DataFrame] = {}
    external_errors: dict[str, str] = {}
    external_exchange = None
    external_attempts = pd.DataFrame()
    external_allocated = pd.DataFrame()
    external_eval = _empty_external_eval()

    venue_attempts: list[dict[str, Any]] = []
    for exchange_id in EXTERNAL_EXCHANGE_ORDER:
        frames, errors = fetch_ccxt_frames(exchange_id, LOCKED_SYMBOLS, bars=EXTERNAL_BARS)
        venue_attempts.append({"exchange": exchange_id, "symbol_count": len(frames), "failures": errors})
        if len(frames) >= MIN_VENUE_SYMBOLS:
            external_exchange = exchange_id
            external_frames, external_errors = frames, errors
            break
    if external_exchange is not None:
        external_attempts = _raw_attempts(spec, external_frames, tournament, policy)
        if not external_attempts.empty:
            external_allocated = apply_portfolio_allocator_v25(
                external_attempts,
                spec,
                risk_policy=policy,
                budget=budget,
            )
            external_eval = _evaluate(external_allocated, validation)
            external_allocated.to_csv(output / "external_replication_ledger_v26.csv", index=False)
        _snapshot(external_frames, output / "external_ohlcv_snapshot_v26.csv.gz")

    external_metrics = compact_replication_metrics(external_eval, prefix="external")

    temporal_frames, temporal_errors = fetch_coinex_frames(LOCKED_SYMBOLS, bars=TEMPORAL_WARMUP_BARS)
    temporal_attempts_all = _raw_attempts(spec, temporal_frames, tournament, policy) if temporal_frames else pd.DataFrame()
    temporal_attempts = filter_fresh_temporal_attempts(temporal_attempts_all)
    temporal_allocated = pd.DataFrame()
    temporal_eval = _empty_external_eval()
    if not temporal_attempts.empty:
        temporal_allocated = apply_portfolio_allocator_v25(
            temporal_attempts,
            spec,
            risk_policy=policy,
            budget=budget,
        )
        temporal_eval = _evaluate(temporal_allocated, validation)
        temporal_allocated.to_csv(output / "fresh_temporal_oos_ledger_v26.csv", index=False)
    if temporal_frames:
        _snapshot(temporal_frames, output / "temporal_coinex_ohlcv_snapshot_v26.csv.gz")

    temporal_metrics = compact_replication_metrics(temporal_eval, prefix="temporal")
    decision = qualification_decision(external_metrics, temporal_metrics, policy=qualification)

    external_provenance = _provenance(
        external_frames,
        external_errors,
        source=f"{external_exchange or 'none'} public spot OHLCV via ccxt",
        requested_bars=EXTERNAL_BARS,
    )
    external_provenance["venue_selection_order"] = list(EXTERNAL_EXCHANGE_ORDER)
    external_provenance["venue_attempts"] = venue_attempts
    temporal_provenance = _provenance(
        temporal_frames,
        temporal_errors,
        source="CoinEx public spot OHLCV",
        requested_bars=TEMPORAL_WARMUP_BARS,
    )
    temporal_provenance["strict_signal_time_after"] = V25_SELECTION_LOCK_UTC.isoformat()
    temporal_provenance["post_lock_raw_attempts"] = int(len(temporal_attempts))

    manifest = {
        "version": "v0.26",
        "experiment": "LOCKED_WINNER_FRESH_EXTERNAL_AND_TEMPORAL_OOS_MONITOR",
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "locked_source": {
            "winner": LOCKED_WINNER_V26,
            "timeframe": LOCKED_TIMEFRAME_V26,
            "v25_workflow_run_id": V25_WORKFLOW_RUN_ID,
            "v25_artifact_id": V25_ARTIFACT_ID,
            "v25_selection_lock_utc": V25_SELECTION_LOCK_UTC.isoformat(),
            "strategy_retuning": False,
        },
        "frozen_gates": {
            "min_external_trades": qualification.min_external_trades,
            "min_temporal_trades": qualification.min_temporal_trades,
            "min_profit_factor": qualification.min_profit_factor,
            "expectancy_r": ">0",
            "min_positive_asset_fraction": qualification.min_positive_asset_fraction,
            "max_drawdown": qualification.max_drawdown,
            "block_ci_low": ">0",
            "aggregate_open_risk_cap": budget.aggregate_open_risk_cap,
            "directional_open_risk_cap": budget.directional_open_risk_cap,
            "round_trip_cost_fraction": ROUND_TRIP_COST_FRACTION,
        },
        "external_provenance": external_provenance,
        "temporal_provenance": temporal_provenance,
        "external_metrics": external_metrics,
        "temporal_metrics": temporal_metrics,
        "decision": decision,
        "research_contract": {
            "threshold_relaxation": False,
            "winner_reselection": False,
            "historical_test_recycling": False,
            "live_execution_authorized": False,
        },
    }

    (output / "external_metrics_v26.json").write_text(json.dumps(_safe(external_metrics), indent=2, sort_keys=True), encoding="utf-8")
    (output / "temporal_metrics_v26.json").write_text(json.dumps(_safe(temporal_metrics), indent=2, sort_keys=True), encoding="utf-8")
    (output / "decision_v26.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "manifest_v26.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
