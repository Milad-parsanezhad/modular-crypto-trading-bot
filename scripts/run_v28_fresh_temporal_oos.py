from __future__ import annotations

"""v0.28 strictly-post-lock temporal OOS runner.

The runner is intentionally fail-closed. It reads v0.27 evidence first and
performs no market-data request unless the untouched KuCoin holdout has already
passed exactly as preregistered.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.fresh_temporal_oos_v28 import (
    V28TemporalPolicy,
    compact_temporal_metrics,
    filter_strictly_post_lock,
    v27_prerequisite_failures,
    v28_decision,
)
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
    V20ValidationConfig,
    generate_direction_v20,
    simulate_v20_trades,
)
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25, apply_portfolio_allocator_v25


LOCKED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]
COINEX_PERIOD_BY_TIMEFRAME = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day",
}
WARMUP_BARS_BY_TIMEFRAME = {
    "1m": 8000,
    "5m": 7000,
    "15m": 6000,
    "1h": 5000,
    "4h": 3200,
    "1d": 2200,
}
MIN_VENUE_SYMBOLS = 15


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _empty_eval() -> dict[str, Any]:
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


def _peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((frame for other, frame in frames.items() if other != symbol), None)


def _drop_incomplete(frame: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    now = pd.Timestamp.now(tz="UTC")
    return x.loc[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= now].reset_index(drop=True)


def _frame_digest(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    x = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).astype(str)
    return hashlib.sha256(x.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()


def fetch_coinex_frames(timeframe: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    period = COINEX_PERIOD_BY_TIMEFRAME[timeframe]
    bars = WARMUP_BARS_BY_TIMEFRAME[timeframe]
    end_ms = utc_now_ms()
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in LOCKED_SYMBOLS:
        try:
            frame = fetch_coinex_klines(
                symbol=symbol,
                period=period,
                market_type="spot",
                end_ms=end_ms,
                bars=bars,
            )
            frame = _drop_incomplete(frame, PERIOD_MS[period]).tail(bars).reset_index(drop=True)
            if len(frame) >= min(600, max(250, bars // 3)):
                frames[symbol] = frame
            else:
                errors[symbol] = f"insufficient rows={len(frame)}"
        except Exception as exc:  # pragma: no cover - network/runtime path
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return frames, errors


def _raw_attempts(spec: Any, frames: dict[str, pd.DataFrame], tournament: TournamentConfig, risk: RiskPsychologyPolicy) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        peer = _peer_for(symbol, frames) if ("CORRELATION" in spec.name or spec.family == "correlation_divergence") else None
        direction, features = generate_direction_v20(spec, frame, peer=peer)
        ledger = simulate_v20_trades(
            spec,
            frame,
            direction,
            features,
            symbol,
            tournament=tournament,
            policy=risk,
        )
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.28 strictly fresh temporal OOS")
    ap.add_argument("--v27-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v28-fresh-temporal-oos")
    args = ap.parse_args()

    source = Path(args.v27_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v27_decision_payload = json.loads((source / "decision_v27.json").read_text(encoding="utf-8"))
    development_lock = json.loads((source / "development_lock_v27.json").read_text(encoding="utf-8"))
    prerequisite_failures = v27_prerequisite_failures(v27_decision_payload, development_lock)

    policy = V28TemporalPolicy()

    # Fail closed before any network access.  A failed/non-final v0.27 may not
    # consume new temporal data because there is no qualified frozen candidate.
    if prerequisite_failures:
        decision = v28_decision(v27_decision_payload, development_lock, None, policy=policy)
        manifest = {
            "version": "v0.28",
            "experiment": "STRICTLY_POST_V27_LOCK_TEMPORAL_OOS",
            "network_access_performed": False,
            "v27_prerequisite_failures": prerequisite_failures,
            "decision": decision,
        }
        (output / "decision_v28.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
        (output / "manifest_v28.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(_safe(decision), indent=2, sort_keys=True))
        return

    winner = str(v27_decision_payload["winner"])
    timeframe = str(v27_decision_payload["timeframe"])
    spec = next((s for s in STRATEGY_REGISTRY_V20 if s.name == winner), None)
    if spec is None or spec.timeframe != timeframe:
        raise RuntimeError("v0.28 frozen winner/timeframe does not match strategy registry")

    risk = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    validation = V20ValidationConfig(min_external_trades=policy.min_trades)
    tournament = TournamentConfig(
        risk_per_trade=risk.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=policy.min_trades,
    )

    frames, errors = fetch_coinex_frames(timeframe)
    raw = _raw_attempts(spec, frames, tournament, risk) if len(frames) >= MIN_VENUE_SYMBOLS else pd.DataFrame()
    fresh_raw = filter_strictly_post_lock(raw, development_lock)
    allocated = pd.DataFrame()
    evaluated = _empty_eval()
    if not fresh_raw.empty:
        allocated = apply_portfolio_allocator_v25(fresh_raw, spec, risk_policy=risk, budget=budget)
        evaluated = evaluate_external_replication_v25(allocated, validation=validation)
        allocated.to_csv(output / "fresh_temporal_ledger_v28.csv", index=False)

    metrics = compact_temporal_metrics(evaluated)
    decision = v28_decision(v27_decision_payload, development_lock, metrics, policy=policy)

    provenance = {
        "source": "CoinEx public spot OHLCV",
        "timeframe": timeframe,
        "requested_symbols": LOCKED_SYMBOLS,
        "successful_symbols": sorted(frames),
        "symbol_count": len(frames),
        "failures": errors,
        "rows": {symbol: int(len(frame)) for symbol, frame in sorted(frames.items())},
        "sha256_by_symbol": {symbol: _frame_digest(frame) for symbol, frame in sorted(frames.items())},
        "strict_signal_time_after": development_lock["locked_at_utc"],
        "post_lock_raw_attempts": int(len(fresh_raw)),
    }
    manifest = {
        "version": "v0.28",
        "experiment": "STRICTLY_POST_V27_LOCK_TEMPORAL_OOS",
        "winner": winner,
        "timeframe": timeframe,
        "temporal_anchor_utc": development_lock["locked_at_utc"],
        "frozen_gates": {
            "min_trades": policy.min_trades,
            "min_profit_factor": policy.min_profit_factor,
            "expectancy_r": ">0",
            "min_positive_asset_fraction": policy.min_positive_asset_fraction,
            "max_drawdown": policy.max_drawdown,
            "block_ci_low": ">0",
            "aggregate_open_risk_cap": budget.aggregate_open_risk_cap,
            "directional_open_risk_cap": budget.directional_open_risk_cap,
        },
        "network_access_performed": True,
        "provenance": provenance,
        "metrics": metrics,
        "decision": decision,
        "research_contract": {
            "threshold_relaxation": False,
            "strategy_parameter_retuning": False,
            "winner_reselection": False,
            "historical_test_recycling": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        },
    }

    (output / "metrics_v28.json").write_text(json.dumps(_safe(metrics), indent=2, sort_keys=True), encoding="utf-8")
    (output / "decision_v28.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "manifest_v28.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
