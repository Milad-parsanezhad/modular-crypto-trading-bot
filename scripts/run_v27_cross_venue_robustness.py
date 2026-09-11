from __future__ import annotations

"""v0.27 two-venue development robustness tournament + untouched KuCoin holdout.

CoinEx validation and OKX are consumed development evidence. The strategy
registry and v0.25 portfolio allocator are frozen. Exactly one development
winner may be locked before KuCoin is fetched. KuCoin is never used to choose
among candidates.
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

from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.cross_venue_robustness_v27 import (
    DEVELOPMENT_VENUES_V27,
    FINAL_HOLDOUT_VENUE_V27,
    V27RobustnessPolicy,
    compact_external_metrics,
    development_screen,
    select_development_winner,
    v27_decision,
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


FROZEN_V20_RUN_ID = 34482338133
FROZEN_V20_ARTIFACT = "v20-global-strategy-risk-lab"
DEVELOPMENT_EXCHANGE = "okx"
HOLDOUT_EXCHANGE = FINAL_HOLDOUT_VENUE_V27
LOCKED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]
BARS_BY_TIMEFRAME = {"1m": 8000, "5m": 7000, "15m": 6000, "1h": 5000, "4h": 3000, "1d": 1800}
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


def _snapshot(frames: dict[str, pd.DataFrame], path: Path) -> None:
    parts: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        x = frame.copy()
        x.insert(0, "symbol", symbol)
        parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(path, index=False, compression="gzip")


def _provenance(frames: dict[str, pd.DataFrame], errors: dict[str, str], *, exchange: str, timeframe: str, bars: int) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "requested_bars_per_symbol": bars,
        "successful_symbols": sorted(frames),
        "symbol_count": len(frames),
        "rows": {symbol: int(len(frame)) for symbol, frame in sorted(frames.items())},
        "ranges": {
            symbol: {
                "first": pd.to_datetime(frame["timestamp"], utc=True).min().isoformat(),
                "last": pd.to_datetime(frame["timestamp"], utc=True).max().isoformat(),
            }
            for symbol, frame in sorted(frames.items()) if len(frame)
        },
        "sha256_by_symbol": {symbol: _frame_digest(frame) for symbol, frame in sorted(frames.items())},
        "failures": errors,
    }


def fetch_ccxt_timeframe(exchange_id: str, timeframe: str, symbols: list[str], *, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
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
                while cursor < now_ms and len(rows) < bars + 80 and loops < 120:
                    loops += 1
                    batch = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=cursor,
                        limit=min(300, bars + 80 - len(rows)),
                    )
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
                frame = _drop_incomplete(frame.drop_duplicates("timestamp").sort_values("timestamp"), step_ms)
                frame = frame.tail(bars).reset_index(drop=True)
                minimum = min(600, max(250, bars // 3))
                if len(frame) >= minimum:
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


def _peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((frame for other, frame in frames.items() if other != symbol), None)


def _raw_attempts(spec: Any, frames: dict[str, pd.DataFrame], tournament: TournamentConfig, policy: RiskPsychologyPolicy) -> pd.DataFrame:
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
            policy=policy,
        )
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def _evaluate_raw(raw: pd.DataFrame, spec: Any, *, policy: RiskPsychologyPolicy, budget: PortfolioRiskBudgetV25, validation: V20ValidationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        return pd.DataFrame(), _empty_eval()
    allocated = apply_portfolio_allocator_v25(raw, spec, risk_policy=policy, budget=budget)
    return allocated, evaluate_external_replication_v25(allocated, validation=validation)


def _flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "venue"}


def _load_frozen_coinex_validation(artifact_dir: Path) -> pd.DataFrame:
    path = artifact_dir / "trade_attempt_ledger.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen v0.20 ledger: {path}")
    frame = pd.read_csv(path)
    for column in ("signal_time", "entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    required = {"strategy", "segment", "entry_time", "exit_time", "symbol", "r_multiple", "side", "entry", "stop", "risk_scale_volatility"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frozen v0.20 ledger missing columns: {sorted(missing)}")
    return frame.loc[frame["segment"].astype(str) == "validation"].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.27 cross-venue robustness tournament")
    ap.add_argument("--v20-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v27-cross-venue-robustness")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    policy = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    robustness = V27RobustnessPolicy()
    validation = V20ValidationConfig(min_external_trades=robustness.min_holdout_trades)
    tournament = TournamentConfig(
        risk_per_trade=policy.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=robustness.min_holdout_trades,
    )

    frozen_validation = _load_frozen_coinex_validation(Path(args.v20_artifact_dir))
    coinex_evals: dict[str, dict[str, Any]] = {}
    for spec in STRATEGY_REGISTRY_V20:
        raw = frozen_validation.loc[frozen_validation["strategy"].astype(str) == spec.name].copy()
        _, evaluated = _evaluate_raw(raw, spec, policy=policy, budget=budget, validation=validation)
        coinex_evals[spec.name] = evaluated

    okx_frames_by_tf: dict[str, dict[str, pd.DataFrame]] = {}
    okx_provenance: dict[str, Any] = {}
    okx_evals: dict[str, dict[str, Any]] = {}

    for timeframe, bars in BARS_BY_TIMEFRAME.items():
        frames, errors = fetch_ccxt_timeframe(DEVELOPMENT_EXCHANGE, timeframe, LOCKED_SYMBOLS, bars=bars)
        okx_frames_by_tf[timeframe] = frames
        okx_provenance[timeframe] = _provenance(frames, errors, exchange=DEVELOPMENT_EXCHANGE, timeframe=timeframe, bars=bars)
        if frames:
            _snapshot(frames, output / f"okx_{timeframe}_ohlcv_v27.csv.gz")
        for spec in [s for s in STRATEGY_REGISTRY_V20 if s.timeframe == timeframe]:
            if len(frames) < MIN_VENUE_SYMBOLS:
                okx_evals[spec.name] = _empty_eval()
                continue
            raw = _raw_attempts(spec, frames, tournament, policy)
            _, evaluated = _evaluate_raw(raw, spec, policy=policy, budget=budget, validation=validation)
            okx_evals[spec.name] = evaluated

    rows: list[dict[str, Any]] = []
    for spec in STRATEGY_REGISTRY_V20:
        coinex = compact_external_metrics(coinex_evals.get(spec.name, _empty_eval()), venue="coinex_frozen")
        okx = compact_external_metrics(okx_evals.get(spec.name, _empty_eval()), venue="okx_consumed")
        screen = development_screen(coinex, okx, policy=robustness)
        rows.append({
            "strategy": spec.name,
            "timeframe": spec.timeframe,
            "family": spec.family,
            **_flatten("coinex", coinex),
            **_flatten("okx", okx),
            **screen,
        })

    summary = pd.DataFrame(rows).sort_values(
        [
            "development_eligible_v27",
            "robust_floor_block_ci_low_v27",
            "robust_floor_breadth_v27",
            "robust_floor_profit_factor_v27",
            "robust_floor_expectancy_r_v27",
        ],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(output / "development_summary_v27.csv", index=False)

    winner = select_development_winner(rows)
    selection_lock_utc = pd.Timestamp.now(tz="UTC")
    lock_payload = {
        "version": "v0.27",
        "locked_at_utc": selection_lock_utc.isoformat(),
        "winner": None if winner is None else str(winner["strategy"]),
        "timeframe": None if winner is None else str(winner["timeframe"]),
        "candidate_count": len(STRATEGY_REGISTRY_V20),
        "development_venues": list(DEVELOPMENT_VENUES_V27),
        "reserved_holdout_venue": HOLDOUT_EXCHANGE,
        "strategy_parameter_retuning": False,
        "threshold_relaxation": False,
        "holdout_inspected_before_lock": False,
    }
    (output / "development_lock_v27.json").write_text(json.dumps(_safe(lock_payload), indent=2, sort_keys=True), encoding="utf-8")

    holdout_compact: dict[str, Any] | None = None
    holdout_provenance: dict[str, Any] | None = None
    holdout_eval = _empty_eval()

    # The reserved venue is touched only after the single winner has been locked.
    if winner is not None:
        spec = next(s for s in STRATEGY_REGISTRY_V20 if s.name == str(winner["strategy"]))
        bars = BARS_BY_TIMEFRAME[spec.timeframe]
        holdout_frames, holdout_errors = fetch_ccxt_timeframe(HOLDOUT_EXCHANGE, spec.timeframe, LOCKED_SYMBOLS, bars=bars)
        holdout_provenance = _provenance(
            holdout_frames,
            holdout_errors,
            exchange=HOLDOUT_EXCHANGE,
            timeframe=spec.timeframe,
            bars=bars,
        )
        if holdout_frames:
            _snapshot(holdout_frames, output / "kucoin_holdout_ohlcv_v27.csv.gz")
        if len(holdout_frames) >= MIN_VENUE_SYMBOLS:
            raw = _raw_attempts(spec, holdout_frames, tournament, policy)
            allocated, holdout_eval = _evaluate_raw(raw, spec, policy=policy, budget=budget, validation=validation)
            if not allocated.empty:
                allocated.to_csv(output / "kucoin_holdout_ledger_v27.csv", index=False)
            holdout_compact = compact_external_metrics(holdout_eval, venue=HOLDOUT_EXCHANGE)

    decision = v27_decision(winner, holdout_compact, policy=robustness)

    selected_row = None
    if winner is not None:
        selected_row = next(row for row in rows if row["strategy"] == winner["strategy"])

    manifest = {
        "version": "v0.27",
        "experiment": "CONSUMED_TWO_VENUE_ROBUSTNESS_SELECTION_THEN_UNTOUCHED_KUCOIN_HOLDOUT",
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "frozen_source": {
            "coinex_workflow_run_id": FROZEN_V20_RUN_ID,
            "coinex_artifact_name": FROZEN_V20_ARTIFACT,
            "coinex_segment": "validation",
        },
        "selection_contract": {
            "candidate_count": len(STRATEGY_REGISTRY_V20),
            "development_venues": list(DEVELOPMENT_VENUES_V27),
            "min_development_trades_per_venue": robustness.min_development_trades_per_venue,
            "min_profit_factor": robustness.min_profit_factor,
            "expectancy_r": ">0 on both development venues",
            "max_drawdown": robustness.max_drawdown,
            "ranking": "maximize worst-venue Block-CI, then breadth, PF, expectancy, trade count, then minimize worst drawdown",
            "parameter_retuning": False,
            "threshold_relaxation": False,
        },
        "development_eligible_count": int(summary["development_eligible_v27"].fillna(False).astype(bool).sum()),
        "development_winner": _safe(selected_row),
        "development_lock": lock_payload,
        "okx_provenance": okx_provenance,
        "holdout_contract": {
            "venue": HOLDOUT_EXCHANGE,
            "winner_reselection_allowed": False,
            "min_trades": robustness.min_holdout_trades,
            "min_profit_factor": robustness.min_profit_factor,
            "expectancy_r": ">0",
            "min_positive_asset_fraction": robustness.min_positive_asset_fraction,
            "max_drawdown": robustness.max_drawdown,
            "block_ci_low": ">0",
        },
        "holdout_provenance": holdout_provenance,
        "holdout_metrics": holdout_compact,
        "decision": decision,
        "next_if_holdout_passes": "Start a new strictly post-v0.27-lock temporal OOS monitor; do not reuse KuCoin for selection or retuning.",
    }

    (output / "holdout_metrics_v27.json").write_text(json.dumps(_safe(holdout_compact), indent=2, sort_keys=True), encoding="utf-8")
    (output / "decision_v27.json").write_text(json.dumps(_safe(decision), indent=2, sort_keys=True), encoding="utf-8")
    (output / "manifest_v27.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_safe({
        "development_eligible_count": manifest["development_eligible_count"],
        "winner": lock_payload["winner"],
        "timeframe": lock_payload["timeframe"],
        "holdout_metrics": holdout_compact,
        "decision": decision,
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
