from __future__ import annotations

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
from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines
from research_bot.cross_venue_robustness_v27 import V27RobustnessPolicy, compact_external_metrics
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig, simulate_v20_trades
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25, apply_portfolio_allocator_v25
from research_bot.qualification_v30 import screen_v30_candidate, select_v30_winner, v30_decision
from research_bot.regime_event_alpha_v30 import (
    DEVELOPMENT_VENUES_V30,
    FINAL_HOLDOUT_VENUE_V30,
    TOTAL_EFFECTIVE_TRIALS_V30,
    V30_CANDIDATES,
    cross_sectional_dispersion,
    generate_direction_v30,
    preregistration_manifest_v30,
)


LOCKED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]
BARS_BY_TIMEFRAME = {"4h": 3000, "1d": 1800}
COINEX_PERIOD = {"4h": "4hour", "1d": "1day"}
MIN_VENUE_SYMBOLS = 15
DEVELOPMENT_CCXT_EXCHANGE = "okx"
HOLDOUT_CCXT_EXCHANGE = FINAL_HOLDOUT_VENUE_V30


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def drop_incomplete(frame: pd.DataFrame, step_ms: int, as_of: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    return x.loc[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= as_of].reset_index(drop=True)


def frame_digest(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    x = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).astype(str)
    return hashlib.sha256(x.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()


def snapshot(frames: dict[str, pd.DataFrame], path: Path) -> None:
    parts: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        x = frame.copy()
        x.insert(0, "symbol", symbol)
        parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(path, index=False, compression="gzip")


def provenance(
    frames: dict[str, pd.DataFrame],
    errors: dict[str, str],
    *,
    venue: str,
    timeframe: str,
    bars: int,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "venue": venue,
        "timeframe": timeframe,
        "as_of_utc": as_of.isoformat(),
        "requested_bars_per_symbol": bars,
        "symbol_count": len(frames),
        "successful_symbols": sorted(frames),
        "rows": {k: int(len(v)) for k, v in sorted(frames.items())},
        "ranges": {
            k: {
                "first": pd.to_datetime(v["timestamp"], utc=True).min().isoformat(),
                "last": pd.to_datetime(v["timestamp"], utc=True).max().isoformat(),
            }
            for k, v in sorted(frames.items()) if len(v)
        },
        "sha256_by_symbol": {k: frame_digest(v) for k, v in sorted(frames.items())},
        "failures": errors,
    }


def fetch_coinex_timeframe(
    timeframe: str,
    symbols: list[str],
    *,
    bars: int,
    as_of: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    period = COINEX_PERIOD[timeframe]
    step_ms = PERIOD_MS[period]
    end_ms = int(as_of.timestamp() * 1000)
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            raw = fetch_coinex_klines(
                symbol=symbol,
                period=period,
                market_type="spot",
                end_ms=end_ms,
                bars=bars + 120,
            )
            raw = drop_incomplete(raw, step_ms, as_of).tail(bars).reset_index(drop=True)
            minimum = min(600, max(250, bars // 3))
            if len(raw) >= minimum:
                frames[symbol] = raw
            else:
                errors[symbol] = f"insufficient rows={len(raw)}"
        except Exception as exc:  # pragma: no cover - network path
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return frames, errors


def fetch_ccxt_timeframe(
    exchange_id: str,
    timeframe: str,
    symbols: list[str],
    *,
    bars: int,
    as_of: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    end_ms = int(as_of.timestamp() * 1000)
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    try:
        for symbol in symbols:
            if symbol not in exchange.markets:
                errors[symbol] = "market unavailable"
                continue
            try:
                cursor = end_ms - int((bars + 120) * step_ms)
                rows: list[list[float]] = []
                loops = 0
                while cursor < end_ms and len(rows) < bars + 120 and loops < 140:
                    loops += 1
                    limit = min(300, bars + 120 - len(rows))
                    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                    if not batch:
                        break
                    rows.extend(row for row in batch if int(row[0]) < end_ms)
                    nxt = int(batch[-1][0]) + step_ms
                    if nxt <= cursor:
                        break
                    cursor = nxt
                    time.sleep(exchange.rateLimit / 1000.0 if exchange.rateLimit else 0.05)
                if not rows:
                    errors[symbol] = "no rows"
                    continue
                raw = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)
                raw = raw[["timestamp", "open", "high", "low", "close", "volume"]]
                raw = raw.drop_duplicates("timestamp").sort_values("timestamp")
                raw = drop_incomplete(raw, step_ms, as_of).tail(bars).reset_index(drop=True)
                minimum = min(600, max(250, bars // 3))
                if len(raw) >= minimum:
                    frames[symbol] = raw
                else:
                    errors[symbol] = f"insufficient rows={len(raw)}"
            except Exception as exc:  # pragma: no cover - network path
                errors[symbol] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    return frames, errors


def empty_eval() -> dict[str, Any]:
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


def raw_attempts(
    candidate: Any,
    frames: dict[str, pd.DataFrame],
    *,
    tournament: TournamentConfig,
    risk_policy: RiskPsychologyPolicy,
) -> pd.DataFrame:
    if "BTC/USDT" not in frames:
        return pd.DataFrame()
    dispersion = cross_sectional_dispersion(frames, candidate.timeframe)
    market = frames["BTC/USDT"]
    spec = candidate.strategy_spec()
    parts: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        direction, features = generate_direction_v30(
            candidate,
            frame,
            market_frame=market,
            dispersion=dispersion,
        )
        ledger = simulate_v20_trades(
            spec,
            frame,
            direction,
            features,
            symbol,
            tournament=tournament,
            policy=risk_policy,
        )
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def evaluate_candidate(
    candidate: Any,
    frames: dict[str, pd.DataFrame],
    *,
    tournament: TournamentConfig,
    risk_policy: RiskPsychologyPolicy,
    budget: PortfolioRiskBudgetV25,
    validation: V20ValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = raw_attempts(candidate, frames, tournament=tournament, risk_policy=risk_policy)
    if raw.empty:
        return pd.DataFrame(), empty_eval()
    allocated = apply_portfolio_allocator_v25(raw, candidate.strategy_spec(), risk_policy=risk_policy, budget=budget)
    return allocated, evaluate_external_replication_v25(allocated, validation=validation)


def flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items() if k != "venue"}


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.30 preregistered regime/event-aware robustness tournament")
    ap.add_argument("--output-dir", default="artifacts/v30-regime-event-alpha")
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.now(tz="UTC").floor("s")

    risk = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    qualification = V27RobustnessPolicy()
    validation = V20ValidationConfig(min_external_trades=qualification.min_holdout_trades)
    tournament = TournamentConfig(
        risk_per_trade=risk.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=qualification.min_holdout_trades,
    )

    manifest = preregistration_manifest_v30()
    manifest["as_of_utc"] = as_of.isoformat()
    write_json(output / "preregistration_v30.json", manifest)

    venue_frames: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        "coinex_consumed": {},
        "okx_consumed": {},
    }
    venue_provenance: dict[str, dict[str, Any]] = {"coinex_consumed": {}, "okx_consumed": {}}

    for timeframe, bars in BARS_BY_TIMEFRAME.items():
        coinex_frames, coinex_errors = fetch_coinex_timeframe(
            timeframe, LOCKED_SYMBOLS, bars=bars, as_of=as_of
        )
        okx_frames, okx_errors = fetch_ccxt_timeframe(
            DEVELOPMENT_CCXT_EXCHANGE, timeframe, LOCKED_SYMBOLS, bars=bars, as_of=as_of
        )
        venue_frames["coinex_consumed"][timeframe] = coinex_frames
        venue_frames["okx_consumed"][timeframe] = okx_frames
        venue_provenance["coinex_consumed"][timeframe] = provenance(
            coinex_frames, coinex_errors, venue="coinex", timeframe=timeframe, bars=bars, as_of=as_of
        )
        venue_provenance["okx_consumed"][timeframe] = provenance(
            okx_frames, okx_errors, venue="okx", timeframe=timeframe, bars=bars, as_of=as_of
        )
        if coinex_frames:
            snapshot(coinex_frames, output / f"coinex_{timeframe}_ohlcv_v30.csv.gz")
        if okx_frames:
            snapshot(okx_frames, output / f"okx_{timeframe}_ohlcv_v30.csv.gz")

    rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in V30_CANDIDATES:
        evaluations[candidate.name] = {}
        compact_by_venue: dict[str, dict[str, Any]] = {}
        for venue in DEVELOPMENT_VENUES_V30:
            frames = venue_frames[venue].get(candidate.timeframe, {})
            if len(frames) < MIN_VENUE_SYMBOLS or "BTC/USDT" not in frames:
                evaluated = empty_eval()
            else:
                _, evaluated = evaluate_candidate(
                    candidate,
                    frames,
                    tournament=tournament,
                    risk_policy=risk,
                    budget=budget,
                    validation=validation,
                )
            evaluations[candidate.name][venue] = evaluated
            compact_by_venue[venue] = compact_external_metrics(evaluated, venue=venue)

        screen = screen_v30_candidate(
            compact_by_venue["coinex_consumed"], compact_by_venue["okx_consumed"]
        )
        rows.append(
            {
                "strategy": candidate.name,
                "timeframe": candidate.timeframe,
                "family": candidate.family,
                **flatten("coinex", compact_by_venue["coinex_consumed"]),
                **flatten("okx", compact_by_venue["okx_consumed"]),
                **screen,
            }
        )

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
    summary.to_csv(output / "development_summary_v30.csv", index=False)

    winner = select_v30_winner(rows)
    locked_at = pd.Timestamp.now(tz="UTC")
    lock = {
        "version": "v0.30",
        "locked_at_utc": locked_at.isoformat(),
        "as_of_utc": as_of.isoformat(),
        "candidate_count": len(V30_CANDIDATES),
        "effective_trials": TOTAL_EFFECTIVE_TRIALS_V30,
        "winner": None if winner is None else str(winner["strategy"]),
        "timeframe": None if winner is None else str(winner["timeframe"]),
        "development_venues": list(DEVELOPMENT_VENUES_V30),
        "reserved_holdout_venue": FINAL_HOLDOUT_VENUE_V30,
        "holdout_inspected_before_lock": False,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "live_execution_authorized": False,
    }
    write_json(output / "development_lock_v30.json", lock)

    holdout_compact: dict[str, Any] | None = None
    holdout_provenance: dict[str, Any] | None = None
    if winner is not None:
        candidate = next(c for c in V30_CANDIDATES if c.name == str(winner["strategy"]))
        bars = BARS_BY_TIMEFRAME[candidate.timeframe]
        frames, errors = fetch_ccxt_timeframe(
            HOLDOUT_CCXT_EXCHANGE,
            candidate.timeframe,
            LOCKED_SYMBOLS,
            bars=bars,
            as_of=as_of,
        )
        holdout_provenance = provenance(
            frames,
            errors,
            venue=HOLDOUT_CCXT_EXCHANGE,
            timeframe=candidate.timeframe,
            bars=bars,
            as_of=as_of,
        )
        if frames:
            snapshot(frames, output / f"kucoin_{candidate.timeframe}_ohlcv_v30.csv.gz")
        if len(frames) >= MIN_VENUE_SYMBOLS and "BTC/USDT" in frames:
            allocated, evaluated = evaluate_candidate(
                candidate,
                frames,
                tournament=tournament,
                risk_policy=risk,
                budget=budget,
                validation=validation,
            )
            holdout_compact = compact_external_metrics(evaluated, venue="kucoin")
            if not allocated.empty:
                allocated.to_csv(output / "kucoin_holdout_ledger_v30.csv", index=False)

    decision = v30_decision(
        winner,
        holdout_compact,
        holdout_available=(winner is None or (holdout_provenance is not None and holdout_provenance["symbol_count"] >= MIN_VENUE_SYMBOLS)),
    )
    write_json(output / "decision_v30.json", decision)
    write_json(output / "development_metrics_v30.json", evaluations)
    write_json(output / "data_provenance_v30.json", venue_provenance)
    write_json(output / "holdout_provenance_v30.json", holdout_provenance)
    write_json(output / "holdout_metrics_v30.json", holdout_compact)

    final_manifest = {
        **manifest,
        "development_lock": lock,
        "development_eligible_count": int(summary["development_eligible_v27"].fillna(False).sum()),
        "decision": decision,
        "holdout_provenance": holdout_provenance,
        "holdout_metrics": holdout_compact,
        "research_contract": {
            "threshold_relaxation": False,
            "strategy_parameter_retuning": False,
            "winner_reselection": False,
            "historical_test_recycling": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        },
    }
    write_json(output / "manifest_v30.json", final_manifest)
    print(json.dumps(safe({
        "winner": lock["winner"],
        "timeframe": lock["timeframe"],
        "development_eligible_count": final_manifest["development_eligible_count"],
        "decision": decision,
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
