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
from research_bot.cross_venue_robustness_v27 import V27RobustnessPolicy, compact_external_metrics
from research_bot.drawdown_firewall_v31 import (
    TOTAL_EFFECTIVE_TRIALS_V31,
    allocator_diagnostics_v31,
    apply_drawdown_firewall_v31,
    preregistration_manifest_v31,
)
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig, simulate_v20_trades
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25
from research_bot.qualification_v31 import screen_v31_candidate, select_v31_winner, v31_decision
from research_bot.regime_event_alpha_v30 import V30_CANDIDATES, cross_sectional_dispersion, generate_direction_v30


LOCKED_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LTC/USDT", "BCH/USDT", "LINK/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
    "ETC/USDT", "ATOM/USDT", "XLM/USDT", "UNI/USDT", "FIL/USDT", "AAVE/USDT",
    "NEAR/USDT", "ALGO/USDT",
]
BARS_BY_TIMEFRAME = {"4h": 3000, "1d": 1800}
MIN_VENUE_SYMBOLS = 15
V30_SOURCE_RUN_ID = 34632401230
V30_SOURCE_ARTIFACT_ID = 10277250718
V30_SOURCE_ARTIFACT_SHA256 = "ba172dae4a7aca13a442b19707dd0a1da7ad63dedafa431019cce8d1c575ad0c"


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(path, compression="gzip")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in x.groupby("symbol", sort=True):
        frames[str(symbol)] = group[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
    return frames


def frame_digest(frame: pd.DataFrame) -> str:
    x = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).astype(str)
    return hashlib.sha256(x.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()


def snapshot(frames: dict[str, pd.DataFrame], path: Path) -> None:
    parts = []
    for symbol, frame in sorted(frames.items()):
        x = frame.copy()
        x.insert(0, "symbol", symbol)
        parts.append(x)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, compression="gzip")


def fetch_kucoin(
    timeframe: str,
    *,
    bars: int,
    as_of: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    exchange = ccxt.kucoin({"enableRateLimit": True})
    exchange.load_markets()
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    end_ms = int(as_of.timestamp() * 1000)
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    try:
        for symbol in LOCKED_SYMBOLS:
            if symbol not in exchange.markets:
                errors[symbol] = "market unavailable"
                continue
            try:
                cursor = end_ms - int((bars + 120) * step_ms)
                rows: list[list[float]] = []
                loops = 0
                while cursor < end_ms and len(rows) < bars + 120 and loops < 160:
                    loops += 1
                    limit = min(500, bars + 120 - len(rows))
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
                raw = raw.loc[raw["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= as_of].tail(bars).reset_index(drop=True)
                minimum = min(600, max(250, bars // 3))
                if len(raw) >= minimum:
                    frames[symbol] = raw
                else:
                    errors[symbol] = f"insufficient rows={len(raw)}"
            except Exception as exc:  # pragma: no cover
                errors[symbol] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    return frames, errors


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
        direction, features = generate_direction_v30(candidate, frame, market_frame=market, dispersion=dispersion)
        ledger = simulate_v20_trades(spec, frame, direction, features, symbol, tournament=tournament, policy=risk_policy)
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def empty_eval() -> dict[str, Any]:
    return {
        "external_trades": 0,
        "external_profit_factor": np.nan,
        "external_expectancy_r": np.nan,
        "external_max_drawdown": np.nan,
        "external_positive_asset_fraction": 0.0,
        "external_block_ci_low": np.nan,
        "external_block_ci_high": np.nan,
        "external_pass_v25": False,
    }


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
    allocated = apply_drawdown_firewall_v31(raw, candidate.strategy_spec(), risk_policy=risk_policy, budget=budget)
    metrics = dict(evaluate_external_replication_v25(allocated, validation=validation))
    metrics.update(allocator_diagnostics_v31(allocated))
    return allocated, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.31 drawdown-firewall paired tournament")
    ap.add_argument("--v30-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v31-drawdown-firewall")
    args = ap.parse_args()

    source = Path(args.v30_artifact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v30_manifest = json.loads((source / "preregistration_v30.json").read_text(encoding="utf-8"))
    as_of = pd.Timestamp(v30_manifest["as_of_utc"])
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    risk = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    qualification = V27RobustnessPolicy()
    validation = V20ValidationConfig(min_external_trades=qualification.min_holdout_trades)
    tournament = TournamentConfig(risk_per_trade=risk.base_risk_per_trade, min_pretest_trades=1000, min_test_trades=qualification.min_holdout_trades)

    manifest = preregistration_manifest_v31()
    manifest.update({
        "as_of_utc": as_of.isoformat(),
        "source_v30_run_id": V30_SOURCE_RUN_ID,
        "source_v30_artifact_id": V30_SOURCE_ARTIFACT_ID,
        "source_v30_artifact_sha256": V30_SOURCE_ARTIFACT_SHA256,
    })
    write_json(output / "preregistration_v31.json", manifest)

    frames_by_venue: dict[str, dict[str, dict[str, pd.DataFrame]]] = {"coinex_consumed": {}, "okx_consumed": {}}
    source_files: dict[str, dict[str, Any]] = {}
    for venue, stem in [("coinex_consumed", "coinex"), ("okx_consumed", "okx")]:
        source_files[venue] = {}
        for timeframe in ("4h", "1d"):
            path = source / f"{stem}_{timeframe}_ohlcv_v30.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            frames = load_snapshot(path)
            frames_by_venue[venue][timeframe] = frames
            source_files[venue][timeframe] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "symbol_count": len(frames),
                "rows": {k: int(len(v)) for k, v in sorted(frames.items())},
                "frame_sha256": {k: frame_digest(v) for k, v in sorted(frames.items())},
            }
    write_json(output / "frozen_development_sources_v31.json", source_files)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, dict[str, Any]] = {}
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in V30_CANDIDATES:
        detailed[candidate.name] = {}
        compact: dict[str, dict[str, Any]] = {}
        for venue in ("coinex_consumed", "okx_consumed"):
            frames = frames_by_venue[venue][candidate.timeframe]
            if len(frames) < MIN_VENUE_SYMBOLS or "BTC/USDT" not in frames:
                evaluated = empty_eval()
                allocated = pd.DataFrame()
            else:
                allocated, evaluated = evaluate_candidate(
                    candidate,
                    frames,
                    tournament=tournament,
                    risk_policy=risk,
                    budget=budget,
                    validation=validation,
                )
            ledgers[(candidate.name, venue)] = allocated
            detailed[candidate.name][venue] = evaluated
            compact[venue] = compact_external_metrics(evaluated, venue=venue)

        screen = screen_v31_candidate(compact["coinex_consumed"], compact["okx_consumed"])
        rows.append({
            "strategy": candidate.name,
            "timeframe": candidate.timeframe,
            "family": candidate.family,
            "coinex_trades": compact["coinex_consumed"].get("trades"),
            "coinex_profit_factor": compact["coinex_consumed"].get("profit_factor"),
            "coinex_expectancy_r": compact["coinex_consumed"].get("expectancy_r"),
            "coinex_max_drawdown": compact["coinex_consumed"].get("max_drawdown"),
            "coinex_positive_asset_fraction": compact["coinex_consumed"].get("positive_asset_fraction"),
            "coinex_block_ci_low": compact["coinex_consumed"].get("block_ci_low"),
            "okx_trades": compact["okx_consumed"].get("trades"),
            "okx_profit_factor": compact["okx_consumed"].get("profit_factor"),
            "okx_expectancy_r": compact["okx_consumed"].get("expectancy_r"),
            "okx_max_drawdown": compact["okx_consumed"].get("max_drawdown"),
            "okx_positive_asset_fraction": compact["okx_consumed"].get("positive_asset_fraction"),
            "okx_block_ci_low": compact["okx_consumed"].get("block_ci_low"),
            "coinex_firewall_scaled_trades": evaluated.get("firewall_scaled_trades") if False else detailed[candidate.name]["coinex_consumed"].get("firewall_scaled_trades"),
            "okx_firewall_scaled_trades": detailed[candidate.name]["okx_consumed"].get("firewall_scaled_trades"),
            **screen,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["development_eligible_v27", "robust_floor_block_ci_low_v27", "robust_floor_breadth_v27", "robust_floor_profit_factor_v27", "robust_floor_expectancy_r_v27"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(output / "development_summary_v31.csv", index=False)
    write_json(output / "development_metrics_v31.json", detailed)

    winner = select_v31_winner(rows)
    locked_at = pd.Timestamp.now(tz="UTC")
    lock = {
        "version": "v0.31",
        "locked_at_utc": locked_at.isoformat(),
        "as_of_utc": as_of.isoformat(),
        "effective_trials": TOTAL_EFFECTIVE_TRIALS_V31,
        "winner": None if winner is None else str(winner["strategy"]),
        "timeframe": None if winner is None else str(winner["timeframe"]),
        "development_venues": ["coinex_consumed", "okx_consumed"],
        "reserved_holdout_venue": "kucoin",
        "holdout_inspected_before_lock": False,
        "hard_drawdown_cap": 0.05,
        "alpha_parameter_retuning": False,
        "qualification_threshold_relaxation": False,
        "live_execution_authorized": False,
    }
    write_json(output / "development_lock_v31.json", lock)

    holdout_metrics: dict[str, Any] | None = None
    holdout_provenance: dict[str, Any] | None = None
    if winner is not None:
        candidate = next(c for c in V30_CANDIDATES if c.name == str(winner["strategy"]))
        frames, errors = fetch_kucoin(candidate.timeframe, bars=BARS_BY_TIMEFRAME[candidate.timeframe], as_of=as_of)
        holdout_provenance = {
            "venue": "kucoin",
            "timeframe": candidate.timeframe,
            "as_of_utc": as_of.isoformat(),
            "symbol_count": len(frames),
            "errors": errors,
            "rows": {k: int(len(v)) for k, v in sorted(frames.items())},
            "sha256_by_symbol": {k: frame_digest(v) for k, v in sorted(frames.items())},
        }
        if frames:
            snapshot(frames, output / f"kucoin_{candidate.timeframe}_ohlcv_v31.csv.gz")
        if len(frames) >= MIN_VENUE_SYMBOLS and "BTC/USDT" in frames:
            allocated, evaluated = evaluate_candidate(candidate, frames, tournament=tournament, risk_policy=risk, budget=budget, validation=validation)
            holdout_metrics = compact_external_metrics(evaluated, venue="kucoin")
            holdout_metrics.update(allocator_diagnostics_v31(allocated))
            if not allocated.empty:
                allocated.to_csv(output / "kucoin_holdout_ledger_v31.csv", index=False)

    decision = v31_decision(
        winner,
        holdout_metrics,
        holdout_available=(winner is None or (holdout_provenance is not None and holdout_provenance["symbol_count"] >= MIN_VENUE_SYMBOLS)),
    )
    write_json(output / "holdout_provenance_v31.json", holdout_provenance)
    write_json(output / "holdout_metrics_v31.json", holdout_metrics)
    write_json(output / "decision_v31.json", decision)
    write_json(output / "manifest_v31.json", {
        **manifest,
        "development_eligible_count": int(summary["development_eligible_v27"].fillna(False).sum()),
        "development_lock": lock,
        "decision": decision,
        "holdout_provenance": holdout_provenance,
        "holdout_metrics": holdout_metrics,
    })

    print(json.dumps(safe({
        "development_eligible_count": int(summary["development_eligible_v27"].fillna(False).sum()),
        "winner": lock["winner"],
        "timeframe": lock["timeframe"],
        "decision": decision,
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
