from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.backtest import backtest_positions
from research_bot.contracts import ExecutionMode, ExperimentManifest
from research_bot.decision import DecisionEngine
from research_bot.execution import ExecutionPolicy, PaperExecutionEngine
from research_bot.ichimoku_advanced import detect_kumo_triangle_breakout
from research_bot.orchestrator import ResearchTradingOrchestrator
from research_bot.reproducibility import dataframe_fingerprint, save_experiment_manifest
from research_bot.risk import RiskEngine, RiskSnapshot
from research_bot.universe import EligibilityPolicy, MarketListing, coverage_report


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _synthetic_ohlcv(n: int = 400) -> pd.DataFrame:
    """Deterministic smoke-test data; NEVER valid for thesis performance claims."""

    rng = np.random.default_rng(808)
    returns = rng.normal(0.0002, 0.009, n)
    close = 50_000 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    volume = rng.lognormal(11, 0.4, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
            "open": np.r_[close[0], close[:-1]],
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _load_input(path: str | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        return _synthetic_ohlcv(), "SYNTHETIC_SMOKE_ONLY"
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    return df, "USER_SUPPLIED_REAL_DATA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default="artifacts/v08")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, data_status = _load_input(args.input_csv)
    features = detect_kumo_triangle_breakout(df)
    fingerprint = dataframe_fingerprint(df)

    # Universe/coverage smoke path.  Values are explicit fixture metadata for the
    # engineering test; they are not claims about current market coverage.
    fixtures = [
        MarketListing(
            exchange="fixture",
            symbol="BTC/USDT",
            base="BTC",
            quote="USDT",
            market_type="spot",
            active=True,
            volume_24h_quote=100_000_000,
            spread_bps=4,
            history_bars=len(df),
            missing_fraction=0.0,
            abnormal_fraction=0.0,
        ),
        MarketListing(
            exchange="fixture",
            symbol="ETH/USDT",
            base="ETH",
            quote="USDT",
            market_type="spot",
            active=True,
            volume_24h_quote=80_000_000,
            spread_bps=5,
            history_bars=len(df),
            missing_fraction=0.0,
            abnormal_fraction=0.0,
        ),
    ]
    _, coverage = coverage_report(fixtures, policy=EligibilityPolicy(min_history_bars=200))

    # Cost-aware backtest smoke: triangle candidate is intentionally evaluated as
    # a raw candidate position, not a profitability claim.
    future_return = features["close"].shift(-1) / features["close"] - 1.0
    position = features["triangle_candidate"].fillna(0.0)
    _, bt_metrics = backtest_positions(
        future_return,
        position,
        timeframe="4h",
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    execution = PaperExecutionEngine(
        ExecutionPolicy(
            mode=ExecutionMode.PAPER,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            max_order_notional=5_000,
        )
    )
    orchestrator = ResearchTradingOrchestrator(
        decision_engine=DecisionEngine(),
        risk_engine=RiskEngine(),
        execution_engine=execution,
    )
    last = features.dropna(subset=["close"]).iloc[-1]
    risk_snapshot = RiskSnapshot(
        equity=10_000,
        peak_equity=10_000,
        gross_exposure=0.0,
        asset_weight=0.0,
        turnover=0.0,
        spread_bps=4.0,
        slippage_bps=args.slippage_bps,
        recent_returns=tuple(features["close"].pct_change().dropna().tail(60).tolist()),
    )
    outcome = orchestrator.process_forecast(
        asset="BTC",
        symbol="BTC/USDT",
        timestamp=pd.Timestamp(last["timestamp"]).to_pydatetime(),
        expected_return=0.008,
        expected_cost=(args.fee_bps + args.slippage_bps) / 10_000.0,
        risk_penalty=0.001,
        uncertainty_penalty=0.001,
        confidence=0.75,
        currently_long=False,
        risk_snapshot=risk_snapshot,
        reference_price=float(last["close"]),
        quantity=min(0.01, 5_000 / float(last["close"])),
        client_order_id="v08-smoke-btc-001",
        metadata={"data_status": data_status},
    )

    manifest = ExperimentManifest.now(
        experiment_id="v08-research-core-smoke",
        git_commit=os.getenv("GITHUB_SHA", os.getenv("GIT_COMMIT", "UNKNOWN")),
        dataset_id=f"sha256:{fingerprint}",
        feature_set="ichimoku_advanced_triangle_v08",
        model_name="FIXED_FORECAST_SMOKE_NOT_A_MODEL",
        random_seed=808,
        execution_mode=ExecutionMode.PAPER,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        notes=f"data_status={data_status}; no profitability claim",
    )
    save_experiment_manifest(manifest, out / "experiment_manifest.json")

    report = {
        "research_status": "ENGINEERING_SMOKE_ONLY" if data_status.startswith("SYNTHETIC") else "RESEARCH_RUN_NOT_VALIDATED_OOS",
        "data_status": data_status,
        "rows": int(len(df)),
        "dataset_sha256": fingerprint,
        "triangle_candidates": int(features["triangle_candidate"].fillna(0).sum()),
        "coverage": asdict(coverage),
        "candidate_backtest": bt_metrics,
        "decision": {
            "status": outcome.status,
            "signal": outcome.signal.decision.value,
            "net_alpha": outcome.signal.net_alpha,
            "reasons": outcome.signal.reasons,
            "risk": asdict(outcome.risk) if outcome.risk is not None else None,
            "fill": asdict(outcome.fill) if outcome.fill is not None else None,
        },
        "warning": "Synthetic smoke data or a single user CSV must never be presented as a validated thesis result without the full OOS protocol.",
    }
    (out / "research_core_report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
