from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ml_meta_v21 import (
    MLConfig,
    build_labeled_candidate_attempts,
    choose_classifier_champion,
    finalize_candidate_risk,
    train_classifier_zoo,
    train_regressor_zoo,
    write_dataset_manifest,
)
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import (
    STRATEGY_REGISTRY_V20,
    RiskPsychologyPolicy,
)

PERIOD_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_BARS = {"1m": 30000, "5m": 24000, "15m": 16000, "1h": 12000, "4h": 8000, "1d": 3000}
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


def fetch_timeframe(symbols: list[str], timeframe: str, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    period, end_ms = PERIOD_MAP[timeframe], utc_now_ms()
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
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


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((f for s, f in frames.items() if s != symbol), None)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.21 leakage-safe labeled strategy-event ML meta lab")
    ap.add_argument("--output-dir", default="artifacts/v21-ml-meta")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--max-fit-rows", type=int, default=60000)
    ap.add_argument("--max-fit-rows-slow", type=int, default=12000)
    ap.add_argument("--min-selected-validation", type=int, default=200)
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    policy = RiskPsychologyPolicy()
    tournament = TournamentConfig(risk_per_trade=policy.base_risk_per_trade, min_pretest_trades=1000, min_test_trades=200)
    cfg = MLConfig(max_fit_rows=args.max_fit_rows, max_fit_rows_slow=args.max_fit_rows_slow, min_selected_validation=args.min_selected_validation)

    all_attempts: list[pd.DataFrame] = []
    provenance: dict[str, dict] = {}
    coverage_rows: list[dict] = []

    for timeframe in PERIOD_MAP:
        frames, errors = fetch_timeframe(symbols, timeframe, DEFAULT_BARS[timeframe])
        provenance[timeframe] = {
            "source": "CoinEx public spot OHLCV",
            "requested_bars_per_symbol": DEFAULT_BARS[timeframe],
            "successful_symbols": sorted(frames), "failures": errors,
            "rows": {s: int(len(f)) for s, f in frames.items()},
        }
        specs = [s for s in STRATEGY_REGISTRY_V20 if s.timeframe == timeframe]
        for spec in specs:
            parts: list[pd.DataFrame] = []
            for symbol, frame in frames.items():
                needs_peer = "CORRELATION" in spec.name or spec.family == "correlation_divergence"
                peer = peer_for(symbol, frames) if needs_peer else None
                labeled = build_labeled_candidate_attempts(spec, frame, symbol, tournament, policy, peer=peer)
                if not labeled.empty:
                    parts.append(labeled)
            candidate = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if not candidate.empty:
                candidate = finalize_candidate_risk(candidate, spec, policy)
                all_attempts.append(candidate)
            coverage_rows.append({
                "strategy": spec.name, "timeframe": timeframe,
                "attempts": int(len(candidate)),
                "symbols_with_data": int(candidate["symbol"].nunique()) if not candidate.empty else 0,
                "profitable_rate": float(candidate["label_profitable_net"].mean()) if not candidate.empty else None,
                "mean_r": float(candidate["label_r_multiple"].mean()) if not candidate.empty else None,
            })
            print(json.dumps(coverage_rows[-1], default=str))

    if not all_attempts:
        raise RuntimeError("No labeled strategy attempts were generated; ML training is fail-closed.")

    dataset = pd.concat(all_attempts, ignore_index=True)
    dataset["signal_time"] = pd.to_datetime(dataset["signal_time"], utc=True)
    dataset["entry_time"] = pd.to_datetime(dataset["entry_time"], utc=True)
    dataset["exit_time"] = pd.to_datetime(dataset["exit_time"], utc=True)
    dataset = dataset.sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)

    # Hard integrity checks before any learner sees the data.
    required_labels = ["label_profitable_net", "label_target_hit", "label_ge_1r", "label_ge_2r", "label_outcome_3class", "label_r_multiple"]
    if dataset[required_labels].isna().any().any():
        raise RuntimeError("Label integrity failure: missing target values")
    if dataset["segment"].nunique() < 3:
        raise RuntimeError("Split integrity failure: development/validation/test are required")

    dataset_path = output / "labeled_ml_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "strategy_label_coverage.csv", index=False)
    (output / "data_provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    manifest = write_dataset_manifest(dataset_path, dataset, output)

    classifier_board, val_pred, test_pred = train_classifier_zoo(dataset, output, cfg)
    classifier_board.to_csv(output / "classifier_leaderboard.csv", index=False)
    if not val_pred.empty:
        val_pred.to_csv(output / "validation_predictions.csv", index=False)
    if not test_pred.empty:
        test_pred.to_csv(output / "test_predictions.csv", index=False)

    regressor_board = train_regressor_zoo(dataset, output, cfg)
    regressor_board.to_csv(output / "regressor_leaderboard.csv", index=False)
    champion = choose_classifier_champion(classifier_board, cfg)
    (output / "champion.json").write_text(json.dumps(champion, indent=2, default=str), encoding="utf-8")

    decision = {
        "version": "v0.21",
        "research_status": "LABELED_STRATEGY_EVENT_META_ML",
        "dataset_rows": int(len(dataset)),
        "strategy_candidates": int(dataset["strategy"].nunique()),
        "timeframes": sorted(dataset["timeframe"].unique().tolist()),
        "symbols": sorted(dataset["symbol"].unique().tolist()),
        "classifier_models_attempted": int(len(classifier_board)),
        "classifier_models_succeeded": int((classifier_board["status"] == "ok").sum()) if not classifier_board.empty else 0,
        "regressor_models_attempted": int(len(regressor_board)),
        "regressor_models_succeeded": int((regressor_board["status"] == "ok").sum()) if not regressor_board.empty else 0,
        "champion": champion,
        "dataset_sha256": manifest["dataset_sha256"],
        "selection_contract": "fit=development; threshold/champion=validation only; test inspected only after champion freeze",
        "label_contract": "Future trade outcome is target only; every f_* input is a signal-time causal snapshot",
        "execution_contract": "Research/PAPER only. ML is an abstention/meta-label layer on top of strategy signals; it does not authorize live orders.",
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
