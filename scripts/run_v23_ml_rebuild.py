from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ml_meta_v21 import build_labeled_candidate_attempts, finalize_candidate_risk
from research_bot.ml_rebuild_v23 import (
    MLRebuildConfig,
    assert_unique_columns,
    audit_labeled_dataset,
    choose_classifier_champion,
    train_classifier_tournament,
    train_regressor_tournament,
    write_manifest,
)
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import STRATEGY_REGISTRY_V20, RiskPsychologyPolicy


PERIOD_MAP = {"1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_BARS = {"1h": 12000, "4h": 8000, "1d": 3000}
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "LINK/USDT", "AVAX/USDT", "DOGE/USDT",
]
DEFAULT_CLASSIFIERS = [
    "dummy_prior", "logistic", "random_forest", "extra_trees",
    "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
]
DEFAULT_REGRESSORS = [
    "dummy_mean", "ridge", "random_forest_regressor",
    "extra_trees_regressor", "hist_gradient_boosting_regressor",
    "xgboost_regressor", "lightgbm_regressor", "catboost_regressor",
]


def drop_incomplete(frame: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= now].reset_index(drop=True)


def fetch_frames(symbols: list[str], timeframe: str, bars: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    period = PERIOD_MAP[timeframe]
    end_ms = utc_now_ms()
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = fetch_coinex_klines(
                symbol=symbol,
                period=period,
                market_type="spot",
                end_ms=end_ms,
                bars=bars,
            )
            frame = drop_incomplete(frame, PERIOD_MS[period])
            assert_unique_columns(frame)
            if len(frame) < min(500, bars // 3):
                errors[symbol] = f"insufficient_rows:{len(frame)}"
                continue
            frames[symbol] = frame
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}:{exc}"
    return frames, errors


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((frame for name, frame in frames.items() if name != symbol), None)


def build_dataset(
    frames: dict[str, pd.DataFrame],
    timeframe: str,
    policy: RiskPsychologyPolicy,
    tournament: TournamentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [spec for spec in STRATEGY_REGISTRY_V20 if spec.timeframe == timeframe]
    if not specs:
        raise RuntimeError(f"no frozen v0.20 strategies for timeframe={timeframe}")

    all_attempts: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for spec in specs:
        parts: list[pd.DataFrame] = []
        for symbol, frame in frames.items():
            needs_peer = "CORRELATION" in spec.name or spec.family == "correlation_divergence"
            peer = peer_for(symbol, frames) if needs_peer else None
            labeled = build_labeled_candidate_attempts(
                spec, frame, symbol, tournament, policy, peer=peer
            )
            if labeled.empty:
                continue
            # A duplicate feature name is a data-contract error, not something to
            # silently drop or rename after seeing outcomes.
            assert_unique_columns(labeled)
            parts.append(labeled)

        candidate = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not candidate.empty:
            assert_unique_columns(candidate)
            candidate = finalize_candidate_risk(candidate, spec, policy)
            assert_unique_columns(candidate)
            all_attempts.append(candidate)
        coverage.append({
            "strategy": spec.name,
            "family": spec.family,
            "timeframe": timeframe,
            "attempts": int(len(candidate)),
            "symbols": int(candidate["symbol"].nunique()) if not candidate.empty else 0,
            "profitable_rate": float(candidate["label_profitable_net"].mean()) if not candidate.empty else None,
            "mean_post_cost_r": float(candidate["label_r_multiple"].mean()) if not candidate.empty else None,
        })
        print(json.dumps({"coverage": coverage[-1]}, default=str))

    if not all_attempts:
        raise RuntimeError("no labeled strategy attempts generated; fail closed")
    dataset = pd.concat(all_attempts, ignore_index=True)
    assert_unique_columns(dataset)
    dataset["signal_time"] = pd.to_datetime(dataset["signal_time"], utc=True, errors="raise")
    for col in ("entry_time", "exit_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)
    audit_labeled_dataset(dataset)
    return dataset, pd.DataFrame(coverage)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.23 clean leakage-safe ML rebuild")
    ap.add_argument("--output-dir", default="artifacts/v23-ml-rebuild")
    ap.add_argument("--timeframe", choices=sorted(PERIOD_MAP), default="4h")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--bars", type=int, default=None)
    ap.add_argument("--classifiers", default=",".join(DEFAULT_CLASSIFIERS))
    ap.add_argument("--regressors", default=",".join(DEFAULT_REGRESSORS))
    ap.add_argument("--max-fit-rows", type=int, default=30000)
    ap.add_argument("--max-fit-rows-slow", type=int, default=8000)
    ap.add_argument("--min-selected-validation", type=int, default=100)
    ap.add_argument("--min-selected-test", type=int, default=60)
    ap.add_argument("--bootstrap-resamples", type=int, default=300)
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timeframe = args.timeframe
    bars = args.bars or DEFAULT_BARS[timeframe]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    classifiers = [m.strip() for m in args.classifiers.split(",") if m.strip()]
    regressors = [m.strip() for m in args.regressors.split(",") if m.strip()]

    policy = RiskPsychologyPolicy()
    tournament = TournamentConfig(
        risk_per_trade=policy.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=200,
    )
    cfg = MLRebuildConfig(
        max_fit_rows=args.max_fit_rows,
        max_fit_rows_slow=args.max_fit_rows_slow,
        min_selected_validation=args.min_selected_validation,
        min_selected_test=args.min_selected_test,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    frames, fetch_errors = fetch_frames(symbols, timeframe, bars)
    provenance = {
        "version": "v0.23-clean-rebuild",
        "source": "CoinEx public spot OHLCV",
        "timeframe": timeframe,
        "requested_symbols": symbols,
        "requested_bars_per_symbol": bars,
        "successful_symbols": sorted(frames),
        "fetch_errors": fetch_errors,
        "rows": {symbol: int(len(frame)) for symbol, frame in frames.items()},
        "data_contract": "closed candles only; strategy signal at closed bar; labels inherited from frozen v0.20 cost-aware bracket simulator",
    }
    (output / "data_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    if len(frames) < 3:
        raise RuntimeError(f"fewer than three usable symbols: {sorted(frames)}")

    dataset, coverage = build_dataset(frames, timeframe, policy, tournament)
    dataset_path = output / "labeled_strategy_events.csv"
    dataset.to_csv(dataset_path, index=False)
    coverage.to_csv(output / "strategy_coverage.csv", index=False)
    manifest = write_manifest(dataset_path, dataset, output, cfg)

    classifier_board, val_pred, test_pred, model_meta = train_classifier_tournament(
        dataset,
        output,
        cfg,
        include_unsupervised=True,
        model_names=classifiers,
    )
    classifier_board.to_csv(output / "classifier_leaderboard.csv", index=False)
    if not val_pred.empty:
        val_pred.to_csv(output / "validation_predictions.csv", index=False)
    if not test_pred.empty:
        test_pred.to_csv(output / "test_predictions.csv", index=False)
    (output / "classifier_feature_contract.json").write_text(
        json.dumps(model_meta, indent=2, default=str), encoding="utf-8"
    )

    regressor_board = train_regressor_tournament(
        dataset,
        output,
        cfg,
        model_names=regressors,
    )
    regressor_board.to_csv(output / "regressor_leaderboard.csv", index=False)

    champion = choose_classifier_champion(classifier_board, cfg)
    (output / "champion.json").write_text(
        json.dumps(champion, indent=2, default=str), encoding="utf-8"
    )
    decision = {
        "version": "v0.23-clean-rebuild",
        "timeframe": timeframe,
        "dataset_rows": int(len(dataset)),
        "strategies": int(dataset["strategy"].nunique()),
        "symbols": sorted(dataset["symbol"].unique().tolist()),
        "classifiers_requested": classifiers,
        "classifiers_succeeded": int((classifier_board["status"] == "ok").sum()) if not classifier_board.empty else 0,
        "regressors_requested": regressors,
        "regressors_succeeded": int((regressor_board["status"] == "ok").sum()) if not regressor_board.empty else 0,
        "champion": champion,
        "dataset_sha256": manifest["dataset_sha256"],
        "selection_contract": "development fit; validation threshold/champion; test read once after freeze",
        "risk_contract": "post-cost R; 0.25% event risk; 1.0% concurrent signal-time risk cap",
        "vision_to_rl_state_connected": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
