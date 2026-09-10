from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ict_localization_v22c import (
    LOCALIZATION_LABELS,
    LocalizationConfig,
    build_localization_dataset,
    chronological_split,
    fit_localizer,
    localization_metrics,
    make_localizer,
    predict_localizer,
    save_localization_manifest,
)

PERIOD_MAP = {"15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)


def fetch_frame(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    period = PERIOD_MAP[timeframe]
    frame = fetch_coinex_klines(symbol=symbol, period=period, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    return _drop_incomplete(frame, PERIOD_MS[period])


def _parts_for_split(parts, split_name: str):
    which = {"development": 0, "validation": 1, "test": 2}[split_name]
    xs, ys, times = [], [], []
    for x, y, t in parts:
        s = chronological_split(len(x))[which]
        xs.append(x[s]); ys.append(y[s]); times.extend(t[s].tolist())
    return np.concatenate(xs), np.concatenate(ys), pd.DatetimeIndex(times)


def _decision(val_metrics: dict, test_metrics: dict) -> dict:
    # Pre-registered conservative representation gate. This authorizes only a
    # frozen research embedding candidate; it never authorizes PAPER or LIVE.
    enough_labels = min(val_metrics["supported_labels"], test_metrics["supported_labels"]) >= 8
    val_dice_ok = np.isfinite(val_metrics["supported_macro_dice"]) and val_metrics["supported_macro_dice"] >= 0.15
    test_dice_ok = np.isfinite(test_metrics["supported_macro_dice"]) and test_metrics["supported_macro_dice"] >= 0.10
    stability_ok = (
        np.isfinite(val_metrics["supported_macro_dice"])
        and np.isfinite(test_metrics["supported_macro_dice"])
        and test_metrics["supported_macro_dice"] >= 0.60 * val_metrics["supported_macro_dice"]
    )
    passed = bool(enough_labels and val_dice_ok and test_dice_ok and stability_ok)
    return {
        "decision": "LOCALIZATION_REPRESENTATION_CANDIDATE" if passed else "NO_LOCALIZATION_ENCODER_PROMOTED",
        "passed": passed,
        "gates": {
            "min_supported_labels_each_split": 8,
            "min_validation_macro_dice": 0.15,
            "min_test_macro_dice": 0.10,
            "min_test_to_validation_dice_ratio": 0.60,
        },
        "gate_results": {
            "enough_labels": bool(enough_labels),
            "validation_dice": bool(val_dice_ok),
            "test_dice": bool(test_dice_ok),
            "stability": bool(stability_ok),
        },
        "downstream_use": "frozen localization probabilities/embedding candidate for v0.22d multimodal ablation only",
        "vision_to_rl_state_connected": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.22c causal raw-candle ICT spatial localization")
    ap.add_argument("--output-dir", default="artifacts/v22c-ict-localization")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframe", choices=sorted(PERIOD_MAP), default="4h")
    ap.add_argument("--bars", type=int, default=3600)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--max-samples-per-symbol", type=int, default=480)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = LocalizationConfig(
        lookback=args.lookback,
        height=args.height,
        width=args.width,
        min_history=max(160, args.lookback + 60),
        epochs=args.epochs,
        patience=2,
    )
    save_localization_manifest(out / "localization_protocol.json", cfg)

    parts = []
    provenance = {}
    prevalence_rows = []
    for symbol in symbols:
        try:
            frame = fetch_frame(symbol, args.timeframe, args.bars)
            x, y, times, metas = build_localization_dataset(
                frame, cfg, stride=1, max_samples=args.max_samples_per_symbol
            )
            if len(x) < 180:
                provenance[symbol] = {"status": "insufficient", "bars": int(len(frame)), "samples": int(len(x))}
                continue
            parts.append((x, y, times))
            provenance[symbol] = {
                "status": "ok",
                "source": "CoinEx public spot OHLCV",
                "timeframe": args.timeframe,
                "bars": int(len(frame)),
                "samples": int(len(x)),
                "first_signal_time": times[0].isoformat(),
                "last_signal_time": times[-1].isoformat(),
                "engineered_ict_channels_in_input": False,
                "causal": True,
            }
            row = {"symbol": symbol, "samples": int(len(x))}
            for j, label in enumerate(LOCALIZATION_LABELS):
                row[label] = float(y[:, j].mean())
            prevalence_rows.append(row)
        except Exception as exc:
            provenance[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not parts:
        raise RuntimeError("No usable series for v0.22c; fail-closed")

    x_dev, y_dev, t_dev = _parts_for_split(parts, "development")
    x_val, y_val, t_val = _parts_for_split(parts, "validation")
    x_test, y_test, t_test = _parts_for_split(parts, "test")

    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pd.DataFrame(prevalence_rows).to_csv(out / "mask_pixel_prevalence.csv", index=False)
    (out / "split_manifest.json").write_text(json.dumps({
        "development": {"n": int(len(x_dev)), "first": t_dev.min().isoformat(), "last": t_dev.max().isoformat()},
        "validation": {"n": int(len(x_val)), "first": t_val.min().isoformat(), "last": t_val.max().isoformat()},
        "test": {"n": int(len(x_test)), "first": t_test.min().isoformat(), "last": t_test.max().isoformat()},
        "note": "Each symbol is chronologically split before concatenation. Test is untouched until the localizer and early stopping are frozen.",
    }, indent=2), encoding="utf-8")
    np.savez_compressed(out / "dataset_audit_sample.npz", image=x_dev[:8].astype(np.float16), mask=y_dev[:8].astype(np.int8))

    model = make_localizer("small_unet", in_channels=x_dev.shape[1])
    model, history, pos_weight = fit_localizer(model, x_dev, y_dev, x_val, y_val, cfg)
    history.to_csv(out / "training_history.csv", index=False)
    val_prob = predict_localizer(model, x_val)
    test_prob = predict_localizer(model, x_test)
    val_metrics = localization_metrics(y_val, val_prob, cfg)
    test_metrics = localization_metrics(y_test, test_prob, cfg)
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "development_positive_weights": {name: float(pos_weight[j]) for j, name in enumerate(LOCALIZATION_LABELS)},
    }
    (out / "localization_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import torch
    torch.save({
        "version": "v0.22c",
        "model": "small_unet",
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "labels": list(LOCALIZATION_LABELS),
        "input_channels": ["bull_body", "bear_body", "wick", "volume"],
        "engineered_ict_channels_in_input": False,
    }, out / "small_unet_localizer.pt")

    # Compact probability summaries are retained for the next, separately gated,
    # multimodal experiment. No trading policy consumes them here.
    val_summary = np.concatenate([val_prob.mean(axis=(2, 3)), val_prob.max(axis=(2, 3))], axis=1)
    test_summary = np.concatenate([test_prob.mean(axis=(2, 3)), test_prob.max(axis=(2, 3))], axis=1)
    np.savez_compressed(out / "localizer_probability_summaries.npz",
                        validation=val_summary.astype(np.float16), test=test_summary.astype(np.float16))

    decision = _decision(val_metrics, test_metrics)
    decision.update({
        "version": "v0.22c",
        "timeframe": args.timeframe,
        "symbols": symbols,
        "development_samples": int(len(x_dev)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "validation_supported_macro_dice": val_metrics["supported_macro_dice"],
        "validation_supported_macro_iou": val_metrics["supported_macro_iou"],
        "test_supported_macro_dice": test_metrics["supported_macro_dice"],
        "test_supported_macro_iou": test_metrics["supported_macro_iou"],
        "scientific_interpretation": "Tests spatial recovery of formalized ICT/SMC proxies from raw candle/volume geometry. It does not validate market-maker intent, Wyckoff/ICT causality, or trading alpha.",
    })
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
