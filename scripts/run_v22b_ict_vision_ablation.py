from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.vision_ablation_v22b import (
    CHANNEL_MODES,
    VisionAblationConfig,
    decide_ablation,
    fit_balanced_multitask_model,
    model_record,
    predict_multitask,
    save_ablation_protocol,
    select_channels,
)
from research_bot.vision_ict_v22 import (
    WEAK_LABEL_NAMES,
    VisionConfig,
    build_vision_dataset,
    chronological_split,
    make_vision_model,
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
    df = fetch_coinex_klines(symbol=symbol, period=period, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    return _drop_incomplete(df, PERIOD_MS[period])


def _split_parts(parts, name: str):
    which = {"development": 0, "validation": 1, "test": 2}[name]
    xs, ws, ys, ts = [], [], [], []
    for x, w, y, t in parts:
        s = chronological_split(len(x))[which]
        xs.append(x[s]); ws.append(w[s]); ys.append(y[s]); ts.extend(t[s].tolist())
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(ys), pd.DatetimeIndex(ts)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.22b balanced raw-vs-structured candlestick ICT vision ablation")
    ap.add_argument("--output-dir", default="artifacts/v22b-ict-vision-ablation")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframe", choices=sorted(PERIOD_MAP), default="4h")
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--max-samples-per-symbol", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--architectures", default="small_cnn,small_vit")
    ap.add_argument("--channel-modes", default="raw_candles,raw_plus_ichimoku,structure_augmented")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    architectures = [s.strip() for s in args.architectures.split(",") if s.strip()]
    channel_modes = [s.strip() for s in args.channel_modes.split(",") if s.strip()]
    for mode in channel_modes:
        if mode not in CHANNEL_MODES:
            raise ValueError(f"unknown channel mode {mode}")

    vcfg = VisionConfig(
        lookback=args.lookback,
        height=args.height,
        width=args.width,
        min_history=max(160, args.lookback + 60),
        epochs=args.epochs,
        patience=2,
    )
    acfg = VisionAblationConfig()
    save_ablation_protocol(out / "ablation_protocol.json", acfg)

    provenance = {}
    parts = []
    prevalence = []
    for symbol in symbols:
        try:
            frame = fetch_frame(symbol, args.timeframe, args.bars)
            x, weak, y, times, metas = build_vision_dataset(
                frame,
                vcfg,
                stride=1,
                max_samples=args.max_samples_per_symbol,
            )
            if len(x) < 200:
                provenance[symbol] = {"status": "insufficient", "bars": int(len(frame)), "samples": int(len(x))}
                continue
            parts.append((x, weak, y, times))
            provenance[symbol] = {
                "status": "ok",
                "source": "CoinEx public spot OHLCV",
                "timeframe": args.timeframe,
                "bars": int(len(frame)),
                "samples": int(len(x)),
                "first_signal_time": times[0].isoformat(),
                "last_signal_time": times[-1].isoformat(),
                "first_render_sha256": metas[0]["render_sha256"] if metas else None,
                "last_render_sha256": metas[-1]["render_sha256"] if metas else None,
            }
            row = {"symbol": symbol, "samples": int(len(x)), "future_up_rate": float(y.mean())}
            for j, label in enumerate(WEAK_LABEL_NAMES):
                row[label] = float(weak[:, j].mean())
            prevalence.append(row)
        except Exception as exc:
            provenance[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not parts:
        raise RuntimeError("No usable series for v0.22b; fail-closed")

    x_dev, w_dev, y_dev, t_dev = _split_parts(parts, "development")
    x_val, w_val, y_val, t_val = _split_parts(parts, "validation")
    x_test, w_test, y_test, t_test = _split_parts(parts, "test")
    pd.DataFrame(prevalence).to_csv(out / "weak_label_prevalence.csv", index=False)
    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps({
        "development": {"n": len(x_dev), "first": t_dev.min().isoformat(), "last": t_dev.max().isoformat()},
        "validation": {"n": len(x_val), "first": t_val.min().isoformat(), "last": t_val.max().isoformat()},
        "test": {"n": len(x_test), "first": t_test.min().isoformat(), "last": t_test.max().isoformat()},
        "note": "Each symbol is chronologically split before cross-symbol concatenation. Test is never used for early stopping or architecture/channel selection.",
    }, indent=2), encoding="utf-8")

    # Small immutable sample for rendering/data-shape audit, not training reuse.
    np.savez_compressed(
        out / "dataset_audit_sample.npz",
        image=x_dev[:16].astype(np.float16),
        weak=w_dev[:16].astype(np.int8),
        outcome=y_dev[:16].astype(np.int8),
    )

    records = []
    predictions = []
    detailed_metrics = {}
    for mode in channel_modes:
        xd, xv, xt = select_channels(x_dev, mode), select_channels(x_val, mode), select_channels(x_test, mode)
        for architecture in architectures:
            key = f"{architecture}:{mode}"
            rec = {"architecture": architecture, "channel_mode": mode, "status": "failed"}
            try:
                model = make_vision_model(architecture, in_channels=xd.shape[1], config=vcfg)
                model, history, pos_weight = fit_balanced_multitask_model(
                    model, xd, w_dev, y_dev, xv, w_val, y_val, vcfg, acfg
                )
                history.to_csv(out / f"history_{architecture}_{mode}.csv", index=False)
                wpv, opv, _ = predict_multitask(model, xv)
                wpt, opt, embt = predict_multitask(model, xt)
                rec0, val_detail, test_detail = model_record(
                    architecture=architecture,
                    channel_mode=mode,
                    weak_val=w_val,
                    weak_prob_val=wpv,
                    y_val=y_val,
                    outcome_prob_val=opv,
                    weak_test=w_test,
                    weak_prob_test=wpt,
                    y_test=y_test,
                    outcome_prob_test=opt,
                    cfg=acfg,
                )
                rec = {"status": "ok", **rec0}
                detailed_metrics[key] = {
                    "development_weak_positive_weights": {name: float(pos_weight[j]) for j, name in enumerate(WEAK_LABEL_NAMES)},
                    "validation": val_detail,
                    "test": test_detail,
                }
                import torch
                torch.save({
                    "architecture": architecture,
                    "channel_mode": mode,
                    "state_dict": model.state_dict(),
                    "vision_config": vcfg.__dict__,
                    "ablation_config": acfg.__dict__,
                    "input_channels": list(CHANNEL_MODES[mode]),
                    "weak_labels": list(WEAK_LABEL_NAMES),
                }, out / f"model_{architecture}_{mode}.pt")
                np.savez_compressed(out / f"test_embedding_{architecture}_{mode}.npz", embedding=embt.astype(np.float16))
                pred = pd.DataFrame({
                    "signal_time": t_test,
                    "architecture": architecture,
                    "channel_mode": mode,
                    "future_up_true": y_test,
                    "future_up_probability": opt,
                })
                predictions.append(pred)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            print(json.dumps(rec, default=str))

    board = pd.DataFrame(records)
    board.to_csv(out / "ablation_leaderboard.csv", index=False)
    (out / "per_label_metrics.json").write_text(json.dumps(detailed_metrics, indent=2), encoding="utf-8")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    decision = decide_ablation(board, acfg)
    decision.update({
        "version": "v0.22b",
        "timeframe": args.timeframe,
        "symbols": symbols,
        "development_samples": int(len(x_dev)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "scientific_interpretation": "Raw-candle detection and engineered structure augmentation are separate hypotheses. Passing one does not prove ICT causality or trading alpha.",
    })
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
