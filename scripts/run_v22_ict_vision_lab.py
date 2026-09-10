from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.vision_ict_v22 import (
    CHANNEL_NAMES,
    WEAK_LABEL_NAMES,
    VisionConfig,
    build_vision_dataset,
    chronological_split,
    fit_multitask_model,
    make_vision_model,
    predict_multitask,
)

PERIOD_MAP = {"15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT"]


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


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def weak_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    per_label = {}
    f1s = []
    for j, name in enumerate(WEAK_LABEL_NAMES):
        score = float(f1_score(y[:, j], pred[:, j], zero_division=0))
        per_label[name] = score
        f1s.append(score)
    return {"weak_macro_f1": float(np.mean(f1s)), "weak_per_label_f1": per_label}


def outcome_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "outcome_auc": _safe_auc(y, p),
        "outcome_balanced_accuracy": float(balanced_accuracy_score(y.astype(int), pred)),
    }


def combine_series(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]], split_name: str):
    idx = {"development": 0, "validation": 1, "test": 2}[split_name]
    xs, ws, ys, ts = [], [], [], []
    for x, w, y, t in parts:
        slices = chronological_split(len(x))
        s = slices[idx]
        xs.append(x[s]); ws.append(w[s]); ys.append(y[s]); ts.extend(t[s].tolist())
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(ys), pd.DatetimeIndex(ts)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.22 deterministic candlestick + ICT weak-label vision lab")
    ap.add_argument("--output-dir", default="artifacts/v22-ict-vision")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframe", choices=sorted(PERIOD_MAP), default="4h")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--max-samples-per-symbol", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--models", default="small_cnn,small_vit,resnet18,efficientnet_b0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = VisionConfig(lookback=args.lookback, height=args.height, width=args.width, min_history=max(120, args.lookback + 40), epochs=args.epochs)

    series = []
    provenance = {}
    label_audit = []
    sample_renderer_hashes = []
    for symbol in symbols:
        try:
            frame = fetch_frame(symbol, args.timeframe, args.bars)
            x, weak, y, times, metas = build_vision_dataset(frame, cfg, stride=1, max_samples=args.max_samples_per_symbol)
            if len(x) < 100:
                provenance[symbol] = {"status": "insufficient", "bars": int(len(frame)), "samples": int(len(x))}
                continue
            series.append((x, weak, y, times))
            provenance[symbol] = {
                "status": "ok", "source": "CoinEx public spot OHLCV", "timeframe": args.timeframe,
                "bars": int(len(frame)), "samples": int(len(x)), "first": times[0].isoformat(), "last": times[-1].isoformat(),
            }
            sample_renderer_hashes.extend(m["render_sha256"] for m in metas[:3])
            row = {"symbol": symbol, "samples": int(len(x)), "outcome_positive_rate": float(y.mean())}
            for j, name in enumerate(WEAK_LABEL_NAMES):
                row[f"label_rate_{name}"] = float(weak[:, j].mean())
            label_audit.append(row)
        except Exception as exc:
            provenance[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not series:
        raise RuntimeError("No usable public-data vision series; fail-closed")

    x_dev, w_dev, y_dev, t_dev = combine_series(series, "development")
    x_val, w_val, y_val, t_val = combine_series(series, "validation")
    x_test, w_test, y_test, t_test = combine_series(series, "test")
    np.savez_compressed(out / "vision_dataset_audit_sample.npz", x=x_dev[:32].astype(np.float16), weak=w_dev[:32], outcome=y_dev[:32])
    pd.DataFrame(label_audit).to_csv(out / "weak_label_prevalence.csv", index=False)
    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    manifest = {
        "version": "v0.22", "timeframe": args.timeframe, "symbols_requested": symbols,
        "config": cfg.__dict__, "channels": list(CHANNEL_NAMES), "weak_structural_labels": list(WEAK_LABEL_NAMES),
        "split_sizes": {"development": len(x_dev), "validation": len(x_val), "test": len(x_test)},
        "renderer_hash_chain_sha256": sha256("".join(sample_renderer_hashes).encode()).hexdigest(),
        "causal_contract": "image at t contains bars <=t only; no axes/text/timestamp glyphs; weak ICT labels are causal rule-derived targets",
        "test_contract": "model and early stopping use development/validation; test metrics are computed after training without refit",
    }
    (out / "vision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    test_predictions = []
    for kind in models:
        record = {"model": kind, "status": "failed"}
        try:
            model = make_vision_model(kind, config=cfg)
            model, history = fit_multitask_model(model, x_dev, w_dev, y_dev, x_val, w_val, y_val, cfg)
            history.to_csv(out / f"history_{kind}.csv", index=False)
            wpv, opv, _ = predict_multitask(model, x_val)
            wpt, opt, embt = predict_multitask(model, x_test)
            vm = {**weak_metrics(w_val, wpv), **outcome_metrics(y_val, opv)}
            tm = {**weak_metrics(w_test, wpt), **outcome_metrics(y_test, opt)}
            objective = float(vm["weak_macro_f1"] + 0.25 * (vm["outcome_auc"] if np.isfinite(vm["outcome_auc"]) else 0.5))
            record = {
                "model": kind, "status": "ok", "validation_objective": objective,
                "validation_weak_macro_f1": vm["weak_macro_f1"], "validation_outcome_auc": vm["outcome_auc"],
                "validation_outcome_balanced_accuracy": vm["outcome_balanced_accuracy"],
                "test_weak_macro_f1": tm["weak_macro_f1"], "test_outcome_auc": tm["outcome_auc"],
                "test_outcome_balanced_accuracy": tm["outcome_balanced_accuracy"],
                "test_samples": int(len(y_test)),
            }
            for name, score in vm["weak_per_label_f1"].items(): record[f"validation_f1_{name}"] = score
            for name, score in tm["weak_per_label_f1"].items(): record[f"test_f1_{name}"] = score
            torch_path = out / f"vision_{kind}.pt"
            import torch
            torch.save({"model_kind": kind, "state_dict": model.state_dict(), "config": cfg.__dict__, "channels": CHANNEL_NAMES, "weak_labels": WEAK_LABEL_NAMES}, torch_path)
            pred = pd.DataFrame({"signal_time": t_test, "outcome_true": y_test, "outcome_probability": opt, "model": kind})
            test_predictions.append(pred)
            np.savez_compressed(out / f"test_embeddings_{kind}.npz", embedding=embt.astype(np.float16), outcome=y_test.astype(np.int8))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        print(json.dumps(record, default=str))

    board = pd.DataFrame(results)
    board.to_csv(out / "vision_leaderboard.csv", index=False)
    if test_predictions:
        pd.concat(test_predictions, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    ok = board[board["status"] == "ok"].copy() if len(board) else board
    if ok.empty:
        decision = {"decision": "NO_VISION_ENCODER_CANDIDATE", "champion": None, "live_execution_authorized": False}
    else:
        champ = ok.sort_values("validation_objective", ascending=False).iloc[0]
        decision = {
            "decision": "VISION_REPRESENTATION_CANDIDATE",
            "champion": str(champ["model"]),
            "selection_basis": "validation weak-structure macro-F1 plus 0.25*outcome AUC; test never used to rank models",
            "validation_objective": float(champ["validation_objective"]),
            "test_weak_macro_f1": float(champ["test_weak_macro_f1"]),
            "test_outcome_auc": float(champ["test_outcome_auc"]),
            "downstream_use": "frozen embedding/probability candidate for multimodal fusion; not a trading strategy",
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
