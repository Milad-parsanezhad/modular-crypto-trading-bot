from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.deep_temporal_v22 import (
    TemporalConfig, build_sequences, choose_validation_threshold, economic_backtest,
    fit_binary_model, make_model, predict_probability,
)

PERIODS = {"15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
DEFAULT_BARS = {"15m": 14000, "1h": 10000, "4h": 7000, "1d": 2800}
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "DOGE/USDT"]
MODELS = ["lstm", "gru", "cnn_lstm", "tcn", "transformer"]


def fetch(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    period = PERIODS[timeframe]
    x = fetch_coinex_klines(symbol=symbol, period=period, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    closed = x["timestamp"] + pd.Timedelta(milliseconds=PERIOD_MS[period]) <= pd.Timestamp.now(tz="UTC")
    return x[closed].sort_values("timestamp").reset_index(drop=True)


def cap(x, y, r, t, limit: int):
    if len(x) <= limit: return x, y, r, t
    idx = np.linspace(0, len(x) - 1, limit, dtype=int)
    return x[idx], y[idx], r[idx], t[idx]


def assemble(symbols: list[str], timeframe: str, cfg: TemporalConfig, bars: int, split=(0.60, 0.20, 0.20), cap_per_symbol_split=12000):
    data = {"development": [], "validation": [], "test": []}
    provenance = {}
    features = None
    for symbol in symbols:
        try:
            frame = fetch(symbol, timeframe, bars)
            X, y, r, ts, cols = build_sequences(frame, cfg)
            features = cols
            n = len(X); a = int(n * split[0]); b = int(n * (split[0] + split[1]))
            pieces = {"development": (X[:a], y[:a], r[:a], ts[:a]), "validation": (X[a:b], y[a:b], r[a:b], ts[a:b]), "test": (X[b:], y[b:], r[b:], ts[b:])}
            for name, vals in pieces.items():
                z = cap(*vals, cap_per_symbol_split)
                data[name].append((symbol, *z))
            provenance[symbol] = {"bars": int(len(frame)), "sequences": int(n), "first": str(frame.timestamp.min()), "last": str(frame.timestamp.max())}
        except Exception as exc:
            provenance[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    out = {}
    for split_name, parts in data.items():
        if not parts: raise RuntimeError(f"no {split_name} sequences")
        X = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts]); r = np.concatenate([p[3] for p in parts])
        meta = pd.DataFrame({"symbol": np.concatenate([[p[0]] * len(p[1]) for p in parts]), "timestamp": np.concatenate([p[4].astype(str) for p in parts])})
        order = np.argsort(pd.to_datetime(meta["timestamp"], utc=True).astype("int64").to_numpy())
        out[split_name] = (X[order], y[order], r[order], meta.iloc[order].reset_index(drop=True))
    return out, provenance, features


def class_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    m = {"balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    try: m["roc_auc"] = float(roc_auc_score(y, p))
    except Exception: m["roc_auc"] = None
    return m


def main():
    ap = argparse.ArgumentParser(description="v0.22 causal temporal deep-learning benchmark")
    ap.add_argument("--output-dir", default="artifacts/v22-temporal")
    ap.add_argument("--timeframes", default="4h")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="314,2718,1618")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cap-per-symbol-split", type=int, default=6000)
    args = ap.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tfs = [x.strip() for x in args.timeframes.split(",") if x.strip()]
    cfg = TemporalConfig(epochs=args.epochs)
    leaderboard, predictions, provenance_all = [], [], {}

    for tf in tfs:
        splits, provenance, features = assemble(symbols, tf, cfg, DEFAULT_BARS[tf], cap_per_symbol_split=args.cap_per_symbol_split)
        provenance_all[tf] = provenance
        Xd, yd, rd, md = splits["development"]; Xv, yv, rv, mv = splits["validation"]; Xt, yt, rt, mt = splits["test"]
        for kind in models:
            for seed in seeds:
                model = make_model(kind, Xd.shape[-1], cfg)
                model, history = fit_binary_model(model, Xd, yd, Xv, yv, cfg, seed)
                pv = predict_probability(model, Xv); threshold, ev = choose_validation_threshold(pv, rv, min_active=min(200, max(30, len(rv) // 20)), cost_bps_each_way=cfg.fee_bps_each_way + cfg.slippage_bps_each_way)
                pt = predict_probability(model, Xt); et = economic_backtest(pt, rt, threshold, cost_bps_each_way=cfg.fee_bps_each_way + cfg.slippage_bps_each_way)
                cv, ct = class_metrics(yv, pv), class_metrics(yt, pt)
                name = f"{tf}_{kind}_seed{seed}"
                torch.save({"state_dict": model.state_dict(), "model_kind": kind, "input_dim": Xd.shape[-1], "config": cfg.__dict__, "features": features, "threshold": threshold, "seed": seed}, output / f"{name}.pt")
                history.to_csv(output / f"{name}_learning_curve.csv", index=False)
                leaderboard.append({"timeframe": tf, "model": kind, "seed": seed, "threshold": threshold, "development_rows": len(Xd), "validation_rows": len(Xv), "test_rows": len(Xt), **{f"validation_{k}": v for k, v in cv.items()}, **{f"validation_{k}": v for k, v in ev.items()}, **{f"test_{k}": v for k, v in ct.items()}, **{f"test_{k}": v for k, v in et.items()}})
                z = mt.copy(); z["timeframe"] = tf; z["model"] = kind; z["seed"] = seed; z["probability"] = pt; z["future_log_return"] = rt; z["selected_position"] = np.where(pt >= threshold, 1, np.where(pt <= 1 - threshold, -1, 0)); predictions.append(z)
                print(json.dumps(leaderboard[-1], default=str))

    board = pd.DataFrame(leaderboard)
    board.to_csv(output / "per_seed_leaderboard.csv", index=False)
    if predictions: pd.concat(predictions, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    agg = board.groupby(["timeframe", "model"], as_index=False).agg(validation_objective_mean=("validation_validation_objective", "mean"), validation_objective_std=("validation_validation_objective", "std"), test_total_return_mean=("test_total_return", "mean"), test_total_return_std=("test_total_return", "std"), test_max_drawdown_mean=("test_max_drawdown", "mean"), test_profit_factor_mean=("test_profit_factor", "mean"), seeds=("seed", "nunique"))
    agg = agg.sort_values(["validation_objective_mean", "validation_objective_std"], ascending=[False, True])
    agg.to_csv(output / "model_family_leaderboard.csv", index=False)
    champion = agg.iloc[0].to_dict() if not agg.empty else None
    # Test metrics are reported, not used to re-rank the champion.
    decision = {"version": "v0.22", "task": "causal_historical_sequence_learning", "selection": "model family ranked by mean validation objective across seeds", "champion_family": champion, "minimum_final_promotion_seeds": 5, "current_seed_count": len(seeds), "promotion_ready": bool(len(seeds) >= 5), "paper_authorized": False, "live_execution_authorized": False}
    (output / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    provenance_payload = {"source": "CoinEx public spot OHLCV", "timeframes": provenance_all, "features": features, "sequence_contract": "window ends at t; target begins after t; expanding normalization shifted by one bar"}
    raw = json.dumps(provenance_payload, sort_keys=True, default=str).encode(); provenance_payload["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    (output / "provenance.json").write_text(json.dumps(provenance_payload, indent=2, default=str), encoding="utf-8")

if __name__ == "__main__": main()
