from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.multimodal_fusion_v22d import (
    CORE_NUMERIC_FEATURES,
    MultimodalConfig,
    RobustNumericScaler,
    binary_metrics,
    build_multimodal_dataset,
    chronological_split,
    decide_multimodal,
    economic_diagnostic,
    fit_binary_model,
    make_model,
    predict_binary,
    save_protocol,
)
from research_bot.scientific_liquidity_wyckoff_v22d import (
    COURSE_HYPOTHESIS_FEATURES,
    SUPPORTED_COMPONENT_FEATURES,
    feature_evidence_table,
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


def _split_one(ds: dict[str, object], split_name: str) -> dict[str, object]:
    which = {"development": 0, "validation": 1, "test": 2}[split_name]
    s = chronological_split(len(ds["target"]))[which]
    return {k: (v[s] if hasattr(v, "__getitem__") else v) for k, v in ds.items()}


def _concat(parts: list[dict[str, object]], key: str):
    values = [p[key] for p in parts]
    if key == "timestamp":
        return pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in values]))
    return np.concatenate(values)


def _build_split(parts: list[dict[str, object]], split_name: str) -> dict[str, object]:
    split_parts = [_split_one(p, split_name) for p in parts]
    return {k: _concat(split_parts, k) for k in split_parts[0]}


def _matrix(split: dict[str, object], arm: str) -> np.ndarray:
    core = np.asarray(split["core"], dtype=np.float32)
    supported = np.asarray(split["supported"], dtype=np.float32)
    course = np.asarray(split["course"], dtype=np.float32)
    if arm == "numeric_core":
        return core
    if arm in {"numeric_supported", "fusion_supported"}:
        return np.concatenate([core, supported], axis=1)
    if arm == "fusion_supported_plus_course":
        return np.concatenate([core, supported, course], axis=1)
    if arm == "image_raw":
        return np.zeros((len(core), 1), dtype=np.float32)
    raise ValueError(arm)


def _kind(arm: str) -> str:
    if arm == "image_raw":
        return "image"
    if arm.startswith("numeric_"):
        return "numeric"
    return "fusion"


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.22d cost-aware raw-chart + causal numeric multimodal ablation")
    ap.add_argument("--output-dir", default="artifacts/v22d-multimodal-fusion")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframe", choices=sorted(PERIOD_MAP), default="4h")
    ap.add_argument("--bars", type=int, default=3600)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--max-samples-per-symbol", type=int, default=480)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = MultimodalConfig(
        lookback=args.lookback,
        height=args.height,
        width=args.width,
        max_samples=args.max_samples_per_symbol,
        epochs=args.epochs,
    )
    save_protocol(out / "multimodal_protocol.json", cfg)
    feature_evidence_table().to_csv(out / "liquidity_wyckoff_feature_evidence.csv", index=False)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    datasets: list[dict[str, object]] = []
    provenance = {}
    for symbol in symbols:
        try:
            frame = fetch_frame(symbol, args.timeframe, args.bars)
            ds = build_multimodal_dataset(frame, cfg)
            if len(ds["target"]) < 180:
                provenance[symbol] = {"status": "insufficient", "bars": int(len(frame)), "samples": int(len(ds["target"]))}
                continue
            datasets.append(ds)
            provenance[symbol] = {
                "status": "ok",
                "source": "CoinEx public spot OHLCV",
                "timeframe": args.timeframe,
                "bars": int(len(frame)),
                "samples": int(len(ds["target"])),
                "first_signal_time": ds["timestamp"][0].isoformat(),
                "last_signal_time": ds["timestamp"][-1].isoformat(),
                "roundtrip_cost": cfg.roundtrip_cost,
                "target": "open[t+1] to open[t+2] gross return > round-trip cost hurdle",
            }
        except Exception as exc:
            provenance[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not datasets:
        raise RuntimeError("No usable series for v0.22d; fail-closed")

    dev = _build_split(datasets, "development")
    val = _build_split(datasets, "validation")
    test = _build_split(datasets, "test")
    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps({
        "development": {"n": int(len(dev["target"])), "first": dev["timestamp"].min().isoformat(), "last": dev["timestamp"].max().isoformat()},
        "validation": {"n": int(len(val["target"])), "first": val["timestamp"].min().isoformat(), "last": val["timestamp"].max().isoformat()},
        "test": {"n": int(len(test["target"])), "first": test["timestamp"].min().isoformat(), "last": test["timestamp"].max().isoformat()},
        "note": "Each symbol is chronologically split before concatenation. Test is never used for early stopping, arm selection, or scaling. This is a representation diagnostic, not a portfolio backtest.",
    }, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "dataset_audit_sample.npz",
        image=np.asarray(dev["image"][:8], dtype=np.float16),
        core=np.asarray(dev["core"][:8], dtype=np.float16),
        supported=np.asarray(dev["supported"][:8], dtype=np.float16),
        course=np.asarray(dev["course"][:8], dtype=np.float16),
        target=np.asarray(dev["target"][:8], dtype=np.int8),
    )

    arms = [
        "numeric_core",
        "numeric_supported",
        "image_raw",
        "fusion_supported",
        "fusion_supported_plus_course",
    ]
    records: dict[str, dict] = {}
    prediction_frames = []

    for arm in arms:
        ndev = _matrix(dev, arm)
        nval = _matrix(val, arm)
        ntest = _matrix(test, arm)
        if arm != "image_raw":
            scaler = RobustNumericScaler.fit(ndev)
            ndev = scaler.transform(ndev); nval = scaler.transform(nval); ntest = scaler.transform(ntest)
        else:
            scaler = None

        image_dev = np.asarray(dev["image"], dtype=np.float32)
        image_val = np.asarray(val["image"], dtype=np.float32)
        image_test = np.asarray(test["image"], dtype=np.float32)
        ydev = np.asarray(dev["target"], dtype=np.float32)
        yval = np.asarray(val["target"], dtype=np.float32)
        ytest = np.asarray(test["target"], dtype=np.float32)

        model = make_model(_kind(arm), ndev.shape[1])
        model, history = fit_binary_model(model, image_dev, ndev, ydev, image_val, nval, yval, cfg)
        history.to_csv(out / f"history_{arm}.csv", index=False)
        pval, _ = predict_binary(model, image_val, nval)
        ptest, embtest = predict_binary(model, image_test, ntest)
        val_metrics = binary_metrics(yval, pval)
        test_metrics = binary_metrics(ytest, ptest)
        econ = economic_diagnostic(np.asarray(test["gross_return"], dtype=float), ptest, cfg.roundtrip_cost, threshold=0.60)
        records[arm] = {
            "validation": val_metrics,
            "test": test_metrics,
            "test_economic_diagnostic": econ,
            "scientific_role": (
                "course-hypothesis exploratory; ineligible for promotion"
                if arm == "fusion_supported_plus_course"
                else "pre-registered primary/unimodal ablation arm"
            ),
        }
        print(json.dumps({"arm": arm, **records[arm]}, default=str))

        import torch
        torch.save({
            "version": "v0.22d",
            "arm": arm,
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "numeric_dim": int(ndev.shape[1]),
            "scaler_median": None if scaler is None else scaler.median,
            "scaler_scale": None if scaler is None else scaler.scale,
            "core_features": list(CORE_NUMERIC_FEATURES),
            "supported_component_features": list(SUPPORTED_COMPONENT_FEATURES),
            "course_hypothesis_features": list(COURSE_HYPOTHESIS_FEATURES) if arm == "fusion_supported_plus_course" else [],
        }, out / f"model_{arm}.pt")
        np.savez_compressed(out / f"test_embedding_{arm}.npz", embedding=embtest.astype(np.float16))
        prediction_frames.append(pd.DataFrame({
            "signal_time": test["timestamp"],
            "arm": arm,
            "target_true": ytest.astype(int),
            "gross_return_open1_open2": np.asarray(test["gross_return"], dtype=float),
            "probability_net_edge_positive": ptest,
        }))

    (out / "arm_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)

    scientific_gain = {
        "validation_auc_gain_numeric_supported_vs_core": float(records["numeric_supported"]["validation"]["auc"] - records["numeric_core"]["validation"]["auc"]),
        "test_auc_gain_numeric_supported_vs_core": float(records["numeric_supported"]["test"]["auc"] - records["numeric_core"]["test"]["auc"]),
        "interpretation": "incremental diagnostic for the scientifically supported liquidity component family; not a causal or alpha proof",
    }
    decision = decide_multimodal(records, cfg)
    decision.update({
        "version": "v0.22d",
        "timeframe": args.timeframe,
        "symbols": symbols,
        "development_samples": int(len(dev["target"])),
        "validation_samples": int(len(val["target"])),
        "test_samples": int(len(test["target"])),
        "roundtrip_cost": cfg.roundtrip_cost,
        "scientific_liquidity_increment": scientific_gain,
        "localization_required_before_rl": True,
        "multimodal_required_before_rl": True,
        "vision_to_rl_state_connected": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    })
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
