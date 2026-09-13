from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from research_bot.deep_models_v52 import DeepModelConfigV52, build_lstm_v52, build_mamba_v52, build_patchtst_v52
from research_bot.sequence_benchmark_v52 import CostConfigV52, SplitConfigV52, chronological_split_v52, evaluate_scores_v52
from research_bot.training_v52 import TrainingConfigV52, fit_torch_regressor_v52, make_sequences_v52, predict_torch_v52


def _numeric_features(frame: pd.DataFrame, *, target: str, time_col: str) -> list[str]:
    cols = [c for c in frame.columns if c not in {target, time_col} and pd.api.types.is_numeric_dtype(frame[c])]
    if not cols:
        raise ValueError("no numeric feature columns")
    return cols


def _build(name: str, cfg: DeepModelConfigV52):
    if name == "lstm":
        return build_lstm_v52(cfg)
    if name == "patchtst":
        return build_patchtst_v52(cfg)
    if name == "mamba":
        return build_mamba_v52(cfg)
    raise ValueError(f"unknown model: {name}")


def run(path: Path, *, model_name: str, target: str, time_col: str, seeds: list[int], sequence_length: int) -> dict:
    frame = pd.read_csv(path)
    if target not in frame:
        raise ValueError(f"missing target: {target}")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    if frame[target].isna().any() or not np.isfinite(frame[target]).all():
        raise ValueError("target must be finite")
    parts = chronological_split_v52(frame, time_col=time_col, config=SplitConfigV52())
    features = _numeric_features(frame, target=target, time_col=time_col)
    scaler = StandardScaler().fit(parts["train"][features])

    arrays = {}
    for split, block in parts.items():
        x = scaler.transform(block[features]).astype(np.float32)
        y = block[target].to_numpy(dtype=np.float32)
        arrays[split] = make_sequences_v52(x, y, sequence_length=sequence_length)

    model_cfg = DeepModelConfigV52(input_dim=len(features))
    train_cfg = TrainingConfigV52(sequence_length=sequence_length)
    runs = []
    for seed in seeds:
        model = _build(model_name, model_cfg)
        model, fit_meta = fit_torch_regressor_v52(
            model,
            arrays["train"][0], arrays["train"][1],
            arrays["validation"][0], arrays["validation"][1],
            seed=seed,
            config=train_cfg,
        )
        scores = predict_torch_v52(model, arrays["test"][0])
        metrics = evaluate_scores_v52(scores, arrays["test"][1], costs=CostConfigV52())
        runs.append({"seed": seed, "fit": fit_meta, "metrics": metrics})

    return {
        "status": "RESEARCH_ONLY",
        "model": model_name,
        "seeds": seeds,
        "features": features,
        "sequence_length": sequence_length,
        "split_rows": {k: len(v) for k, v in parts.items()},
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
        "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--model", choices=["lstm", "patchtst", "mamba"], required=True)
    p.add_argument("--target", default="future_return")
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--seeds", default="11,23,37,41,53")
    p.add_argument("--sequence-length", type=int, default=48)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be unique and non-empty")
    result = run(args.csv, model_name=args.model, target=args.target, time_col=args.time_col, seeds=seeds, sequence_length=args.sequence_length)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
