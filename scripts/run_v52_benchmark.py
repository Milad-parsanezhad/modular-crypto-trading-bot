from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research_bot.sequence_benchmark_v52 import CostConfigV52, SplitConfigV52, chronological_split_v52, evaluate_scores_v52


def _feature_columns(frame: pd.DataFrame, target: str, time_col: str) -> list[str]:
    cols = [c for c in frame.columns if c not in {target, time_col} and pd.api.types.is_numeric_dtype(frame[c])]
    if not cols:
        raise ValueError("no numeric feature columns")
    return cols


def run(path: Path, *, target: str, time_col: str, seed: int) -> dict:
    frame = pd.read_csv(path)
    if target not in frame:
        raise ValueError(f"missing target column: {target}")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    if frame[target].isna().any() or not np.isfinite(frame[target]).all():
        raise ValueError("target must be finite")
    parts = chronological_split_v52(frame, time_col=time_col, config=SplitConfigV52())
    features = _feature_columns(frame, target, time_col)

    scaler = StandardScaler().fit(parts["train"][features])
    x_train = scaler.transform(parts["train"][features])
    x_test = scaler.transform(parts["test"][features])
    y_train = parts["train"][target].to_numpy(dtype=float)
    y_test = parts["test"][target].to_numpy(dtype=float)

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores = np.asarray(model.predict(x_test), dtype=float)
        results[name] = evaluate_scores_v52(scores, y_test, costs=CostConfigV52())

    return {
        "status": "RESEARCH_ONLY",
        "seed": seed,
        "features": features,
        "rows": {k: len(v) for k, v in parts.items()},
        "costs_bps": {"base": 24.0, "stress": 36.0},
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", default="future_return")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.csv, target=args.target, time_col=args.time_col, seed=args.seed)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
