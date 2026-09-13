from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_dataset(path: Path, *, time_col: str, target: str) -> dict:
    frame = pd.read_csv(path)
    if time_col not in frame or target not in frame:
        raise ValueError("dataset must contain time and target columns")
    ts = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("dataset time axis must be valid, strictly ordered and unique")
    y = pd.to_numeric(frame[target], errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("target must be finite")
    numeric_features = [c for c in frame.columns if c not in {time_col, target} and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric_features:
        raise ValueError("dataset requires at least one numeric feature")
    bad = [c for c in numeric_features if not np.isfinite(pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)).all()]
    if bad:
        raise ValueError(f"non-finite numeric feature columns: {bad}")
    return {
        "experiment": "v0.52",
        "role": "frozen_benchmark_dataset",
        "path_name": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "numeric_features": numeric_features,
        "time_col": time_col,
        "target": target,
        "start": ts.iloc[0].isoformat(),
        "end": ts.iloc[-1].isoformat(),
        "paper_execution": False,
        "live_execution": False,
        "kraken_authorized": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--target", default="future_return")
    p.add_argument("--output", type=Path, default=Path("artifacts/v52/dataset_manifest.json"))
    args = p.parse_args()
    manifest = freeze_dataset(args.csv, time_col=args.time_col, target=args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
