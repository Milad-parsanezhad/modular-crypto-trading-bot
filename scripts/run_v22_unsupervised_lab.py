from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from research_bot.unsupervised_v22 import RegimeAnomalyFeatureLab, UnsupervisedConfig, tail_enrichment


def _sample_for_silhouette(x: np.ndarray, labels: np.ndarray, limit: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= limit:
        return x, labels
    idx = np.linspace(0, len(x) - 1, limit, dtype=int)
    return x[idx], labels[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.22 unsupervised regime and anomaly representation lab")
    ap.add_argument("--dataset", default="artifacts/v21-ml-meta/labeled_ml_dataset.csv")
    ap.add_argument("--output-dir", default="artifacts/v22-unsupervised")
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--clusters", type=int, default=8)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    df = pd.read_csv(dataset_path)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    dev = df[df["segment"] == "development"].sort_values("signal_time").copy()
    val = df[df["segment"] == "validation"].sort_values("signal_time").copy()
    test = df[df["segment"] == "test"].sort_values("signal_time").copy()
    if min(len(dev), len(val), len(test)) == 0:
        raise RuntimeError("development/validation/test are all required")

    lab = RegimeAnomalyFeatureLab(UnsupervisedConfig(n_components=args.components, n_clusters=args.clusters))
    lab.fit(dev)
    lab.save(output)

    frames = {}
    for name, part in [("development", dev), ("validation", val), ("test", test)]:
        rep = lab.transform(part).reset_index(drop=True)
        keep = part[["signal_time", "strategy", "timeframe", "symbol", "segment", "label_profitable_net", "label_r_multiple"]].reset_index(drop=True)
        merged = pd.concat([keep, rep], axis=1)
        merged.to_csv(output / f"{name}_unsupervised_features.csv", index=False)
        frames[name] = merged

    validation_audit = {}
    test_audit = {}
    anomaly_cols = [c for c in frames["validation"].columns if c.endswith("_anomaly")]
    for col in anomaly_cols:
        validation_audit[col] = tail_enrichment(frames["validation"], col)
        test_audit[col] = tail_enrichment(frames["test"], col)

    # Silhouette is descriptive only, calculated after development fit.
    silhouette = None
    try:
        pca_cols = [c for c in frames["validation"].columns if c.startswith("u_pca_")]
        x = frames["validation"][pca_cols].to_numpy(float)
        labels = frames["validation"]["u_kmeans_regime"].to_numpy(int)
        x, labels = _sample_for_silhouette(x, labels)
        if len(np.unique(labels)) > 1:
            silhouette = float(silhouette_score(x, labels))
    except Exception:
        silhouette = None

    summary = {
        "version": "v0.22",
        "status": "UNSUPERVISED_REPRESENTATION_COMPLETE",
        "fit_rows": int(len(dev)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        "input_features": lab.columns,
        "pca_components": int(lab.pca.n_components_) if lab.pca is not None else None,
        "pca_explained_variance_ratio": lab.pca.explained_variance_ratio_.tolist() if lab.pca is not None else None,
        "validation_kmeans_silhouette_descriptive": silhouette,
        "validation_tail_enrichment": validation_audit,
        "test_tail_enrichment": test_audit,
        "decision": "REPRESENTATIONS_AVAILABLE_FOR_DOWNSTREAM_ABLATION",
        "direct_trading_authorized": False,
        "live_execution_authorized": False,
    }
    (output / "decision.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
