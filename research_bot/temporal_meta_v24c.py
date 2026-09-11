from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research_bot.deep_temporal_v22 import TemporalConfig, causal_bar_features, fit_binary_model, make_model, predict_probability
from research_bot.strategy_family_meta_v24b import V24BContract, choose_family_threshold


@dataclass(frozen=True)
class TemporalMetaContract:
    lookback: int = 64
    hidden_dim: int = 48
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 2
    seeds: tuple[int, ...] = (314, 2718, 1618)
    model_kinds: tuple[str, ...] = ("lstm", "gru", "tcn", "cnn_lstm", "transformer")
    min_development_events: int = 250
    min_validation_events: int = 80
    min_validation_selected: int = 25
    test_may_select_model: bool = False
    live_execution_authorized: bool = False

    def torch_config(self) -> TemporalConfig:
        return TemporalConfig(
            lookback=self.lookback,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def causal_normalized_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build feature values whose normalization at t uses observations <= t-1."""
    feat = causal_bar_features(frame)
    cols = [c for c in feat.columns if c != "timestamp"]
    f = feat[cols].astype(float)
    mean = f.expanding(min_periods=30).mean().shift(1)
    std = f.expanding(min_periods=30).std(ddof=0).shift(1).replace(0, np.nan)
    z = ((f - mean) / std).clip(-10, 10)
    out = pd.concat([pd.to_datetime(feat["timestamp"], utc=True).rename("timestamp"), z], axis=1)
    return out, cols


def build_event_sequences(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lookback: int = 64,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Align causal OHLCV sequence windows to strategy signal timestamps.

    The label is never used to construct X. Each sequence ends at the signal close,
    and expanding normalization uses only information strictly before each row.
    """
    if lookback < 8:
        raise ValueError("lookback must be >= 8")
    if events.empty:
        return np.empty((0, lookback, 0), dtype=np.float32), events.copy(), []
    normalized, cols = causal_normalized_feature_frame(frame)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    matrix = normalized[cols].to_numpy(dtype=np.float32)
    time_to_idx = {t: i for i, t in enumerate(normalized["timestamp"].tolist())}
    x: list[np.ndarray] = []
    keep: list[int] = []
    e = events.copy().reset_index(drop=True)
    e["signal_time"] = pd.to_datetime(e["signal_time"], utc=True)
    for i, t in enumerate(e["signal_time"]):
        j = time_to_idx.get(t)
        if j is None or j - lookback + 1 < 0:
            continue
        window = matrix[j - lookback + 1 : j + 1]
        if window.shape != (lookback, len(cols)) or not np.isfinite(window).all():
            continue
        x.append(window)
        keep.append(i)
    arr = np.stack(x).astype(np.float32) if x else np.empty((0, lookback, len(cols)), dtype=np.float32)
    meta = e.iloc[keep].reset_index(drop=True)
    return arr, meta, cols


def build_multisymbol_sequences(
    frames: Mapping[str, pd.DataFrame],
    events: pd.DataFrame,
    *,
    lookback: int = 64,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    parts_x: list[np.ndarray] = []
    parts_meta: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None
    for symbol, group in events.groupby("symbol", sort=True):
        if symbol not in frames:
            continue
        x, meta, cols = build_event_sequences(frames[symbol], group, lookback=lookback)
        if feature_cols is None:
            feature_cols = cols
        elif cols != feature_cols:
            raise RuntimeError("temporal feature schema mismatch across symbols")
        if len(meta):
            parts_x.append(x)
            parts_meta.append(meta)
    if not parts_x:
        nfeat = len(feature_cols or [])
        return np.empty((0, lookback, nfeat), dtype=np.float32), pd.DataFrame(), feature_cols or []
    x_all = np.concatenate(parts_x, axis=0)
    meta_all = pd.concat(parts_meta, ignore_index=True)
    order = np.argsort(pd.to_datetime(meta_all["signal_time"], utc=True).astype("int64").to_numpy(), kind="mergesort")
    return x_all[order], meta_all.iloc[order].reset_index(drop=True), feature_cols or []


def temporal_validation_tournament(
    frames: Mapping[str, pd.DataFrame],
    dataset: pd.DataFrame,
    *,
    strategies: tuple[str, ...] = ("H4_S6_BREAKOUT", "H4_D1_OB_BOS_RISK"),
    contract: TemporalMetaContract | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Development-fit / validation-only temporal challenger tournament.

    The already-observed v0.24b terminal segment is deliberately not scored here.
    A temporal champion must later consume fresh external/future evidence.
    """
    c = contract or TemporalMetaContract()
    cfg = c.torch_config()
    rows: list[dict] = []
    frozen: dict = {}
    threshold_contract = V24BContract(min_validation_selected=c.min_validation_selected)

    for strategy in strategies:
        fam = dataset[dataset["strategy"] == strategy].copy()
        dev_events = fam[fam["segment"] == "development"].copy()
        val_events = fam[fam["segment"] == "validation"].copy()
        if len(dev_events) < c.min_development_events or len(val_events) < c.min_validation_events:
            rows.append({
                "strategy": strategy,
                "status": "DATA_INSUFFICIENT",
                "development_events": int(len(dev_events)),
                "validation_events": int(len(val_events)),
            })
            continue
        x_dev, m_dev, features = build_multisymbol_sequences(frames, dev_events, lookback=c.lookback)
        x_val, m_val, val_features = build_multisymbol_sequences(frames, val_events, lookback=c.lookback)
        if features != val_features:
            raise RuntimeError("development/validation temporal feature schema mismatch")
        if len(m_dev) < c.min_development_events or len(m_val) < c.min_validation_events:
            rows.append({
                "strategy": strategy,
                "status": "DATA_INSUFFICIENT_AFTER_SEQUENCE_ALIGNMENT",
                "development_sequences": int(len(m_dev)),
                "validation_sequences": int(len(m_val)),
            })
            continue
        y_dev = m_dev["label_meta_execute"].astype(float).to_numpy(np.float32)
        y_val = m_val["label_meta_execute"].astype(float).to_numpy(np.float32)
        search: list[tuple[float, dict, object]] = []
        for seed in c.seeds:
            for kind in c.model_kinds:
                try:
                    model = make_model(kind, x_dev.shape[-1], cfg)
                    fitted, history = fit_binary_model(model, x_dev, y_dev, x_val, y_val, cfg, seed=seed)
                    score = predict_probability(fitted, x_val)
                    threshold, val_result = choose_family_threshold(m_val, score, threshold_contract)
                    objective = float(val_result["validation_objective"])
                    row = {
                        "strategy": strategy,
                        "status": "ok",
                        "model_kind": kind,
                        "seed": int(seed),
                        "threshold": float(threshold),
                        "validation_objective": objective,
                        "validation_selected": int(val_result["filtered"]["selected"]),
                        "validation_mean_r": float(val_result["filtered"]["mean_r"]),
                        "validation_profit_factor": float(val_result["filtered"]["profit_factor"]),
                        "validation_total_return": float(val_result["filtered"]["total_return"]),
                        "validation_max_drawdown": float(val_result["filtered"]["max_drawdown"]),
                        "epochs_ran": int(len(history)),
                        "development_sequences": int(len(m_dev)),
                        "validation_sequences": int(len(m_val)),
                        "feature_count": int(len(features)),
                    }
                    rows.append(row)
                    if np.isfinite(objective):
                        search.append((objective, row, fitted))
                except Exception as exc:
                    rows.append({
                        "strategy": strategy,
                        "status": "failed",
                        "model_kind": kind,
                        "seed": int(seed),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        if search:
            search.sort(key=lambda z: (z[0], z[1]["validation_selected"], z[1]["model_kind"], -z[1]["seed"]), reverse=True)
            _, champion_row, champion_model = search[0]
            frozen[strategy] = {
                "strategy": strategy,
                "model_kind": champion_row["model_kind"],
                "seed": champion_row["seed"],
                "threshold": champion_row["threshold"],
                "validation_objective": champion_row["validation_objective"],
                "feature_names": features,
                "model": champion_model,
                "terminal_test_scored": False,
                "live_execution_authorized": False,
            }
    return pd.DataFrame(rows), frozen
