from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import performance_metrics


FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "SMC_ICT": ("smc_", "ict_"),
    "BROOKS": ("brooks_",),
    "ICHIMOKU": ("ichi_",),
    "HTF": ("4h_",),
}

EXCLUDED_COLUMNS = {
    "timestamp", "decision_at", "bar_open_at", "bar_close_at", "available_at",
    "1h_available_at", "4h_available_at", "asset", "symbol", "timeframe",
    "open", "high", "low", "close", "volume", "future_return", "target_up",
}


@dataclass(frozen=True)
class V54AuditConfig:
    horizon_bars: int = 1
    min_train_rows: int = 600
    test_rows: int = 180
    step_rows: int = 180
    purge_bars: int = 1
    ridge_alpha: float = 5.0
    round_trip_cost_bps: tuple[float, ...] = (0.0, 24.0, 36.0)
    min_features_per_family: int = 2
    timeframe: str = "1h"
    decision_time_col: str = "decision_at"

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0 or self.min_train_rows < 100 or self.test_rows < 30 or self.step_rows <= 0:
            raise ValueError("invalid walk-forward configuration")
        if self.purge_bars < self.horizon_bars:
            raise ValueError("purge_bars must cover the target horizon")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if not self.round_trip_cost_bps or min(self.round_trip_cost_bps) < 0:
            raise ValueError("cost grid must be non-negative")
        if not self.decision_time_col:
            raise ValueError("decision_time_col required")


def stable_audit_fingerprint(frame: pd.DataFrame) -> str:
    required = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if "timestamp" not in required:
        raise ValueError("timestamp required for audit fingerprint")
    x = frame[required].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise").astype(str)
    raw = x.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    return sha256(raw).hexdigest()


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=float(alpha))),
    ])


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in frame.columns:
        if c in EXCLUDED_COLUMNS or c.endswith("_available_at"):
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            cols.append(c)
    return sorted(cols)


def feature_families_v54(frame: pd.DataFrame, *, min_features: int = 2) -> dict[str, list[str]]:
    numeric = _numeric_feature_columns(frame)
    families: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for name, prefixes in FAMILY_PREFIXES.items():
        cols = [c for c in numeric if c.startswith(prefixes)]
        if len(cols) >= min_features:
            families[name] = cols
            claimed.update(cols)
    base = [c for c in numeric if c not in claimed]
    if len(base) >= min_features:
        families["BASE"] = base
    if not families:
        raise ValueError("no auditable feature families found")
    return families


def _prepare_panel(frame: pd.DataFrame, cfg: V54AuditConfig) -> pd.DataFrame:
    x = frame.copy()
    if "timestamp" not in x or "close" not in x or cfg.decision_time_col not in x:
        raise ValueError("timestamp, close and decision_at are required")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    x[cfg.decision_time_col] = pd.to_datetime(x[cfg.decision_time_col], utc=True, errors="raise")
    x = x.sort_values(cfg.decision_time_col, kind="mergesort").reset_index(drop=True)
    if x[cfg.decision_time_col].duplicated().any():
        raise ValueError("duplicate decision timestamps forbidden in v0.54")
    if "bar_close_at" in x:
        close_at = pd.to_datetime(x["bar_close_at"], utc=True, errors="raise")
        if (x[cfg.decision_time_col] < close_at).any():
            raise ValueError("decision precedes decision-bar close")
    for c in [c for c in x.columns if c == "available_at" or c.endswith("_available_at")]:
        avail = pd.to_datetime(x[c], utc=True, errors="coerce")
        bad = avail.notna() & (avail > x[cfg.decision_time_col])
        if bad.any():
            raise ValueError(f"future availability leakage in {c}")
    x["future_return"] = x["close"].shift(-cfg.horizon_bars) / x["close"] - 1.0
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_return"]).reset_index(drop=True)
    return x


def expanding_folds_v54(n: int, cfg: V54AuditConfig) -> list[tuple[slice, slice]]:
    folds: list[tuple[slice, slice]] = []
    test_start = cfg.min_train_rows
    while test_start + cfg.test_rows <= n:
        train_end = test_start - cfg.purge_bars
        if train_end < cfg.min_train_rows - cfg.purge_bars:
            break
        folds.append((slice(0, train_end), slice(test_start, test_start + cfg.test_rows)))
        test_start += cfg.step_rows
    if not folds:
        raise ValueError("insufficient rows for v0.54 walk-forward folds")
    return folds


def _backtest_predictions(test: pd.DataFrame, pred: np.ndarray, cost_bps: float, *, decision_time_col: str) -> pd.DataFrame:
    if len(test) != len(pred):
        raise ValueError("prediction length mismatch")
    position = np.sign(np.asarray(pred, dtype=float))
    previous = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - previous)
    one_way = float(cost_bps) / 20_000.0
    gross = position * test["future_return"].to_numpy(dtype=float)
    net = gross - turnover * one_way
    return pd.DataFrame({
        "decision_at": test[decision_time_col].to_numpy(),
        "position": position,
        "turnover": turnover,
        "gross_return": gross,
        "net_return": net,
    })


def _metrics(bt: pd.DataFrame, timeframe: str) -> dict:
    m = performance_metrics(bt["net_return"], timeframe=timeframe)
    m.update({
        "n": int(len(bt)),
        "net_total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
        "gross_total_return": float((1.0 + bt["gross_return"]).prod() - 1.0),
        "turnover_sum": float(bt["turnover"].sum()),
        "exposure": float((bt["position"] != 0).mean()),
    })
    return m


def _run_variant(panel: pd.DataFrame, features: list[str], cfg: V54AuditConfig) -> dict:
    if not features:
        raise ValueError("variant has no features")
    fold_rows: list[dict] = []
    combined: dict[float, list[pd.DataFrame]] = {c: [] for c in cfg.round_trip_cost_bps}
    folds = expanding_folds_v54(len(panel), cfg)
    for fold_id, (tr_slice, te_slice) in enumerate(folds, start=1):
        train = panel.iloc[tr_slice].copy()
        test = panel.iloc[te_slice].copy()
        if train.empty or test.empty:
            continue
        if train.index.max() >= test.index.min() - cfg.purge_bars:
            raise AssertionError("purge invariant violated")
        model = _ridge(cfg.ridge_alpha)
        model.fit(train[features], train["future_return"].astype(float))
        pred = np.asarray(model.predict(test[features]), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError("non-finite predictions")
        fold_row = {
            "fold": fold_id,
            "train_start": pd.Timestamp(train[cfg.decision_time_col].iloc[0]).isoformat(),
            "train_end": pd.Timestamp(train[cfg.decision_time_col].iloc[-1]).isoformat(),
            "test_start": pd.Timestamp(test[cfg.decision_time_col].iloc[0]).isoformat(),
            "test_end": pd.Timestamp(test[cfg.decision_time_col].iloc[-1]).isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
        for cost in cfg.round_trip_cost_bps:
            bt = _backtest_predictions(test, pred, cost, decision_time_col=cfg.decision_time_col)
            combined[cost].append(bt)
            fold_row[f"cost_{cost:g}"] = _metrics(bt, cfg.timeframe)
        fold_rows.append(fold_row)
    aggregate: dict[str, dict] = {}
    for cost, parts in combined.items():
        if not parts:
            continue
        z = pd.concat(parts, ignore_index=True).sort_values("decision_at").reset_index(drop=True)
        aggregate[f"cost_{cost:g}"] = _metrics(z, cfg.timeframe)
    return {"features": features, "folds": fold_rows, "aggregate": aggregate}


def _finite_metric(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def audit_feature_families_v54(frame: pd.DataFrame, config: V54AuditConfig | None = None) -> dict:
    cfg = config or V54AuditConfig()
    panel = _prepare_panel(frame, cfg)
    families = feature_families_v54(panel, min_features=cfg.min_features_per_family)
    all_features = sorted({c for cols in families.values() for c in cols})
    variants: dict[str, list[str]] = {
        "ALL": all_features,
        **{f"ONLY_{name}": cols for name, cols in families.items()},
        **{f"DROP_{name}": [c for c in all_features if c not in set(cols)] for name, cols in families.items()},
    }
    results = {name: _run_variant(panel, cols, cfg) for name, cols in variants.items() if cols}

    base_key = "cost_24" if 24.0 in cfg.round_trip_cost_bps else f"cost_{cfg.round_trip_cost_bps[0]:g}"
    all_sharpe = _finite_metric(results["ALL"]["aggregate"].get(base_key, {}).get("sharpe"))
    family_value: dict[str, dict] = {}
    for name in families:
        drop = results.get(f"DROP_{name}", {}).get("aggregate", {}).get(base_key, {})
        only = results.get(f"ONLY_{name}", {}).get("aggregate", {}).get(base_key, {})
        drop_sharpe = _finite_metric(drop.get("sharpe"))
        only_sharpe = _finite_metric(only.get("sharpe"))
        family_value[name] = {
            "all_minus_drop_sharpe": all_sharpe - drop_sharpe,
            "only_family_sharpe": only_sharpe,
            "positive_increment": bool(all_sharpe > drop_sharpe),
        }

    return {
        "experiment": "V54_FEATURE_FAMILY_WALK_FORWARD_AUDIT",
        "status": "RESEARCH_ONLY_NO_EXECUTION_AUTHORIZATION",
        "config": asdict(cfg),
        "dataset_fingerprint": stable_audit_fingerprint(frame),
        "rows_input": int(len(frame)),
        "rows_panel": int(len(panel)),
        "families": families,
        "variants": results,
        "family_value": family_value,
        "paper_execution": False,
        "live_execution": False,
    }
