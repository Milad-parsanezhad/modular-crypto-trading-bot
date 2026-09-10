from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


BANNED_FEATURE_EXACT = {
    "label", "target", "future_return", "future_log_return", "future_price",
    "entry", "exit", "stop", "take_profit", "r_multiple", "realized_return",
    "net_return", "gross_return", "pnl", "equity_after", "exit_reason",
    "executed", "filled", "fill_price", "max_favorable_excursion",
    "max_adverse_excursion",
}
BANNED_SUBSTRINGS = (
    "future_", "_future", "realized_", "outcome", "label_", "target_",
    "exit_", "pnl", "profit_after", "equity_after",
)
IDENTITY_FEATURES = ("symbol", "strategy", "family")
CONTEXT_FEATURES = ("timeframe", "side")


@dataclass(frozen=True)
class SplitContract:
    development_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    embargo_rows: int = 1

    def validate(self) -> None:
        total = self.development_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("split fractions must sum to 1")
        if min(self.development_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("all split fractions must be positive")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows must be >= 0")


@dataclass(frozen=True)
class CostContract:
    fee_bps_each_way: float = 10.0
    slippage_bps_each_way: float = 2.0

    @property
    def roundtrip_fraction(self) -> float:
        return 2.0 * (self.fee_bps_each_way + self.slippage_bps_each_way) / 10000.0


@dataclass(frozen=True)
class RiskContract:
    base_risk_per_trade: float = 0.0025
    max_drawdown: float = 0.05
    min_validation_selected: int = 200
    min_test_selected: int = 100
    min_profit_factor: float = 1.05


@dataclass(frozen=True)
class ModelSearchContract:
    seeds: tuple[int, ...] = (314, 2718, 1618)
    threshold_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95)
    allow_identity_features: bool = False
    test_may_select_model: bool = False


@dataclass(frozen=True)
class MLResearchContract:
    split: SplitContract = SplitContract()
    cost: CostContract = CostContract()
    risk: RiskContract = RiskContract()
    search: ModelSearchContract = ModelSearchContract()
    live_execution_authorized: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roundtrip_cost_fraction"] = self.cost.roundtrip_fraction
        return d


def dataframe_sha256(df: pd.DataFrame) -> str:
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    payload = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return sha256(payload).hexdigest()


def assert_unique_columns(df: pd.DataFrame) -> None:
    duplicate = df.columns[df.columns.duplicated()].tolist()
    if duplicate:
        raise ValueError(f"duplicate columns are forbidden: {duplicate}")


def _is_banned_name(name: str) -> bool:
    low = name.lower().strip()
    if low in BANNED_FEATURE_EXACT:
        return True
    return any(token in low for token in BANNED_SUBSTRINGS)


def select_feature_columns(
    df: pd.DataFrame,
    *,
    prefix: str = "f_",
    include_context: bool = True,
    include_identity: bool = False,
) -> tuple[list[str], list[str]]:
    """Strict feature whitelist. Outcome columns cannot become inputs by accident."""
    assert_unique_columns(df)
    numeric: list[str] = []
    for c in df.columns:
        if c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c]):
            if _is_banned_name(c):
                raise ValueError(f"banned outcome-like feature name: {c}")
            numeric.append(c)
    categorical: list[str] = []
    if include_context:
        categorical.extend([c for c in CONTEXT_FEATURES if c in df.columns])
    if include_identity:
        categorical.extend([c for c in IDENTITY_FEATURES if c in df.columns])
    for c in categorical:
        if _is_banned_name(c):
            raise ValueError(f"banned categorical feature name: {c}")
    if not numeric:
        raise ValueError("no numeric f_* features found")
    return sorted(numeric), categorical


def chronological_purged_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "signal_time",
    contract: SplitContract | None = None,
) -> dict[str, pd.DataFrame]:
    contract = contract or SplitContract()
    contract.validate()
    assert_unique_columns(df)
    if timestamp_col not in df:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    x = df.copy()
    x[timestamp_col] = pd.to_datetime(x[timestamp_col], utc=True, errors="coerce")
    if x[timestamp_col].isna().any():
        raise ValueError("timestamp parse failure")
    x = x.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(x)
    if n < 30:
        raise ValueError("dataset too small for development/validation/test research split")
    a = int(n * contract.development_fraction)
    b = int(n * (contract.development_fraction + contract.validation_fraction))
    e = contract.embargo_rows
    dev_end = max(0, a - e)
    val_start = min(n, a + e)
    val_end = max(val_start, b - e)
    test_start = min(n, b + e)
    out = {
        "development": x.iloc[:dev_end].copy(),
        "validation": x.iloc[val_start:val_end].copy(),
        "test": x.iloc[test_start:].copy(),
    }
    if min(len(v) for v in out.values()) == 0:
        raise ValueError("embargo/split produced an empty segment")
    assert_temporal_separation(out, timestamp_col=timestamp_col)
    return out


def assert_temporal_separation(splits: Mapping[str, pd.DataFrame], *, timestamp_col: str = "signal_time") -> None:
    required = ("development", "validation", "test")
    if any(k not in splits for k in required):
        raise ValueError("development/validation/test splits are required")
    bounds = {}
    for k in required:
        t = pd.to_datetime(splits[k][timestamp_col], utc=True)
        if not t.is_monotonic_increasing:
            raise ValueError(f"{k} timestamps are not monotonic")
        bounds[k] = (t.min(), t.max())
    if not (bounds["development"][1] < bounds["validation"][0] < bounds["validation"][1] < bounds["test"][0]):
        raise ValueError(f"temporal split overlap detected: {bounds}")


def cost_aware_next_open_labels(frame: pd.DataFrame, cost: CostContract | None = None) -> pd.DataFrame:
    """Label close[t] decisions by open[t+1] -> open[t+2] net hurdle.

    The future return is emitted only as a target column. It must never be reused
    as an input feature. The final two rows are dropped because their targets are
    not fully observable.
    """
    cost = cost or CostContract()
    required = {"timestamp", "open"}
    if required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    x = frame.copy().sort_values("timestamp").reset_index(drop=True)
    x["signal_time"] = pd.to_datetime(x["timestamp"], utc=True)
    entry = x["open"].shift(-1).astype(float)
    exit_ = x["open"].shift(-2).astype(float)
    gross = exit_ / entry - 1.0
    net = gross - cost.roundtrip_fraction
    x["label_future_gross_return"] = gross
    x["label_future_net_return"] = net
    x["label_positive_net"] = (net > 0).astype("int8")
    x["label_direction_3class"] = np.select(
        [net > 0, net < 0], [1, -1], default=0,
    ).astype("int8")
    return x.iloc[:-2].copy()


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
    ])
    transformers = [("num", num, numeric)]
    if categorical:
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def supervised_model_registry(seed: int = 314) -> dict[str, object]:
    """Compact first-pass registry; deep models are evaluated in a separate sequence track."""
    return {
        "logistic": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        "ridge_classifier": RidgeClassifier(class_weight="balanced"),
        "sgd_logistic": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=20, class_weight="balanced_subsample", random_state=seed, n_jobs=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, max_depth=12, min_samples_leaf=15, class_weight="balanced", random_state=seed, n_jobs=2),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
    }


def unsupervised_model_registry(seed: int = 314) -> dict[str, object]:
    """Regime/anomaly discovery only; unsupervised outputs are not promoted as alpha by themselves."""
    return {
        "isolation_forest": IsolationForest(n_estimators=250, contamination="auto", random_state=seed, n_jobs=2),
        "gmm_3": GaussianMixture(n_components=3, covariance_type="full", random_state=seed, reg_covar=1e-6),
        "gmm_5": GaussianMixture(n_components=5, covariance_type="diag", random_state=seed, reg_covar=1e-6),
    }


def probability_score(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if np.ndim(p) == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))
    p = np.asarray(model.predict(x), dtype=float).ravel()
    return np.clip(p, 0.0, 1.0)


def classification_metrics(y_true: Iterable[int], score: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(score), dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        out["log_loss"] = np.nan
    return out


def economic_metrics(net_returns: Iterable[float], selected: Iterable[bool], *, risk: RiskContract | None = None) -> dict[str, float | int]:
    risk = risk or RiskContract()
    r = np.asarray(list(net_returns), dtype=float)
    s = np.asarray(list(selected), dtype=bool)
    r = r[s]
    if len(r) == 0:
        return {"selected": 0, "coverage": 0.0, "mean_net_return": np.nan, "profit_factor": np.nan, "total_return": np.nan, "max_drawdown": np.nan}
    account = np.clip(r, -0.99, None) * risk.base_risk_per_trade
    equity = np.cumprod(1.0 + account)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "selected": int(len(r)),
        "coverage": float(len(r) / max(len(s), 1)),
        "mean_net_return": float(np.mean(r)),
        "profit_factor": float(pf),
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(dd.min()),
    }


def choose_validation_threshold(
    score: Iterable[float],
    net_return: Iterable[float],
    *,
    search: ModelSearchContract | None = None,
    risk: RiskContract | None = None,
) -> tuple[float, dict]:
    search = search or ModelSearchContract()
    risk = risk or RiskContract()
    p = np.asarray(list(score), dtype=float)
    r = np.asarray(list(net_return), dtype=float)
    best_t, best, best_objective = 1.0, None, -np.inf
    for q in search.threshold_quantiles:
        t = float(np.quantile(p, q))
        m = economic_metrics(r, p >= t, risk=risk)
        if int(m["selected"]) < risk.min_validation_selected:
            continue
        pf = float(m["profit_factor"])
        mean = float(m["mean_net_return"])
        dd = abs(float(m["max_drawdown"]))
        if not (np.isfinite(mean) and np.isfinite(dd) and np.isfinite(pf)):
            continue
        objective = mean * np.sqrt(int(m["selected"])) + 0.10 * (min(pf, 5.0) - 1.0) - 0.75 * dd
        if objective > best_objective:
            best_t, best, best_objective = t, m, objective
    if best is None:
        best = economic_metrics(r, np.zeros(len(r), dtype=bool), risk=risk)
    return best_t, {**best, "validation_objective": float(best_objective)}


def promotion_decision(test_metrics: Mapping[str, float | int], *, risk: RiskContract | None = None) -> dict:
    risk = risk or RiskContract()
    n = int(test_metrics.get("selected", 0))
    mean = float(test_metrics.get("mean_net_return", np.nan))
    pf = float(test_metrics.get("profit_factor", np.nan))
    dd = abs(float(test_metrics.get("max_drawdown", np.nan)))
    passed = bool(
        n >= risk.min_test_selected
        and np.isfinite(mean) and mean > 0
        and np.isfinite(pf) and pf >= risk.min_profit_factor
        and np.isfinite(dd) and dd <= risk.max_drawdown
    )
    return {
        "decision": "FORWARD_PAPER_ML_CANDIDATE" if passed else "NO_ML_MODEL_PROMOTED",
        "test_pass": passed,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def write_contract_manifest(output_dir: str | Path, contract: MLResearchContract, *, extra: dict | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"contract": contract.to_dict(), "extra": extra or {}}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["manifest_sha256"] = sha256(raw).hexdigest()
    path = out / "ml_research_contract.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
