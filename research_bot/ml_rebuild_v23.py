from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SEGMENTS = ("development", "validation", "test")
CATEGORICAL_FEATURES = ("strategy", "family", "timeframe", "symbol", "side")
LABEL_COLUMNS = (
    "label_profitable_net",
    "label_target_hit",
    "label_ge_1r",
    "label_ge_2r",
    "label_outcome_3class",
    "label_r_multiple",
)
BANNED_OUTCOME_COLUMNS = {
    "entry", "exit", "stop", "target", "entry_time", "exit_time", "exit_reason",
    "gross_return", "net_return", "realized_return", "r_multiple",
    "account_return", "account_return_v20", "equity", "equity_v20",
    "executed", "executed_v20", "reject_reason", "reject_reason_v20",
    "risk_fraction", "risk_fraction_v20", "future_return", "future_log_return",
}


@dataclass(frozen=True)
class MLRebuildConfig:
    seed: int = 314
    max_fit_rows: int = 60000
    max_fit_rows_slow: int = 12000
    min_selected_validation: int = 200
    min_selected_test: int = 100
    base_risk_per_trade: float = 0.0025
    max_risk_per_timestamp: float = 0.0100
    max_drawdown: float = 0.05
    min_test_profit_factor: float = 1.05
    min_test_mean_r: float = 0.0
    bootstrap_resamples: int = 500
    bootstrap_block: int = 12
    min_bootstrap_observations: int = 60
    require_nonnegative_bootstrap_lower: bool = True


def assert_unique_columns(frame: pd.DataFrame) -> None:
    dup = frame.columns[frame.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"duplicate columns are forbidden: {dup}")


def safe_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    assert_unique_columns(frame)
    numeric = sorted(
        c for c in frame.columns
        if c.startswith("f_") and pd.api.types.is_numeric_dtype(frame[c])
    )
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    selected = set(numeric) | set(categorical)
    illegal = selected & (BANNED_OUTCOME_COLUMNS | set(LABEL_COLUMNS))
    if illegal:
        raise ValueError(f"outcome leakage in feature set: {sorted(illegal)}")
    for c in numeric:
        lc = c.lower()
        if any(tok in lc for tok in (
            "future", "label_", "realized", "exit_", "target_", "stop_",
            "pnl", "profit_after", "equity_after",
        )):
            raise ValueError(f"suspicious future/outcome feature: {c}")
    if not numeric:
        raise ValueError("no numeric f_* features found")
    return numeric, categorical


def audit_labeled_dataset(frame: pd.DataFrame) -> dict:
    assert_unique_columns(frame)
    required = {
        "signal_time", "segment", "strategy", "timeframe", "symbol",
        "label_profitable_net", "label_r_multiple",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    x = frame.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if x["signal_time"].isna().any():
        raise ValueError("invalid signal_time values")
    present = set(x["segment"].dropna().astype(str).unique())
    if not set(SEGMENTS).issubset(present):
        raise ValueError(f"development/validation/test required; got {sorted(present)}")
    if x[["label_profitable_net", "label_r_multiple"]].isna().any().any():
        raise ValueError("missing primary labels")
    numeric, categorical = safe_feature_columns(x)

    violations = []
    group_cols = ["strategy", "timeframe", "symbol"]
    for key, g in x.groupby(group_cols, dropna=False):
        times = {
            seg: g.loc[g["segment"].eq(seg), "signal_time"]
            for seg in SEGMENTS
        }
        if all(len(times[s]) for s in SEGMENTS):
            if not (times["development"].max() < times["validation"].min()
                    and times["validation"].max() < times["test"].min()):
                violations.append(key)
    if violations:
        raise ValueError(f"chronological split overlap in {len(violations)} groups; first={violations[0]}")

    event_key = ["signal_time", "strategy", "timeframe", "symbol"]
    if "side" in x.columns:
        event_key.append("side")
    exact_dupes = int(x.duplicated(event_key).sum())
    return {
        "rows": int(len(x)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "segment_counts": {str(k): int(v) for k, v in x["segment"].value_counts().items()},
        "exact_duplicate_events": exact_dupes,
        "profitable_rate": float(x["label_profitable_net"].astype(float).mean()),
        "mean_r": float(x["label_r_multiple"].astype(float).mean()),
    }


@dataclass
class DevelopmentOnlyUnsupervised:
    feature_columns: list[str]
    medians: np.ndarray
    scaler: RobustScaler
    kmeans: KMeans
    gmm: GaussianMixture
    isolation: IsolationForest

    @classmethod
    def fit(cls, development: pd.DataFrame, feature_columns: Iterable[str], seed: int = 314) -> "DevelopmentOnlyUnsupervised":
        cols = list(feature_columns)
        arr = development[cols].astype(float).to_numpy()
        med = np.nanmedian(arr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        arr = np.where(np.isfinite(arr), arr, med)
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        z = scaler.fit_transform(arr)
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(z)
        gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=seed, n_init=3).fit(z)
        isolation = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed, n_jobs=2
        ).fit(z)
        return cls(cols, med.astype(float), scaler, kmeans, gmm, isolation)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        arr = frame[self.feature_columns].astype(float).to_numpy()
        arr = np.where(np.isfinite(arr), arr, self.medians)
        z = self.scaler.transform(arr)
        gmm_prob = self.gmm.predict_proba(z)
        out = pd.DataFrame(index=frame.index)
        out["f_u_kmeans_cluster"] = self.kmeans.predict(z).astype(float)
        out["f_u_gmm_cluster"] = self.gmm.predict(z).astype(float)
        out["f_u_anomaly_score"] = (-self.isolation.score_samples(z)).astype(float)
        out["f_u_gmm_confidence"] = gmm_prob.max(axis=1).astype(float)
        return out


def attach_development_only_unsupervised(frame: pd.DataFrame, seed: int = 314) -> tuple[pd.DataFrame, DevelopmentOnlyUnsupervised]:
    numeric, _ = safe_feature_columns(frame)
    dev = frame.loc[frame["segment"].eq("development")].copy()
    if len(dev) < 100:
        raise ValueError("at least 100 development rows required for unsupervised context")
    bank = DevelopmentOnlyUnsupervised.fit(dev, numeric, seed=seed)
    out = frame.copy()
    u = bank.transform(out)
    for c in u.columns:
        if c in out.columns:
            raise ValueError(f"unsupervised feature collision: {c}")
        out[c] = u[c]
    assert_unique_columns(out)
    return out, bank


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
            ]),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ),
    ], remainder="drop", verbose_feature_names_out=False)


def _optional_boosters(seed: int) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor
        out["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=seed, n_jobs=2,
        )
        out["xgboost_regressor"] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=2,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        out["lightgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=2, verbosity=-1,
        )
        out["lightgbm_regressor"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=2, verbosity=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        out["catboost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.04,
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
        out["catboost_regressor"] = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.04,
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
    except Exception:
        pass
    return out


def classifier_zoo(seed: int = 314) -> dict[str, object]:
    models: dict[str, object] = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1500, random_state=seed
        ),
        "ridge_classifier": RidgeClassifier(class_weight="balanced"),
        "sgd_logistic": SGDClassifier(
            loss="log_loss", alpha=1e-4, class_weight="balanced",
            max_iter=2000, random_state=seed,
        ),
        "linear_svc": LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
        "rbf_svc": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=2),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=40, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=seed, n_jobs=2
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=15,
            class_weight="balanced", random_state=seed, n_jobs=2
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.04, max_depth=3, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3,
            learning_rate_init=5e-4, max_iter=220,
            early_stopping=True, random_state=seed
        ),
    }
    opt = _optional_boosters(seed)
    for k in ("xgboost", "lightgbm", "catboost"):
        if k in opt:
            models[k] = opt[k]
    return models


def regressor_zoo(seed: int = 314) -> dict[str, object]:
    models: dict[str, object] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(n_jobs=2),
        "ridge": Ridge(alpha=2.0),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=3000, random_state=seed),
        "huber": HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=500),
        "svr": SVR(C=1.0, epsilon=0.1, kernel="rbf"),
        "knn_regressor": KNeighborsRegressor(n_neighbors=31, weights="distance", n_jobs=2),
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=8, min_samples_leaf=40, random_state=seed),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=20, random_state=seed, n_jobs=2
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=15, random_state=seed, n_jobs=2
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=3, loss="huber", random_state=seed
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        ),
        "mlp_regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3,
            learning_rate_init=5e-4, max_iter=220, early_stopping=True, random_state=seed
        ),
    }
    opt = _optional_boosters(seed)
    for k in ("xgboost_regressor", "lightgbm_regressor", "catboost_regressor"):
        if k in opt:
            models[k] = opt[k]
    return models


def _sample_fit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    idx = np.linspace(0, len(frame) - 1, limit, dtype=int)
    return frame.iloc[idx].copy()


def _score_vector(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        p = np.asarray(pipe.predict_proba(x), dtype=float)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    if hasattr(pipe, "decision_function"):
        raw = np.asarray(pipe.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
    return np.asarray(pipe.predict(x), dtype=float).ravel()


def classification_metrics(y: pd.Series | np.ndarray, score: np.ndarray) -> dict:
    yy = np.asarray(y, dtype=int)
    ss = np.asarray(score, dtype=float)
    pred = (ss >= 0.5).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(yy, pred))}
    if len(np.unique(yy)) > 1:
        out["roc_auc"] = float(roc_auc_score(yy, ss))
        out["average_precision"] = float(average_precision_score(yy, ss))
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
    clipped = np.clip(ss, 1e-6, 1 - 1e-6)
    try:
        out["log_loss"] = float(log_loss(yy, clipped, labels=[0, 1]))
        out["brier"] = float(brier_score_loss(yy, clipped))
    except Exception:
        out["log_loss"] = np.nan
        out["brier"] = np.nan
    return out


def portfolio_economic_metrics(
    rows: pd.DataFrame,
    selected: np.ndarray,
    config: MLRebuildConfig | None = None,
) -> dict:
    cfg = config or MLRebuildConfig()
    mask = np.asarray(selected, dtype=bool)
    if len(rows) != len(mask):
        raise ValueError("selected mask length mismatch")
    if not mask.any():
        return {
            "selected": 0, "coverage": 0.0, "timestamps": 0,
            "mean_r": np.nan, "profit_factor_r": np.nan, "win_rate": np.nan,
            "total_return": 0.0, "max_drawdown": 0.0,
            "mean_timestamp_return": np.nan,
        }
    z = rows.loc[mask].copy()
    z["signal_time"] = pd.to_datetime(z["signal_time"], utc=True)
    r = z["label_r_multiple"].astype(float)
    z["_event_risk_return"] = cfg.base_risk_per_trade * r

    def aggregate(g: pd.DataFrame) -> float:
        raw = g["_event_risk_return"].to_numpy(dtype=float)
        gross_risk = cfg.base_risk_per_trade * len(raw)
        scale = min(1.0, cfg.max_risk_per_timestamp / gross_risk) if gross_risk > 0 else 1.0
        return float(raw.sum() * scale)

    portfolio = z.groupby("signal_time", sort=True).apply(aggregate, include_groups=False)
    eq = (1.0 + portfolio).cumprod()
    dd = eq / eq.cummax() - 1.0
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "selected": int(len(z)),
        "coverage": float(len(z) / len(rows)),
        "timestamps": int(len(portfolio)),
        "mean_r": float(r.mean()),
        "profit_factor_r": float(pf),
        "win_rate": float((r > 0).mean()),
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "mean_timestamp_return": float(portfolio.mean()),
    }


def choose_validation_threshold(
    validation: pd.DataFrame,
    score: np.ndarray,
    config: MLRebuildConfig | None = None,
) -> tuple[float, dict]:
    cfg = config or MLRebuildConfig()
    if len(validation) != len(score):
        raise ValueError("validation score length mismatch")
    best_t = float("inf")
    best_metrics = portfolio_economic_metrics(validation, np.zeros(len(validation), dtype=bool), cfg)
    best_obj = -np.inf
    for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95):
        t = float(np.quantile(score, q))
        selected = np.asarray(score) >= t
        m = portfolio_economic_metrics(validation, selected, cfg)
        if m["selected"] < cfg.min_selected_validation:
            continue
        dd = abs(min(0.0, float(m["max_drawdown"])))
        pf = min(float(m["profit_factor_r"]), 5.0) if np.isfinite(m["profit_factor_r"]) else 5.0
        obj = (
            float(m["mean_r"]) * np.sqrt(m["selected"])
            + 0.10 * (pf - 1.0)
            - 1.25 * dd
            - 0.05 * m["coverage"]
        )
        if obj > best_obj:
            best_obj = obj
            best_t = t
            best_metrics = m
    return best_t, {**best_metrics, "validation_objective": float(best_obj)}


def moving_block_ci(values: Iterable[float], resamples: int = 500, block: int = 12, seed: int = 314) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = float(np.mean(x))
    if n < max(20, block * 2):
        return {"n": n, "mean": mean, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block + 1))
    blocks_needed = int(np.ceil(n / block))
    means = np.empty(resamples, dtype=float)
    for i in range(resamples):
        sample = []
        for _ in range(blocks_needed):
            s = int(rng.choice(starts))
            sample.extend(x[s:s + block])
        means[i] = np.mean(sample[:n])
    return {
        "n": n,
        "mean": mean,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def selected_timestamp_returns(rows: pd.DataFrame, selected: np.ndarray, config: MLRebuildConfig | None = None) -> pd.Series:
    cfg = config or MLRebuildConfig()
    mask = np.asarray(selected, dtype=bool)
    z = rows.loc[mask].copy()
    if z.empty:
        return pd.Series(dtype=float)
    z["signal_time"] = pd.to_datetime(z["signal_time"], utc=True)
    z["_event_risk_return"] = cfg.base_risk_per_trade * z["label_r_multiple"].astype(float)

    def aggregate(g: pd.DataFrame) -> float:
        raw = g["_event_risk_return"].to_numpy(float)
        gross_risk = cfg.base_risk_per_trade * len(raw)
        scale = min(1.0, cfg.max_risk_per_timestamp / gross_risk) if gross_risk > 0 else 1.0
        return float(raw.sum() * scale)

    return z.groupby("signal_time", sort=True).apply(aggregate, include_groups=False)


def train_classifier_tournament(
    dataset: pd.DataFrame,
    output_dir: Path,
    config: MLRebuildConfig | None = None,
    include_unsupervised: bool = True,
    model_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cfg = config or MLRebuildConfig()
    audit_labeled_dataset(dataset)
    x = dataset.copy()
    unsupervised_bank = None
    if include_unsupervised:
        x, unsupervised_bank = attach_development_only_unsupervised(x, seed=cfg.seed)
    numeric, categorical = safe_feature_columns(x)
    cols = numeric + categorical
    dev = x[x["segment"].eq("development")].sort_values("signal_time").copy()
    val = x[x["segment"].eq("validation")].sort_values("signal_time").copy()
    test = x[x["segment"].eq("test")].sort_values("signal_time").copy()
    zoo = classifier_zoo(cfg.seed)
    if model_names is not None:
        wanted = set(model_names)
        zoo = {k: v for k, v in zoo.items() if k in wanted}
    if not zoo:
        raise ValueError("empty classifier zoo")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    rows, val_preds, test_preds = [], [], []
    slow = {"rbf_svc", "knn", "mlp", "catboost"}

    for name, estimator in zoo.items():
        limit = cfg.max_fit_rows_slow if name in slow else cfg.max_fit_rows
        fit = _sample_fit(dev, limit)
        pipe = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", clone(estimator))])
        try:
            pipe.fit(fit[cols], fit["label_profitable_net"].astype(int))
            val_score = _score_vector(pipe, val[cols])
            threshold, econ_val = choose_validation_threshold(val, val_score, cfg)
            val_class = classification_metrics(val["label_profitable_net"], val_score)

            test_score = _score_vector(pipe, test[cols])
            test_mask = test_score >= threshold if np.isfinite(threshold) else np.zeros(len(test), bool)
            econ_test = portfolio_economic_metrics(test, test_mask, cfg)
            test_class = classification_metrics(test["label_profitable_net"], test_score)
            test_ts = selected_timestamp_returns(test, test_mask, cfg)
            boot = moving_block_ci(
                test_ts.values,
                resamples=cfg.bootstrap_resamples,
                block=cfg.bootstrap_block,
                seed=cfg.seed,
            )
            joblib.dump(pipe, models_dir / f"classifier_{name}.joblib", compress=3)
            rows.append({
                "model": name,
                "status": "ok",
                "fit_rows": int(len(fit)),
                "threshold": float(threshold),
                **{f"validation_{k}": v for k, v in val_class.items()},
                **{f"validation_{k}": v for k, v in econ_val.items()},
                **{f"test_{k}": v for k, v in test_class.items()},
                **{f"test_{k}": v for k, v in econ_test.items()},
                "test_bootstrap_mean": boot["mean"],
                "test_bootstrap_ci_low": boot["ci_low"],
                "test_bootstrap_ci_high": boot["ci_high"],
            })
            vp = val[["signal_time", "strategy", "timeframe", "symbol", "label_profitable_net", "label_r_multiple"]].copy()
            vp["model"] = name
            vp["score"] = val_score
            vp["selected"] = val_score >= threshold if np.isfinite(threshold) else False
            val_preds.append(vp)
            tp = test[["signal_time", "strategy", "timeframe", "symbol", "label_profitable_net", "label_r_multiple"]].copy()
            tp["model"] = name
            tp["score"] = test_score
            tp["selected"] = test_mask
            test_preds.append(tp)
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    board = pd.DataFrame(rows)
    if "validation_validation_objective" in board:
        board = board.sort_values(
            ["status", "validation_validation_objective"],
            ascending=[False, False],
            na_position="last",
        )
    metadata = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "unsupervised_enabled": include_unsupervised,
        "unsupervised_fitted_on": "development only" if include_unsupervised else None,
        "test_not_used_for_ranking": True,
    }
    if unsupervised_bank is not None:
        joblib.dump(unsupervised_bank, models_dir / "development_only_unsupervised.joblib", compress=3)
    return (
        board,
        pd.concat(val_preds, ignore_index=True) if val_preds else pd.DataFrame(),
        pd.concat(test_preds, ignore_index=True) if test_preds else pd.DataFrame(),
        metadata,
    )


def train_regressor_tournament(
    dataset: pd.DataFrame,
    output_dir: Path,
    config: MLRebuildConfig | None = None,
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    cfg = config or MLRebuildConfig()
    audit_labeled_dataset(dataset)
    numeric, categorical = safe_feature_columns(dataset)
    cols = numeric + categorical
    dev = dataset[dataset["segment"].eq("development")].sort_values("signal_time").copy()
    val = dataset[dataset["segment"].eq("validation")].sort_values("signal_time").copy()
    test = dataset[dataset["segment"].eq("test")].sort_values("signal_time").copy()
    zoo = regressor_zoo(cfg.seed)
    if model_names is not None:
        wanted = set(model_names)
        zoo = {k: v for k, v in zoo.items() if k in wanted}
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    slow = {"svr", "knn_regressor", "mlp_regressor", "catboost_regressor"}
    rows = []
    for name, estimator in zoo.items():
        fit = _sample_fit(dev, cfg.max_fit_rows_slow if name in slow else cfg.max_fit_rows)
        pipe = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", clone(estimator))])
        try:
            pipe.fit(fit[cols], fit["label_r_multiple"].astype(float))
            pv = np.asarray(pipe.predict(val[cols]), float)
            pt = np.asarray(pipe.predict(test[cols]), float)
            threshold = float(np.quantile(pv, 0.80))
            ev = portfolio_economic_metrics(val, pv >= threshold, cfg)
            et = portfolio_economic_metrics(test, pt >= threshold, cfg)
            rows.append({
                "model": name, "status": "ok", "fit_rows": int(len(fit)),
                "threshold": threshold,
                "validation_rmse": float(np.sqrt(mean_squared_error(val["label_r_multiple"], pv))),
                "validation_mae": float(mean_absolute_error(val["label_r_multiple"], pv)),
                "validation_spearman": float(pd.Series(pv).corr(val["label_r_multiple"].reset_index(drop=True), method="spearman")),
                **{f"validation_{k}": v for k, v in ev.items()},
                "test_rmse": float(np.sqrt(mean_squared_error(test["label_r_multiple"], pt))),
                "test_mae": float(mean_absolute_error(test["label_r_multiple"], pt)),
                "test_spearman": float(pd.Series(pt).corr(test["label_r_multiple"].reset_index(drop=True), method="spearman")),
                **{f"test_{k}": v for k, v in et.items()},
            })
            joblib.dump(pipe, models_dir / f"regressor_{name}.joblib", compress=3)
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    board = pd.DataFrame(rows)
    if "validation_mean_r" in board:
        board = board.sort_values(["status", "validation_mean_r"], ascending=[False, False], na_position="last")
    return board


def choose_classifier_champion(board: pd.DataFrame, config: MLRebuildConfig | None = None) -> dict:
    cfg = config or MLRebuildConfig()
    if board.empty:
        return {
            "decision": "NO_MODEL_PROMOTED", "champion": None,
            "paper_replacement_authorized": False, "live_execution_authorized": False,
        }
    ok = board[board["status"].eq("ok")].copy()
    if ok.empty or "validation_validation_objective" not in ok.columns:
        return {
            "decision": "NO_MODEL_PROMOTED", "champion": None,
            "paper_replacement_authorized": False, "live_execution_authorized": False,
        }
    champ = ok.sort_values("validation_validation_objective", ascending=False).iloc[0]
    ci_low = champ.get("test_bootstrap_ci_low", np.nan)
    ci_ok = (
        (not cfg.require_nonnegative_bootstrap_lower)
        or (pd.notna(ci_low) and float(ci_low) >= 0.0)
    )
    test_pass = bool(
        int(champ.get("test_selected", 0)) >= cfg.min_selected_test
        and float(champ.get("test_mean_r", np.nan)) > cfg.min_test_mean_r
        and float(champ.get("test_profit_factor_r", np.nan)) >= cfg.min_test_profit_factor
        and np.isfinite(float(champ.get("test_max_drawdown", np.nan)))
        and abs(float(champ.get("test_max_drawdown", np.nan))) <= cfg.max_drawdown
        and ci_ok
    )
    return {
        "decision": "FORWARD_PAPER_CANDIDATE" if test_pass else "NO_MODEL_PROMOTED",
        "champion": str(champ["model"]),
        "threshold": float(champ["threshold"]),
        "selection_basis": "validation objective only; test evaluated once after freeze",
        "test_pass": test_pass,
        "test_selected": int(champ.get("test_selected", 0)),
        "test_mean_r": float(champ.get("test_mean_r", np.nan)),
        "test_profit_factor_r": float(champ.get("test_profit_factor_r", np.nan)),
        "test_max_drawdown": float(champ.get("test_max_drawdown", np.nan)),
        "test_bootstrap_ci_low": None if pd.isna(ci_low) else float(ci_low),
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def write_manifest(dataset_path: Path, dataset: pd.DataFrame, output_dir: Path, config: MLRebuildConfig | None = None) -> dict:
    cfg = config or MLRebuildConfig()
    audit = audit_labeled_dataset(dataset)
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    payload = {
        "version": "v0.23-clean-rebuild",
        "dataset_sha256": digest,
        "audit": audit,
        "config": asdict(cfg),
        "labels": list(LABEL_COLUMNS),
        "banned_outcome_columns": sorted(BANNED_OUTCOME_COLUMNS),
        "split_contract": "fit=development; threshold/champion=validation; test read once after freeze",
        "risk_contract": "post-cost R labels; 0.25% event risk; 1% concurrent timestamp risk cap; no double cost subtraction",
        "live_execution_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
