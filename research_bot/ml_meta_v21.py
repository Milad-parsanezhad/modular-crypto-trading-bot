from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from research_bot.multitimeframe_strategies_v19 import StrategySpec, TournamentConfig
from research_bot.multitimeframe_strategies_v20 import (
    RiskPsychologyPolicy,
    apply_candidate_level_policy,
    attach_completed_htf_context,
    generate_direction_v20,
    simulate_v20_trades,
)

RANDOM_SEED = 314
CATEGORICAL_FEATURES = ["strategy", "family", "timeframe", "symbol", "side"]
LABEL_COLUMNS = [
    "label_profitable_net", "label_target_hit", "label_ge_1r", "label_ge_2r",
    "label_outcome_3class", "label_r_multiple",
]
OUTCOME_BANNED = {
    "entry", "exit", "stop", "target", "entry_time", "exit_time", "exit_reason",
    "gross_return", "net_return", "r_multiple", "account_return", "account_return_v20",
    "equity_v20", "executed_v20", "reject_reason_v20", "risk_fraction_v20",
}


@dataclass(frozen=True)
class MLConfig:
    random_seed: int = RANDOM_SEED
    max_fit_rows: int = 60000
    max_fit_rows_slow: int = 12000
    min_selected_validation: int = 200
    base_risk: float = 0.0025
    min_test_selected: int = 100
    min_test_profit_factor: float = 1.05
    max_test_drawdown: float = 0.05


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def causal_snapshot_frame(frame: pd.DataFrame, timeframe: str, peer: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create point-in-time normalized ML features. No outcome information enters this frame."""
    f = attach_completed_htf_context(frame, timeframe, peer=peer).copy()
    atr = f["atr"].replace(0, np.nan)
    out = pd.DataFrame({"signal_time": pd.to_datetime(f["timestamp"], utc=True)})
    out["f_atr_pct"] = f["atr_pct"]
    out["f_ret1"] = f["ret1"]
    out["f_ret12"] = f["ret12"]
    out["f_close_ema20_atr"] = _safe_div(f["close"] - f["ema20"], atr)
    out["f_close_ema50_atr"] = _safe_div(f["close"] - f["ema50"], atr)
    out["f_close_ema200_atr"] = _safe_div(f["close"] - f["ema200"], atr)
    out["f_ema200_slope_atr"] = _safe_div(f["ema200_slope"], atr)
    out["f_tenkan_kijun_atr"] = _safe_div(f["tenkan"] - f["kijun"], atr)
    out["f_close_cloud_top_atr"] = _safe_div(f["close"] - f["cloud_top"], atr)
    out["f_close_cloud_bottom_atr"] = _safe_div(f["close"] - f["cloud_bottom"], atr)
    out["f_swing_high_dist_atr"] = _safe_div(f["last_swing_high"] - f["close"], atr)
    out["f_swing_low_dist_atr"] = _safe_div(f["close"] - f["last_swing_low"], atr)
    out["f_bull_retracement"] = f["bull_retracement"]
    out["f_bear_retracement"] = f["bear_retracement"]
    vol_med = f["volume"].shift(1).rolling(20, min_periods=5).median()
    out["f_volume_ratio20"] = _safe_div(f["volume"], vol_med)
    bool_cols = [
        "bos_up", "bos_down", "sweep_down", "sweep_up", "bull_fvg", "bear_fvg",
        "base_candle", "rbr", "dbr", "dbd", "rbd", "ny_london_window",
        "ny_am_window", "silver_bullet_window", "peer_break_high", "peer_break_low",
    ]
    for c in bool_cols:
        if c in f:
            out[f"f_{c}"] = f[c].fillna(False).astype("int8")
    if "available_time" in f:
        out["f_has_completed_htf"] = f["available_time"].notna().astype("int8")
    htf_atr_proxy = (f.get("htf_atr_pct", pd.Series(np.nan, index=f.index)) * f.get("htf_close", pd.Series(np.nan, index=f.index))).replace(0, np.nan)
    if "htf_close" in f:
        out["f_htf_close_ema50_atr"] = _safe_div(f["htf_close"] - f["htf_ema50"], htf_atr_proxy)
        out["f_htf_close_ema200_atr"] = _safe_div(f["htf_close"] - f["htf_ema200"], htf_atr_proxy)
        out["f_htf_ema200_slope_atr"] = _safe_div(f["htf_ema200_slope"], htf_atr_proxy)
        out["f_htf_bull_retracement"] = f["htf_bull_retracement"]
        out["f_htf_bear_retracement"] = f["htf_bear_retracement"]
        for c in ["htf_sweep_down", "htf_sweep_up", "htf_bos_up", "htf_bos_down"]:
            if c in f:
                out[f"f_{c}"] = f[c].fillna(False).astype("int8")
    t = out["signal_time"]
    out["f_hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24.0)
    out["f_hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24.0)
    out["f_dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7.0)
    out["f_dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)
    return out.replace([np.inf, -np.inf], np.nan)


def build_labeled_candidate_attempts(
    spec: StrategySpec,
    frame: pd.DataFrame,
    symbol: str,
    tournament: TournamentConfig,
    policy: RiskPsychologyPolicy,
    peer: pd.DataFrame | None = None,
) -> pd.DataFrame:
    direction, decision_features = generate_direction_v20(spec, frame, peer=peer)
    ledger = simulate_v20_trades(spec, frame, direction, decision_features, symbol, tournament=tournament, policy=policy)
    if ledger.empty:
        return ledger
    snapshots = causal_snapshot_frame(frame, spec.timeframe, peer=peer)
    x = ledger.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True).astype("datetime64[ns, UTC]")
    snapshots["signal_time"] = pd.to_datetime(snapshots["signal_time"], utc=True).astype("datetime64[ns, UTC]")
    x = x.merge(snapshots, on="signal_time", how="left", validate="many_to_one")
    x["label_profitable_net"] = (x["r_multiple"] > 0).astype("int8")
    x["label_target_hit"] = (x["exit_reason"] == "target").astype("int8")
    x["label_ge_1r"] = (x["r_multiple"] >= 1.0).astype("int8")
    x["label_ge_2r"] = (x["r_multiple"] >= 2.0).astype("int8")
    x["label_outcome_3class"] = np.select([x["r_multiple"] <= 0, x["r_multiple"] >= 1.0], [-1, 1], default=0).astype("int8")
    x["label_r_multiple"] = x["r_multiple"].astype(float)
    return x


def finalize_candidate_risk(attempts: pd.DataFrame, spec: StrategySpec, policy: RiskPsychologyPolicy) -> pd.DataFrame:
    return apply_candidate_level_policy(attempts, spec, policy=policy) if not attempts.empty else attempts


def safe_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = sorted(c for c in df.columns if c.startswith("f_") and pd.api.types.is_numeric_dtype(df[c]))
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    illegal = (set(numeric) | set(categorical)) & OUTCOME_BANNED
    if illegal:
        raise AssertionError(f"future/outcome leakage in feature set: {sorted(illegal)}")
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def _optional_boosters(seed: int) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=2)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2, verbosity=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        out["catboost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, random_seed=seed, verbose=False, allow_writing_files=False)
    except Exception:
        pass
    return out


def classifier_zoo(seed: int = RANDOM_SEED) -> dict[str, object]:
    models: dict[str, object] = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "ridge_classifier": RidgeClassifier(class_weight="balanced"),
        "sgd_logistic": SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced", max_iter=1500, random_state=seed),
        "passive_aggressive": PassiveAggressiveClassifier(C=0.5, class_weight="balanced", max_iter=1500, random_state=seed),
        "linear_svc": LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
        "rbf_svc": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False, random_state=seed),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=2),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", random_state=seed, n_jobs=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_leaf=15, class_weight="balanced", random_state=seed, n_jobs=2),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.04, max_depth=3, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-3, learning_rate_init=5e-4, max_iter=180, early_stopping=True, random_state=seed),
    }
    models.update(_optional_boosters(seed))
    return models


def regressor_zoo(seed: int = RANDOM_SEED) -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(n_jobs=2),
        "ridge": Ridge(alpha=2.0, random_state=seed),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=3000, random_state=seed),
        "huber": HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=500),
        "kernel_ridge": KernelRidge(alpha=1.0, kernel="rbf", gamma=0.02),
        "svr": SVR(C=1.0, epsilon=0.1, kernel="rbf"),
        "knn_regressor": KNeighborsRegressor(n_neighbors=31, weights="distance", n_jobs=2),
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=8, min_samples_leaf=40, random_state=seed),
        "random_forest_regressor": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=20, random_state=seed, n_jobs=2),
        "extra_trees_regressor": ExtraTreesRegressor(n_estimators=300, max_depth=14, min_samples_leaf=15, random_state=seed, n_jobs=2),
        "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.04, max_depth=3, loss="huber", random_state=seed),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(96, 48), alpha=1e-3, learning_rate_init=5e-4, max_iter=180, early_stopping=True, random_state=seed),
    }


def _sample_fit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    # Deterministic, label-agnostic time-spread subsample.
    idx = np.linspace(0, len(df) - 1, limit, dtype=int)
    return df.iloc[idx].copy()


def _score_vector(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(x)
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.asarray(p).ravel()
    if hasattr(pipe, "decision_function"):
        s = np.asarray(pipe.decision_function(x), dtype=float).ravel()
        s = np.clip(s, -30, 30)
        return 1.0 / (1.0 + np.exp(-s))
    return np.asarray(pipe.predict(x), dtype=float).ravel()


def economic_metrics(rows: pd.DataFrame, selected: np.ndarray, base_risk: float = 0.0025) -> dict[str, float | int]:
    if len(rows) == 0 or int(np.sum(selected)) == 0:
        return {"selected": 0, "coverage": 0.0, "mean_r": np.nan, "profit_factor": np.nan, "win_rate": np.nan, "total_return": np.nan, "max_drawdown": np.nan}
    z = rows.loc[np.asarray(selected, dtype=bool)].sort_values(["entry_time", "symbol"]).copy()
    r = z["label_r_multiple"].astype(float)
    acc = base_risk * r
    eq = (1.0 + acc).cumprod()
    dd = eq / eq.cummax() - 1.0
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "selected": int(len(z)), "coverage": float(len(z) / len(rows)), "mean_r": float(r.mean()),
        "profit_factor": pf, "win_rate": float((r > 0).mean()),
        "total_return": float(eq.iloc[-1] - 1.0), "max_drawdown": float(dd.min()),
    }


def choose_threshold(validation: pd.DataFrame, score: np.ndarray, cfg: MLConfig) -> tuple[float, dict]:
    best_t, best_m, best_obj = 1.0, economic_metrics(validation, np.zeros(len(validation), dtype=bool), cfg.base_risk), -np.inf
    for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95):
        t = float(np.quantile(score, q))
        selected = score >= t
        m = economic_metrics(validation, selected, cfg.base_risk)
        if m["selected"] < cfg.min_selected_validation or not np.isfinite(m["mean_r"]):
            continue
        dd = abs(float(m["max_drawdown"])) if np.isfinite(m["max_drawdown"]) else 1.0
        obj = float(m["mean_r"]) * np.sqrt(m["selected"]) + 0.10 * (min(float(m["profit_factor"]), 5.0) - 1.0) - 0.75 * dd
        if obj > best_obj:
            best_t, best_m, best_obj = t, m, obj
    best_m = {**best_m, "validation_economic_score": float(best_obj)}
    return best_t, best_m


def _classification_metrics(y: pd.Series, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    try:
        out["roc_auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["log_loss"] = float(log_loss(y, np.clip(score, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        out["log_loss"] = np.nan
    return out


def train_classifier_zoo(dataset: pd.DataFrame, output_dir: Path, cfg: MLConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or MLConfig()
    numeric, categorical = safe_feature_columns(dataset)
    cols = numeric + categorical
    dev = dataset[dataset["segment"] == "development"].sort_values("signal_time").copy()
    val = dataset[dataset["segment"] == "validation"].sort_values("signal_time").copy()
    test = dataset[dataset["segment"] == "test"].sort_values("signal_time").copy()
    if min(len(dev), len(val), len(test)) == 0:
        raise ValueError("development/validation/test must all contain labeled attempts")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows, val_preds, test_preds = [], [], []
    zoo = classifier_zoo(cfg.random_seed)
    slow = {"rbf_svc", "knn", "mlp", "catboost"}
    for name, estimator in zoo.items():
        limit = cfg.max_fit_rows_slow if name in slow else cfg.max_fit_rows
        fit = _sample_fit(dev, limit)
        pipe = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", clone(estimator))])
        status, err = "ok", ""
        try:
            pipe.fit(fit[cols], fit["label_profitable_net"].astype(int))
            vs = _score_vector(pipe, val[cols])
            threshold, econ_val = choose_threshold(val, vs, cfg)
            # The threshold and champion ranking are frozen before any test score is inspected.
            ts = _score_vector(pipe, test[cols])
            econ_test = economic_metrics(test, ts >= threshold, cfg.base_risk)
            cm_val = _classification_metrics(val["label_profitable_net"].astype(int), vs)
            cm_test = _classification_metrics(test["label_profitable_net"].astype(int), ts)
            joblib.dump(pipe, models_dir / f"classifier_{name}.joblib", compress=3)
            rows.append({"model": name, "status": status, "fit_rows": len(fit), "threshold": threshold, **{f"validation_{k}": v for k, v in cm_val.items()}, **{f"validation_{k}": v for k, v in econ_val.items()}, **{f"test_{k}": v for k, v in cm_test.items()}, **{f"test_{k}": v for k, v in econ_test.items()}})
            vp = val[["signal_time", "strategy", "timeframe", "symbol", "label_profitable_net", "label_r_multiple"]].copy(); vp["model"] = name; vp["score"] = vs; vp["selected"] = vs >= threshold; val_preds.append(vp)
            tp = test[["signal_time", "strategy", "timeframe", "symbol", "label_profitable_net", "label_r_multiple"]].copy(); tp["model"] = name; tp["score"] = ts; tp["selected"] = ts >= threshold; test_preds.append(tp)
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    board = pd.DataFrame(rows)
    if "validation_validation_economic_score" in board:
        board = board.sort_values(["status", "validation_validation_economic_score"], ascending=[False, False], na_position="last")
    return board, (pd.concat(val_preds, ignore_index=True) if val_preds else pd.DataFrame()), (pd.concat(test_preds, ignore_index=True) if test_preds else pd.DataFrame())


def train_regressor_zoo(dataset: pd.DataFrame, output_dir: Path, cfg: MLConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MLConfig()
    numeric, categorical = safe_feature_columns(dataset)
    cols = numeric + categorical
    dev = dataset[dataset["segment"] == "development"].sort_values("signal_time").copy()
    val = dataset[dataset["segment"] == "validation"].sort_values("signal_time").copy()
    test = dataset[dataset["segment"] == "test"].sort_values("signal_time").copy()
    models_dir = output_dir / "models"; models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    slow = {"kernel_ridge", "svr", "knn_regressor", "mlp_regressor"}
    for name, estimator in regressor_zoo(cfg.random_seed).items():
        fit = _sample_fit(dev, cfg.max_fit_rows_slow if name in slow else cfg.max_fit_rows)
        pipe = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", clone(estimator))])
        try:
            pipe.fit(fit[cols], fit["label_r_multiple"].astype(float))
            pv, pt = np.asarray(pipe.predict(val[cols]), float), np.asarray(pipe.predict(test[cols]), float)
            threshold = float(np.quantile(pv, 0.80))
            ev, et = economic_metrics(val, pv >= threshold, cfg.base_risk), economic_metrics(test, pt >= threshold, cfg.base_risk)
            rows.append({
                "model": name, "status": "ok", "fit_rows": len(fit), "threshold": threshold,
                "validation_rmse": float(np.sqrt(mean_squared_error(val["label_r_multiple"], pv))),
                "validation_mae": float(mean_absolute_error(val["label_r_multiple"], pv)),
                "validation_corr": float(pd.Series(pv).corr(val["label_r_multiple"].reset_index(drop=True), method="spearman")),
                **{f"validation_{k}": v for k, v in ev.items()},
                "test_rmse": float(np.sqrt(mean_squared_error(test["label_r_multiple"], pt))),
                "test_mae": float(mean_absolute_error(test["label_r_multiple"], pt)),
                "test_corr": float(pd.Series(pt).corr(test["label_r_multiple"].reset_index(drop=True), method="spearman")),
                **{f"test_{k}": v for k, v in et.items()},
            })
            joblib.dump(pipe, models_dir / f"regressor_{name}.joblib", compress=3)
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows).sort_values(["status", "validation_mean_r"], ascending=[False, False], na_position="last") if rows else pd.DataFrame()


def choose_classifier_champion(board: pd.DataFrame, cfg: MLConfig | None = None) -> dict:
    cfg = cfg or MLConfig()
    ok = board[board["status"] == "ok"].copy() if not board.empty else board
    if ok.empty:
        return {"decision": "NO_ML_MODEL_PROMOTED", "champion": None, "live_execution_authorized": False}
    champ = ok.sort_values("validation_validation_economic_score", ascending=False).iloc[0]
    test_pass = bool(
        int(champ.get("test_selected", 0)) >= cfg.min_test_selected
        and float(champ.get("test_mean_r", np.nan)) > 0
        and float(champ.get("test_profit_factor", np.nan)) >= cfg.min_test_profit_factor
        and np.isfinite(float(champ.get("test_max_drawdown", np.nan)))
        and abs(float(champ.get("test_max_drawdown", np.nan))) <= cfg.max_test_drawdown
    )
    return {
        "decision": "FORWARD_PAPER_META_CANDIDATE" if test_pass else "NO_ML_MODEL_PROMOTED",
        "champion": str(champ["model"]), "threshold": float(champ["threshold"]),
        "selection_basis": "validation economic score only; test used once after selection",
        "test_pass": test_pass, "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def write_dataset_manifest(dataset_path: Path, dataset: pd.DataFrame, output_dir: Path) -> dict:
    numeric, categorical = safe_feature_columns(dataset)
    digest = sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_sha256": digest, "rows": int(len(dataset)), "columns": int(len(dataset.columns)),
        "segment_counts": dataset["segment"].value_counts(dropna=False).to_dict(),
        "label_balance_profitable_net": dataset["label_profitable_net"].value_counts(normalize=True).to_dict(),
        "numeric_features": numeric, "categorical_features": categorical,
        "labels": LABEL_COLUMNS, "banned_outcome_columns": sorted(OUTCOME_BANNED),
        "causal_contract": "All f_* features are computed at signal_time from data available at or before that closed bar; outcomes are labels only.",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
