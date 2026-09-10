from __future__ import annotations

"""v0.23 strict machine-learning evidence utilities.

Scientific contract
-------------------
* Every feature is available at or before signal close[t].
* Target starts after the signal: fill open[t+1], exit open[t+2].
* Course-only Wyckoff/ICT proxies are excluded from promotable ML features.
* Development -> calibration -> validation -> final_test are chronological per asset,
  with an embargo around boundaries.
* Model/threshold selection happens before final_test is inspected.
* Backtests include explicit trading frictions, ATR/risk-based sizing, gross caps,
  and a causal 5% drawdown kill switch. No result authorizes live trading.
"""

from dataclasses import asdict, dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_bot.multimodal_fusion_v22d import CORE_NUMERIC_FEATURES
from research_bot.scientific_liquidity_wyckoff_v22d import (
    COURSE_HYPOTHESIS_FEATURES,
    SUPPORTED_COMPONENT_FEATURES,
    build_scientific_liquidity_features,
)


@dataclass(frozen=True)
class MLV23Config:
    development_fraction: float = 0.50
    calibration_fraction: float = 0.15
    validation_fraction: float = 0.15
    final_fraction: float = 0.20
    embargo_bars: int = 2
    target_horizon_bars: int = 2
    one_way_cost_bps: float = 12.0
    stress_one_way_cost_bps: tuple[float, ...] = (12.0, 18.0, 30.0)
    risk_per_trade: float = 0.0025
    stop_atr: float = 1.5
    min_stop_pct: float = 0.003
    max_asset_weight: float = 0.35
    max_portfolio_gross: float = 0.70
    max_drawdown: float = 0.05
    min_validation_selected: int = 30
    min_final_selected: int = 40
    min_external_selected: int = 30
    min_final_profit_factor: float = 1.05
    min_final_sharpe: float = 0.0
    conformal_alpha: float = 0.10
    annualization_periods: int = 6 * 365
    seed: int = 314

    @property
    def roundtrip_cost(self) -> float:
        return 2.0 * self.one_way_cost_bps / 10_000.0


TABULAR_FEATURES = tuple(CORE_NUMERIC_FEATURES) + tuple(SUPPORTED_COMPONENT_FEATURES) + (
    "trend_strength_atr",
    "tenkan_kijun_atr",
    "cloud_width_atr",
    "volume_ratio20",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)


def _feature_views(frame: pd.DataFrame) -> pd.DataFrame:
    f = build_scientific_liquidity_features(frame).copy()
    close = f["close"].replace(0, np.nan)
    atr = f["atr"].replace(0, np.nan)
    f["ema20_rel"] = f["ema20"] / close - 1.0
    f["ema50_rel"] = f["ema50"] / close - 1.0
    f["ema200_rel"] = f["ema200"] / close - 1.0
    f["ema200_slope_atr"] = f["ema200_slope"] / atr
    f["kijun_rel"] = f["kijun"] / close - 1.0
    f["cloud_top_rel"] = f["cloud_top"] / close - 1.0
    f["cloud_bottom_rel"] = f["cloud_bottom"] / close - 1.0
    f["trend_strength_atr"] = (f["ema20"] - f["ema200"]) / atr
    f["tenkan_kijun_atr"] = (f["tenkan"] - f["kijun"]) / atr
    f["cloud_width_atr"] = (f["cloud_top"] - f["cloud_bottom"]) / atr
    vol_med = f["volume"].shift(1).rolling(20, min_periods=5).median().replace(0, np.nan)
    f["volume_ratio20"] = f["volume"] / vol_med
    ts = pd.to_datetime(f["timestamp"], utc=True)
    f["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    return f.replace([np.inf, -np.inf], np.nan)


def build_v23_rows(frame: pd.DataFrame, symbol: str, cfg: MLV23Config | None = None, min_history: int = 220) -> pd.DataFrame:
    cfg = cfg or MLV23Config()
    f = _feature_views(frame).reset_index(drop=True)
    if len(f) < min_history + cfg.target_horizon_bars + 10:
        raise ValueError("insufficient history for v0.23")
    opens = pd.to_numeric(f["open"], errors="coerce").to_numpy(float)
    rows = f.loc[min_history - 1: len(f) - cfg.target_horizon_bars - 1, ["timestamp", "atr_pct", *TABULAR_FEATURES]].copy()
    idx = rows.index.to_numpy(int)
    gross = opens[idx + cfg.target_horizon_bars] / opens[idx + 1] - 1.0
    rows["symbol"] = symbol
    rows["source_index"] = idx
    rows["gross_return"] = gross
    rows["target"] = (gross > cfg.roundtrip_cost).astype("int8")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    illegal = set(TABULAR_FEATURES) & set(COURSE_HYPOTHESIS_FEATURES)
    if illegal:
        raise AssertionError(f"course hypotheses leaked into promotable features: {sorted(illegal)}")
    return rows


def assign_segments(rows: pd.DataFrame, cfg: MLV23Config | None = None) -> pd.DataFrame:
    cfg = cfg or MLV23Config()
    if not math.isclose(cfg.development_fraction + cfg.calibration_fraction + cfg.validation_fraction + cfg.final_fraction, 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    out = []
    for symbol, g0 in rows.groupby("symbol", sort=True):
        g = g0.sort_values("timestamp").reset_index(drop=True).copy()
        n = len(g)
        b1 = int(n * cfg.development_fraction)
        b2 = int(n * (cfg.development_fraction + cfg.calibration_fraction))
        b3 = int(n * (cfg.development_fraction + cfg.calibration_fraction + cfg.validation_fraction))
        seg = np.full(n, "embargo", dtype=object)
        seg[:max(0, b1 - cfg.target_horizon_bars)] = "development"
        seg[min(n, b1 + cfg.embargo_bars):max(0, b2 - cfg.target_horizon_bars)] = "calibration"
        seg[min(n, b2 + cfg.embargo_bars):max(0, b3 - cfg.target_horizon_bars)] = "validation"
        seg[min(n, b3 + cfg.embargo_bars):] = "final_test"
        g["segment"] = seg
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(TABULAR_FEATURES)].astype(float)


def _optional_models(seed: int) -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss", random_state=seed, n_jobs=2)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=31, max_depth=-1, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=seed, n_jobs=2, verbosity=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        out["catboost"] = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.035, l2_leaf_reg=4.0, random_seed=seed, verbose=False, allow_writing_files=False)
    except Exception:
        pass
    return out


def tabular_model_zoo(seed: int = 314) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic": LogisticRegression(C=0.5, max_iter=1500, class_weight="balanced", random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=2.0, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", random_state=seed, n_jobs=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, max_depth=14, min_samples_leaf=15, class_weight="balanced", random_state=seed, n_jobs=2),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, learning_rate_init=5e-4, early_stopping=True, validation_fraction=0.15, max_iter=220, random_state=seed),
    }
    models.update(_optional_models(seed))
    return models


def make_tabular_pipeline(estimator: object) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", clone(estimator))])


def raw_score(model: Pipeline, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(x), dtype=float)
        p = p[:, 1] if p.ndim == 2 else p.ravel()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float).ravel()
    p = np.clip(np.asarray(model.predict(x), dtype=float).ravel(), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(score: np.ndarray, y: np.ndarray, seed: int = 314) -> LogisticRegression | None:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    cal.fit(np.asarray(score, dtype=float).reshape(-1, 1), y)
    return cal


def calibrated_probability(calibrator: LogisticRegression | None, score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=float).reshape(-1, 1)
    if calibrator is None:
        return 1.0 / (1.0 + np.exp(-np.clip(s.ravel(), -30, 30)))
    return calibrator.predict_proba(s)[:, 1]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1); total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not mask.any(): continue
        total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def predictive_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = (p >= 0.5).astype(int)
    try: auc = float(roc_auc_score(y, p))
    except Exception: auc = None
    try: ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception: ll = None
    try: brier = float(brier_score_loss(y, p))
    except Exception: brier = None
    return {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None, "auc": auc, "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) else None, "log_loss": ll, "brier": brier, "ece10": expected_calibration_error(y, p) if len(y) else None}


def conformal_binary_quantile(y_cal: np.ndarray, p_cal: np.ndarray, alpha: float = 0.10) -> dict[str, float | int]:
    y = np.asarray(y_cal, dtype=int); p = np.asarray(p_cal, dtype=float)
    p_true = np.where(y == 1, p, 1.0 - p); scores = 1.0 - np.clip(p_true, 0, 1); n = len(scores)
    if n == 0: return {"n": 0, "alpha": float(alpha), "q": 1.0, "singleton_long_threshold": 1.0}
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    try: q = float(np.quantile(scores, level, method="higher"))
    except TypeError: q = float(np.quantile(scores, level, interpolation="higher"))
    threshold = float(max(1.0 - q, np.nextafter(q, 1.0)))
    return {"n": int(n), "alpha": float(alpha), "q": q, "singleton_long_threshold": threshold}


def _max_drawdown(ret: np.ndarray) -> tuple[float, np.ndarray]:
    equity = np.cumprod(1.0 + np.asarray(ret, dtype=float)); peak = np.maximum.accumulate(equity) if len(equity) else np.array([]); dd = equity / peak - 1.0 if len(equity) else np.array([])
    return (float(dd.min()) if len(dd) else 0.0), equity


def portfolio_backtest(rows: pd.DataFrame, probability: np.ndarray, threshold: float, cfg: MLV23Config | None = None, one_way_cost_bps: float | None = None) -> tuple[dict[str, float | int | bool | None], pd.DataFrame]:
    cfg = cfg or MLV23Config(); cost_bps = cfg.one_way_cost_bps if one_way_cost_bps is None else float(one_way_cost_bps)
    z = rows[["timestamp", "symbol", "gross_return", "atr_pct"]].copy().reset_index(drop=True); z["probability"] = np.asarray(probability, dtype=float); z["selected"] = z["probability"] >= float(threshold)
    atr_fill = z["atr_pct"].median(); atr_fill = float(atr_fill) if np.isfinite(atr_fill) else 0.02
    stop_pct = np.maximum(cfg.stop_atr * z["atr_pct"].fillna(atr_fill).clip(lower=0).to_numpy(float), cfg.min_stop_pct)
    raw_weight = np.minimum(cfg.max_asset_weight, cfg.risk_per_trade / stop_pct); conf = np.zeros(len(z), dtype=float)
    if threshold < 1:
        selected_idx = z["selected"].to_numpy(); conf[selected_idx] = 0.5 + 0.5 * np.clip((z.loc[z["selected"], "probability"].to_numpy() - threshold) / (1 - threshold), 0, 1)
    z["weight"] = raw_weight * conf
    gross_by_t = z.groupby("timestamp")["weight"].transform("sum").replace(0, np.nan); scale = np.minimum(1.0, cfg.max_portfolio_gross / gross_by_t).fillna(1.0); z["weight"] *= scale
    rt_cost = 2.0 * cost_bps / 10_000.0; z["net_contribution"] = z["weight"] * (z["gross_return"].astype(float) - rt_cost); z["gross_weight"] = z["weight"].abs()
    periods = z.groupby("timestamp", as_index=False).agg(net_return=("net_contribution", "sum"), gross_exposure=("gross_weight", "sum"), executed_events=("selected", "sum"), candidates=("selected", "size")).sort_values("timestamp").reset_index(drop=True)
    equity = 1.0; peak = 1.0; killed = False; kill_time = None; realized = []
    for _, row in periods.iterrows():
        r = 0.0 if killed else float(row["net_return"]); realized.append(r); equity *= 1.0 + r; peak = max(peak, equity); dd = equity / peak - 1.0
        if (not killed) and dd <= -cfg.max_drawdown: killed = True; kill_time = row["timestamp"]
    periods["realized_net_return"] = realized; ret = periods["realized_net_return"].to_numpy(float); mdd, eq = _max_drawdown(ret)
    mean = float(ret.mean()) if len(ret) else 0.0; std = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0; downside = ret[ret < 0]; dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean / std * math.sqrt(cfg.annualization_periods) if std > 0 else None; sortino = mean / dstd * math.sqrt(cfg.annualization_periods) if dstd > 0 else None
    positive = ret[ret > 0].sum(); negative = -ret[ret < 0].sum(); pf = float(positive / negative) if negative > 0 else (float("inf") if positive > 0 else None); total = float(np.prod(1.0 + ret) - 1.0) if len(ret) else 0.0
    if len(periods):
        cutoff = periods.index[periods["timestamp"] == kill_time][0] if kill_time is not None else periods.index[-1]; executed = int(periods.loc[:cutoff, "executed_events"].sum())
    else: executed = 0
    metrics = {"periods": int(len(periods)), "selected_events_pre_kill": executed, "candidate_events": int(len(z)), "selection_rate_pre_kill": float(executed / len(z)) if len(z) else 0.0, "total_return": total, "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd, "profit_factor": pf, "mean_period_return": mean, "average_gross_exposure": float(periods["gross_exposure"].mean()) if len(periods) else 0.0, "kill_switch_triggered": bool(killed), "kill_time": None if kill_time is None else pd.Timestamp(kill_time).isoformat(), "one_way_cost_bps": cost_bps, "roundtrip_cost_bps": 2.0 * cost_bps}
    periods["equity"] = eq if len(eq) else np.array([], dtype=float)
    return metrics, periods


def choose_validation_threshold(rows: pd.DataFrame, p: np.ndarray, cfg: MLV23Config | None = None) -> tuple[float, dict]:
    cfg = cfg or MLV23Config(); p = np.asarray(p, dtype=float)
    candidates = sorted(set([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80] + [float(x) for x in np.quantile(p, [0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95])]))
    best_t, best_m, best_obj = 1.0, {}, -np.inf
    for t in candidates:
        m, _ = portfolio_backtest(rows, p, t, cfg, cfg.one_way_cost_bps)
        if int(m["selected_events_pre_kill"]) < cfg.min_validation_selected: continue
        sh = float(m["sharpe"] or 0.0); dd = abs(float(m["max_drawdown"])); tr = float(m["total_return"]); obj = float(np.log1p(max(tr, -0.999999)) + 0.05 * sh - 1.5 * dd)
        if obj > best_obj: best_t, best_m, best_obj = float(t), m, obj
    return best_t, {**best_m, "validation_objective": float(best_obj)}


def moving_block_ci(diff: pd.Series | np.ndarray, resamples: int = 1000, block: int = 12, seed: int = 314) -> dict[str, float | int | None]:
    x = pd.Series(diff).dropna().astype(float).to_numpy(); n = len(x)
    if n < max(20, 2 * block): return {"n": n, "mean": float(x.mean()) if n else None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed); starts = np.arange(max(1, n - block + 1)); means = []; need = int(np.ceil(n / block))
    for _ in range(resamples):
        sample = []
        for _ in range(need):
            s = int(rng.choice(starts)); sample.extend(x[s:s + block])
        means.append(float(np.mean(sample[:n])))
    return {"n": n, "mean": float(x.mean()), "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def final_promotion_decision(final_metrics: dict, external_metrics: dict | None, cfg: MLV23Config | None = None) -> dict:
    cfg = cfg or MLV23Config()
    final_pass = bool(int(final_metrics.get("selected_events_pre_kill", 0)) >= cfg.min_final_selected and float(final_metrics.get("total_return", -1.0)) > 0.0 and float(final_metrics.get("profit_factor") or 0.0) >= cfg.min_final_profit_factor and float(final_metrics.get("sharpe") or -99.0) > cfg.min_final_sharpe and abs(float(final_metrics.get("max_drawdown", -1.0))) <= cfg.max_drawdown and not bool(final_metrics.get("kill_switch_triggered", False)))
    external_pass = False
    if external_metrics:
        external_pass = bool(int(external_metrics.get("selected_events_pre_kill", 0)) >= cfg.min_external_selected and float(external_metrics.get("total_return", -1.0)) > 0.0 and abs(float(external_metrics.get("max_drawdown", -1.0))) <= cfg.max_drawdown)
    return {"final_internal_pass": final_pass, "external_venue_pass": external_pass, "decision": "FORWARD_PAPER_ML_CANDIDATE" if (final_pass and external_pass) else "NO_ML_ALPHA_PROMOTION", "paper_replacement_authorized": False, "live_execution_authorized": False, "vision_to_rl_state_connected": False, "next_gate": "prospective immutable forward PAPER evidence; RL remains gated"}


def protocol_dict(cfg: MLV23Config | None = None) -> dict:
    c = cfg or MLV23Config()
    return {"version": "v0.23", "config": asdict(c), "features": list(TABULAR_FEATURES), "excluded_course_hypotheses": list(COURSE_HYPOTHESIS_FEATURES), "causal_target": "signal close[t]; hypothetical fill open[t+1]; exit open[t+2]; positive iff gross > baseline round-trip cost", "split": "per-symbol development -> calibration -> validation -> final_test with horizon purge and embargo", "selection": "all model and threshold selection occurs on validation; final_test is inspected only after champion freeze", "uncertainty": "Platt probability calibration + split-conformal diagnostic; no exchangeability guarantee is claimed for financial time series", "risk": "ATR risk sizing, confidence scaling, 35% asset cap, 70% gross cap, 5% causal drawdown kill switch", "execution": "historical simulation only; no live authorization"}
