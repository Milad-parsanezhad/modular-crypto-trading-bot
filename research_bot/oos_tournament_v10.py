from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import performance_metrics
from .features import FEATURE_COLUMNS, add_features
from .regime import REGIME_COLUMNS, add_regime_features


@dataclass(frozen=True)
class TournamentConfig:
    timeframe: str = "4h"
    horizon_bars: int = 1
    n_folds: int = 3
    train_fraction: float = 0.55
    top_quantile: float = 0.25
    one_way_cost_bps: float = 12.0
    label_hurdle_bps: float = 12.0
    min_assets_per_timestamp: int = 5
    min_train_timestamps: int = 180
    min_test_timestamps: int = 30
    random_state: int = 42


CS_BASE = [
    "mom_6",
    "mom_24",
    "vol_24",
    "volume_z_24",
    "price_to_ma_24",
    "ichi_tenkan_kijun",
    "ichi_price_kijun",
    "ichi_cloud_width",
]


def _safe_float(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def build_tournament_panel(
    symbol_bars: Mapping[str, pd.DataFrame],
    config: TournamentConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = config or TournamentConfig()
    frames: list[pd.DataFrame] = []
    for symbol, bars in symbol_bars.items():
        if bars.empty:
            continue
        x = bars.copy().sort_values("timestamp").reset_index(drop=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
        x = add_features(x)
        x = add_regime_features(x)
        x["symbol"] = symbol
        x["future_return"] = x["close"].shift(-cfg.horizon_bars) / x["close"] - 1.0
        x["target_up"] = (x["future_return"] > cfg.label_hurdle_bps / 10_000.0).astype(float)
        frames.append(x)

    if not frames:
        return pd.DataFrame(), []
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["timestamp", "future_return"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    cs_features: list[str] = []
    for col in CS_BASE:
        if col not in panel.columns:
            continue
        name = f"cs_{col}_pct"
        panel[name] = panel.groupby("timestamp")[col].rank(pct=True, method="average")
        cs_features.append(name)

    features = [c for c in FEATURE_COLUMNS + REGIME_COLUMNS if c in panel.columns] + cs_features
    return panel.replace([np.inf, -np.inf], np.nan), features


def _time_folds(panel: pd.DataFrame, cfg: TournamentConfig):
    timestamps = np.array(sorted(pd.unique(panel["timestamp"])))
    if len(timestamps) < cfg.min_train_timestamps + cfg.min_test_timestamps:
        raise ValueError("Insufficient synchronized timestamps for OOS tournament")
    start = max(cfg.min_train_timestamps, int(len(timestamps) * cfg.train_fraction))
    remaining = len(timestamps) - start
    step = max(cfg.min_test_timestamps, remaining // cfg.n_folds)
    folds = []
    for i in range(cfg.n_folds):
        test_start = start + i * step
        test_end = len(timestamps) if i == cfg.n_folds - 1 else min(len(timestamps), test_start + step)
        test_ts = timestamps[test_start:test_end]
        train_ts = timestamps[:test_start]
        if len(test_ts) < cfg.min_test_timestamps or len(train_ts) <= cfg.horizon_bars:
            continue
        safe_train_ts = train_ts[:-cfg.horizon_bars]
        folds.append((i + 1, safe_train_ts, test_ts))
    if not folds:
        raise ValueError("No valid OOS folds after purge")
    return folds


def _make_model(name: str, seed: int):
    if name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.25, max_iter=1500, random_state=seed)),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=220,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=18,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def _portfolio_from_scores(
    frame: pd.DataFrame,
    score_col: str,
    cfg: TournamentConfig,
    *,
    equal_weight: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    prev: dict[str, float] = {}
    for ts, group in frame.groupby("timestamp", sort=True):
        z = group.dropna(subset=["future_return"]).copy()
        if not equal_weight:
            z = z.dropna(subset=[score_col])
        if len(z) < cfg.min_assets_per_timestamp:
            continue
        if equal_weight:
            selected = z
        else:
            k = max(1, int(ceil(len(z) * cfg.top_quantile)))
            selected = z.nlargest(k, score_col)
        weight = 1.0 / len(selected)
        weights = {str(s): weight for s in selected["symbol"]}
        gross = float((selected["future_return"].astype(float) * weight).sum())
        names = set(prev) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - prev.get(s, 0.0)) for s in names))
        cost = turnover * cfg.one_way_cost_bps / 10_000.0
        fold_values = selected["fold"].dropna().unique() if "fold" in selected else []
        rows.append(
            {
                "timestamp": pd.Timestamp(ts),
                "fold": int(fold_values[0]) if len(fold_values) else None,
                "gross_return": gross,
                "net_return": gross - cost,
                "turnover": turnover,
                "cost": cost,
                "n_available": int(len(z)),
                "n_selected": int(len(selected)),
                "selected": tuple(sorted(weights)),
            }
        )
        prev = weights
    return pd.DataFrame(rows)


def _portfolio_metrics(bt: pd.DataFrame, timeframe: str) -> dict:
    if bt.empty:
        return {"n": 0}
    metrics = performance_metrics(bt["net_return"], timeframe=timeframe)
    pos = bt.loc[bt["net_return"] > 0, "net_return"].sum()
    neg = -bt.loc[bt["net_return"] < 0, "net_return"].sum()
    metrics.update(
        {
            "gross_total_return": float((1.0 + bt["gross_return"]).prod() - 1.0),
            "net_total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
            "turnover_sum": float(bt["turnover"].sum()),
            "average_turnover": float(bt["turnover"].mean()),
            "explicit_cost_sum": float(bt["cost"].sum()),
            "profit_factor": float(pos / neg) if neg > 0 else None,
            "positive_period_fraction": float((bt["net_return"] > 0).mean()),
            "rebalances": int(len(bt)),
            "average_selected_assets": float(bt["n_selected"].mean()),
        }
    )
    return metrics


def _fold_metrics(bt: pd.DataFrame, timeframe: str) -> list[dict]:
    out = []
    if bt.empty or "fold" not in bt:
        return out
    for fold, group in bt.groupby("fold"):
        d = _portfolio_metrics(group, timeframe)
        d["fold"] = int(fold)
        out.append(d)
    return out


def run_oos_tournament(
    symbol_bars: Mapping[str, pd.DataFrame],
    config: TournamentConfig | None = None,
) -> dict:
    cfg = config or TournamentConfig()
    panel, features = build_tournament_panel(symbol_bars, cfg)
    if panel.empty:
        raise ValueError("Empty tournament panel")
    folds = _time_folds(panel, cfg)

    prediction_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    learned_names = ("logistic", "hgb", "random_forest")

    for fold, train_ts, test_ts in folds:
        train = panel[panel["timestamp"].isin(train_ts)].copy()
        test = panel[panel["timestamp"].isin(test_ts)].copy()
        test["fold"] = fold
        test_rows.append(test)
        y = train["target_up"].astype(int)
        if y.nunique() < 2:
            continue
        for name in learned_names:
            model = _make_model(name, cfg.random_state)
            model.fit(train[features], y)
            pred = test[["timestamp", "symbol", "future_return", "target_up", "fold"]].copy()
            pred["score"] = model.predict_proba(test[features])[:, 1]
            pred["model"] = name
            prediction_rows.append(pred)

    if not prediction_rows or not test_rows:
        raise ValueError("No OOS predictions were produced")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    test_union = pd.concat(test_rows, ignore_index=True).drop_duplicates(["timestamp", "symbol", "fold"])

    variants: dict[str, pd.DataFrame] = {}
    for name, group in predictions.groupby("model"):
        variants[str(name)] = _portfolio_from_scores(group, "score", cfg)

    baseline = test_union.copy()
    baseline["momentum_score"] = (
        baseline["mom_24"].fillna(0.0)
        + 0.5 * baseline["mom_6"].fillna(0.0)
        - 0.25 * baseline["vol_24"].fillna(baseline["vol_24"].median())
    )
    baseline["ichimoku_score"] = (
        baseline["ichi_tenkan_kijun"].fillna(0.0)
        + baseline["ichi_price_kijun"].fillna(0.0)
        - baseline["ichi_cloud_width"].fillna(0.0)
    )
    variants["momentum_baseline"] = _portfolio_from_scores(baseline, "momentum_score", cfg)
    variants["ichimoku_baseline"] = _portfolio_from_scores(baseline, "ichimoku_score", cfg)
    variants["equal_weight_market"] = _portfolio_from_scores(baseline, "momentum_score", cfg, equal_weight=True)

    summaries: list[dict] = []
    fold_summaries: dict[str, list[dict]] = {}
    for name, bt in variants.items():
        d = _portfolio_metrics(bt, cfg.timeframe)
        d["variant"] = name
        if name in learned_names:
            pred = predictions[predictions["model"].eq(name)]
            yy = pred["target_up"].astype(int)
            pp = pred["score"].astype(float).clip(1e-6, 1 - 1e-6)
            d["auc"] = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else None
            d["brier"] = float(brier_score_loss(yy, pp))
        else:
            d["auc"] = None
            d["brier"] = None
        summaries.append(d)
        fold_summaries[name] = _fold_metrics(bt, cfg.timeframe)

    learned = [x for x in summaries if x["variant"] in learned_names and x.get("n", 0) > 0]
    baselines = [x for x in summaries if x["variant"] not in learned_names and x.get("n", 0) > 0]
    learned.sort(key=lambda x: _safe_float(x.get("sharpe")) if _safe_float(x.get("sharpe")) is not None else -1e9, reverse=True)
    best = learned[0] if learned else None
    best_baseline_sharpe = max(
        [_safe_float(x.get("sharpe")) for x in baselines if _safe_float(x.get("sharpe")) is not None] or [-1e9]
    )

    promotion = "NO_MODEL_PROMOTED"
    promotion_reasons: list[str] = []
    if best is not None:
        positive_folds = sum(1 for x in fold_summaries[best["variant"]] if (x.get("net_total_return") or 0.0) > 0)
        sharpe = _safe_float(best.get("sharpe"))
        mdd = _safe_float(best.get("max_drawdown"))
        if (best.get("net_total_return") or 0.0) <= 0:
            promotion_reasons.append("NON_POSITIVE_NET_RETURN")
        if sharpe is None or sharpe <= best_baseline_sharpe + 0.10:
            promotion_reasons.append("NO_CLEAR_SHARPE_EDGE_OVER_BASELINES")
        if positive_folds < max(2, ceil(len(folds) * 2 / 3)):
            promotion_reasons.append("INSUFFICIENT_FOLD_STABILITY")
        if mdd is None or mdd < -0.35:
            promotion_reasons.append("DRAWDOWN_TOO_LARGE")
        if int(best.get("rebalances", 0)) < 60:
            promotion_reasons.append("INSUFFICIENT_OOS_REBALANCES")
        if not promotion_reasons:
            promotion = "PROVISIONAL_OOS_WINNER_NEEDS_BOOTSTRAP_AND_MULTI_SEED"

    return {
        "research_status": "PURGED_OOS_ALPHA_TOURNAMENT_PILOT_NOT_LIVE_SIGNAL",
        "config": asdict(cfg),
        "symbols": sorted(symbol_bars.keys()),
        "panel_rows": int(len(panel)),
        "panel_timestamps": int(panel["timestamp"].nunique()),
        "coverage_start": panel["timestamp"].min().isoformat(),
        "coverage_end": panel["timestamp"].max().isoformat(),
        "feature_count": len(features),
        "features": features,
        "fold_count": len(folds),
        "summary": summaries,
        "fold_summary": fold_summaries,
        "promotion_status": promotion,
        "provisional_best_model": best["variant"] if best else None,
        "promotion_reasons": promotion_reasons,
        "limitations": [
            "Single random seed in this pilot; multi-seed stability is still required.",
            "No bootstrap/SPA/PBO/DSR promotion test yet.",
            "No point-in-time on-chain/news/fundamental features in this tournament.",
            "Long-only top-quantile portfolio is coverage-matched and cost-aware but is not the final execution policy.",
            "No model output from this module is a BUY/SELL instruction.",
        ],
    }
