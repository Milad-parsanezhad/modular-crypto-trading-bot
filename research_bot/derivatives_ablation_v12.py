from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import performance_metrics
from .robustness_v11 import benjamini_hochberg, moving_block_bootstrap_paired


VARIANT_FAMILIES = {
    "price_only": ("price",),
    "price_ichimoku": ("price", "ichimoku"),
    "price_derivatives": ("price", "derivatives"),
    "full": ("price", "ichimoku", "derivatives"),
}
MODEL_NAMES = ("logistic", "hgb")


@dataclass(frozen=True)
class V12Config:
    timeframe: str = "8h"
    holdout_fraction: float = 0.25
    development_folds: int = 3
    top_quantile: float = 0.25
    one_way_cost_bps: float = 8.0
    min_assets_per_timestamp: int = 6
    min_development_timestamps: int = 240
    min_holdout_timestamps: int = 90
    seeds: tuple[int, ...] = (11, 42, 101)
    bootstrap_samples: int = 500
    block_length: int = 9
    alpha: float = 0.05
    fdr_alpha: float = 0.10
    max_allowed_drawdown: float = -0.35
    min_regime_periods: int = 25
    random_state: int = 12012


def _safe_float(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _canonical_ts(values) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def _group_rolling(x: pd.DataFrame, col: str, window: int, fn: str, min_periods: int | None = None) -> pd.Series:
    minp = min_periods if min_periods is not None else window
    roll = x.groupby("symbol", group_keys=False)[col].rolling(window, min_periods=minp)
    if fn == "mean":
        out = roll.mean()
    elif fn == "std":
        out = roll.std()
    elif fn == "sum":
        out = roll.sum()
    elif fn == "max":
        out = roll.max()
    elif fn == "min":
        out = roll.min()
    else:
        raise ValueError(fn)
    return out.reset_index(level=0, drop=True)


def add_v12_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Point-in-time 8h cross-sectional features for the v0.12 ablation.

    The input is expected to contain completed USD-M perpetual bars. Daily OI
    inputs must already carry the conservative next-day availability shift.
    No backward filling or future Ichimoku displacement is used.
    """

    required = {"timestamp", "symbol", "open", "high", "low", "futures_close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"v0.12 panel missing columns: {sorted(missing)}")

    x = panel.copy()
    x["timestamp"] = _canonical_ts(x["timestamp"])
    x = x.dropna(subset=["timestamp", "symbol", "futures_close"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for c in (
        "open", "high", "low", "futures_close", "futures_quote_volume",
        "futures_taker_buy_quote", "funding_rate", "premium_index_close",
        "binance_open_interest", "binance_open_interest_value",
    ):
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    g = x.groupby("symbol", group_keys=False)
    close = x["futures_close"]
    x["ret_1"] = g["futures_close"].pct_change()
    for w in (3, 6, 9, 21):
        x[f"mom_{w}"] = g["futures_close"].pct_change(w)
    x["vol_9"] = _group_rolling(x, "ret_1", 9, "std", 6)
    x["vol_21"] = _group_rolling(x, "ret_1", 21, "std", 10)
    x["price_to_ma_21"] = close / _group_rolling(x, "futures_close", 21, "mean", 10) - 1.0
    x["price_to_ma_63"] = close / _group_rolling(x, "futures_close", 63, "mean", 30) - 1.0
    qv = pd.to_numeric(x.get("futures_quote_volume"), errors="coerce")
    x["liq_log_quote"] = np.log1p(qv.clip(lower=0))
    qmu = _group_rolling(x.assign(_qv=qv), "_qv", 21, "mean", 10)
    qsd = _group_rolling(x.assign(_qv=qv), "_qv", 21, "std", 10)
    x["quote_volume_z21"] = (qv - qmu) / qsd.replace(0, np.nan)

    # Ichimoku is used as a feature family only. No chart-forward shift and no Chikou.
    high9 = _group_rolling(x, "high", 9, "max", 9)
    low9 = _group_rolling(x, "low", 9, "min", 9)
    high26 = _group_rolling(x, "high", 26, "max", 26)
    low26 = _group_rolling(x, "low", 26, "min", 26)
    high52 = _group_rolling(x, "high", 52, "max", 52)
    low52 = _group_rolling(x, "low", 52, "min", 52)
    tenkan = (high9 + low9) / 2.0
    kijun = (high26 + low26) / 2.0
    span_a_now = (tenkan + kijun) / 2.0
    span_b_now = (high52 + low52) / 2.0
    cloud_top = pd.concat([span_a_now, span_b_now], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a_now, span_b_now], axis=1).min(axis=1)
    x["ichi_tenkan_kijun"] = (tenkan - kijun) / close
    x["ichi_price_kijun"] = (close - kijun) / close
    x["ichi_cloud_width"] = (cloud_top - cloud_bottom).abs() / close
    above = (close - cloud_top) / close
    below = (close - cloud_bottom) / close
    x["ichi_price_cloud"] = np.where(close > cloud_top, above, np.where(close < cloud_bottom, below, 0.0))

    total = qv
    buy = pd.to_numeric(x.get("futures_taker_buy_quote"), errors="coerce")
    x["deriv_orderflow_imbalance"] = (2.0 * buy - total) / total.replace(0, np.nan)
    x["deriv_orderflow_mean9"] = _group_rolling(x, "deriv_orderflow_imbalance", 9, "mean", 5)
    x["deriv_orderflow_std21"] = _group_rolling(x, "deriv_orderflow_imbalance", 21, "std", 10)

    funding = pd.to_numeric(x.get("funding_rate"), errors="coerce")
    x["deriv_funding"] = funding
    x["deriv_funding_mean9"] = _group_rolling(x.assign(deriv_funding=funding), "deriv_funding", 9, "mean", 4)
    x["deriv_funding_sum21"] = _group_rolling(x.assign(deriv_funding=funding), "deriv_funding", 21, "sum", 8)
    fmu = _group_rolling(x.assign(deriv_funding=funding), "deriv_funding", 21, "mean", 10)
    fsd = _group_rolling(x.assign(deriv_funding=funding), "deriv_funding", 21, "std", 10)
    x["deriv_funding_z21"] = (funding - fmu) / fsd.replace(0, np.nan)

    premium = pd.to_numeric(x.get("premium_index_close"), errors="coerce")
    x["deriv_premium"] = premium
    pmu = _group_rolling(x.assign(deriv_premium=premium), "deriv_premium", 21, "mean", 10)
    psd = _group_rolling(x.assign(deriv_premium=premium), "deriv_premium", 21, "std", 10)
    x["deriv_premium_z21"] = (premium - pmu) / psd.replace(0, np.nan)
    x["deriv_premium_change1"] = x.groupby("symbol")["deriv_premium"].diff()

    oi = pd.to_numeric(x.get("binance_open_interest"), errors="coerce")
    oiv = pd.to_numeric(x.get("binance_open_interest_value"), errors="coerce")
    x["deriv_oi_log"] = np.log(oi.where(oi > 0))
    x["deriv_oi_value_log"] = np.log(oiv.where(oiv > 0))
    x["deriv_oi_change1"] = x.groupby("symbol")["deriv_oi_log"].diff()
    x["deriv_oi_value_change1"] = x.groupby("symbol")["deriv_oi_value_log"].diff()
    omu = _group_rolling(x, "deriv_oi_log", 21, "mean", 8)
    osd = _group_rolling(x, "deriv_oi_log", 21, "std", 8)
    x["deriv_oi_z21"] = (x["deriv_oi_log"] - omu) / osd.replace(0, np.nan)

    price_base = [
        "mom_3", "mom_6", "mom_9", "mom_21", "vol_9", "vol_21",
        "price_to_ma_21", "price_to_ma_63", "liq_log_quote", "quote_volume_z21",
    ]
    ichi_base = ["ichi_tenkan_kijun", "ichi_price_kijun", "ichi_cloud_width", "ichi_price_cloud"]
    deriv_base = [
        "deriv_orderflow_imbalance", "deriv_orderflow_mean9", "deriv_orderflow_std21",
        "deriv_funding", "deriv_funding_mean9", "deriv_funding_sum21", "deriv_funding_z21",
        "deriv_premium", "deriv_premium_z21", "deriv_premium_change1",
        "deriv_oi_log", "deriv_oi_value_log", "deriv_oi_change1", "deriv_oi_value_change1", "deriv_oi_z21",
    ]

    for col in price_base + ichi_base + deriv_base:
        if col not in x:
            continue
        x[f"cs_{col}_pct"] = x.groupby("timestamp")[col].rank(pct=True, method="average")

    price = price_base + [f"cs_{c}_pct" for c in price_base]
    ichimoku = ichi_base + [f"cs_{c}_pct" for c in ichi_base]
    derivatives = deriv_base + [f"cs_{c}_pct" for c in deriv_base]

    x["future_return"] = x.groupby("symbol")["futures_close"].shift(-1) / x["futures_close"] - 1.0
    x = x.replace([np.inf, -np.inf], np.nan)
    return x, {"price": price, "ichimoku": ichimoku, "derivatives": derivatives}


def _make_model(name: str, seed: int):
    if name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=1500, random_state=seed)),
        ])
    if name == "hgb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ])
    raise ValueError(name)


def _split_timestamps(panel: pd.DataFrame, cfg: V12Config) -> tuple[np.ndarray, np.ndarray]:
    ts = np.array(sorted(pd.unique(panel["timestamp"])))
    if not 0.10 <= cfg.holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction must be in [0.10, 0.50]")
    hold = max(cfg.min_holdout_timestamps, int(ceil(len(ts) * cfg.holdout_fraction)))
    cut = len(ts) - hold
    if cut < cfg.min_development_timestamps:
        raise ValueError(f"Insufficient development timestamps: {cut}")
    return ts[:cut], ts[cut:]


def _development_folds(dev_ts: np.ndarray, cfg: V12Config):
    start = max(120, int(len(dev_ts) * 0.55))
    remaining = len(dev_ts) - start
    step = max(30, remaining // max(1, cfg.development_folds))
    for i in range(cfg.development_folds):
        s = start + i * step
        e = len(dev_ts) if i == cfg.development_folds - 1 else min(len(dev_ts), s + step)
        test = dev_ts[s:e]
        train = dev_ts[:s]
        if len(train) <= 1 or len(test) < 20:
            continue
        yield i + 1, train[:-1], test


def _portfolio_from_scores(frame: pd.DataFrame, score_col: str, cfg: V12Config) -> pd.DataFrame:
    rows: list[dict] = []
    prev: dict[str, float] = {}
    for ts, group in frame.groupby("timestamp", sort=True):
        z = group.dropna(subset=[score_col, "future_return"]).copy()
        if len(z) < cfg.min_assets_per_timestamp:
            continue
        k = max(1, int(ceil(len(z) * cfg.top_quantile)))
        selected = z.nlargest(k, score_col)
        w = 1.0 / len(selected)
        weights = {str(s): w for s in selected["symbol"]}
        gross = float((selected["future_return"].astype(float) * w).sum())
        names = set(prev) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - prev.get(s, 0.0)) for s in names))
        cost = turnover * cfg.one_way_cost_bps / 10_000.0
        rows.append({
            "timestamp": pd.Timestamp(ts),
            "gross_return": gross,
            "net_return": gross - cost,
            "turnover": turnover,
            "cost": cost,
            "n_available": int(len(z)),
            "n_selected": int(len(selected)),
            "selected": tuple(sorted(weights)),
        })
        prev = weights
    return pd.DataFrame(rows)


def _metrics(bt: pd.DataFrame, timeframe: str) -> dict:
    if bt.empty:
        return {"n": 0}
    m = performance_metrics(bt["net_return"], timeframe=timeframe)
    m.update({
        "gross_total_return": float((1.0 + bt["gross_return"]).prod() - 1.0),
        "net_total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
        "turnover_sum": float(bt["turnover"].sum()),
        "average_turnover": float(bt["turnover"].mean()),
        "explicit_cost_sum": float(bt["cost"].sum()),
        "rebalances": int(len(bt)),
        "average_selected_assets": float(bt["n_selected"].mean()),
    })
    return m


def _market_regime(panel: pd.DataFrame) -> pd.DataFrame:
    px = panel.groupby("timestamp").agg(
        market_ret=("ret_1", "mean"),
        market_mom=("mom_9", "mean"),
        market_vol=("vol_21", "mean"),
    ).sort_index()
    vol_ref = px["market_vol"].rolling(90, min_periods=30).quantile(0.70).shift(1)
    trend_ref = px["market_mom"].abs().rolling(90, min_periods=30).median().shift(1)
    high_vol = px["market_vol"] > vol_ref
    up = px["market_mom"] > trend_ref
    down = px["market_mom"] < -trend_ref
    px["regime"] = np.where(high_vol, "HIGH_VOL", np.where(up, "TREND_UP", np.where(down, "TREND_DOWN", "RANGE")))
    return px.reset_index()[["timestamp", "regime"]]


def _regime_metrics(bt: pd.DataFrame, regime: pd.DataFrame, cfg: V12Config) -> list[dict]:
    if bt.empty:
        return []
    z = bt.merge(regime, on="timestamp", how="left")
    out: list[dict] = []
    for name, g in z.groupby("regime", dropna=False):
        if len(g) < cfg.min_regime_periods:
            out.append({"regime": str(name), "n": int(len(g)), "status": "INSUFFICIENT_PERIODS"})
            continue
        m = performance_metrics(g["net_return"], timeframe=cfg.timeframe)
        out.append({
            "regime": str(name),
            "n": int(len(g)),
            "status": "OK",
            "total_return": _safe_float(m.get("total_return")),
            "sharpe": _safe_float(m.get("sharpe")),
            "max_drawdown": _safe_float(m.get("max_drawdown")),
            "cvar_95_loss": _safe_float(m.get("cvar_95_loss")),
        })
    return out


def _variant_features(families: Mapping[str, list[str]], name: str) -> list[str]:
    cols: list[str] = []
    for fam in VARIANT_FAMILIES[name]:
        cols.extend(families.get(fam, []))
    return list(dict.fromkeys(cols))


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    seed: int,
) -> np.ndarray:
    y = (train["future_return"] > 0.0).astype(int)
    if y.nunique() < 2:
        raise ValueError("Training target has only one class")
    model = _make_model(model_name, seed)
    model.fit(train[features], y)
    return model.predict_proba(test[features])[:, 1]


def run_v12_ablation(panel: pd.DataFrame, config: V12Config | None = None) -> dict:
    cfg = config or V12Config()
    x, families = add_v12_features(panel)
    x = x.dropna(subset=["future_return"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Require each family to be observable somewhere; never silently convert a missing
    # derivatives source into a price-only experiment.
    family_availability = {
        fam: int(x[cols].notna().any(axis=1).sum()) if cols else 0
        for fam, cols in families.items()
    }
    if family_availability["derivatives"] == 0:
        raise ValueError("Derivatives family is DATA_UNAVAILABLE")

    dev_ts, hold_ts = _split_timestamps(x, cfg)
    dev = x[x["timestamp"].isin(dev_ts)].copy()
    hold = x[x["timestamp"].isin(hold_ts)].copy()
    regime = _market_regime(x)

    development_rows: list[pd.DataFrame] = []
    for fold, train_ts, test_ts in _development_folds(dev_ts, cfg):
        train = dev[dev["timestamp"].isin(train_ts)].copy()
        test = dev[dev["timestamp"].isin(test_ts)].copy()
        for model_name in MODEL_NAMES:
            for variant in VARIANT_FAMILIES:
                cols = _variant_features(families, variant)
                for seed in cfg.seeds:
                    score = _fit_predict(train, test, cols, model_name, seed)
                    p = test[["timestamp", "symbol", "future_return"]].copy()
                    p["score"] = score
                    p["fold"] = fold
                    p["seed"] = seed
                    p["model"] = model_name
                    p["variant"] = variant
                    development_rows.append(p)

    development_predictions = pd.concat(development_rows, ignore_index=True) if development_rows else pd.DataFrame()

    # Final holdout is touched only here, after the protocol/feature/model choices are frozen.
    holdout_seed_rows: list[pd.DataFrame] = []
    holdout_summaries: list[dict] = []
    holdout_bt: dict[str, pd.DataFrame] = {}
    seed_stability: dict[str, list[dict]] = {}

    for model_name in MODEL_NAMES:
        for variant in VARIANT_FAMILIES:
            cols = _variant_features(families, variant)
            seed_preds: list[pd.DataFrame] = []
            seed_metrics: list[dict] = []
            for seed in cfg.seeds:
                score = _fit_predict(dev, hold, cols, model_name, seed)
                p = hold[["timestamp", "symbol", "future_return"]].copy()
                p["score"] = score
                p["seed"] = seed
                p["model"] = model_name
                p["variant"] = variant
                seed_preds.append(p)
                holdout_seed_rows.append(p)
                bt = _portfolio_from_scores(p, "score", cfg)
                m = _metrics(bt, cfg.timeframe)
                m["seed"] = seed
                seed_metrics.append(m)

            key = f"{model_name}:{variant}"
            seed_stability[key] = seed_metrics
            stacked = pd.concat(seed_preds, ignore_index=True)
            ensemble = stacked.groupby(["timestamp", "symbol", "future_return"], as_index=False)["score"].mean()
            bt = _portfolio_from_scores(ensemble, "score", cfg)
            holdout_bt[key] = bt
            m = _metrics(bt, cfg.timeframe)
            yy = (hold["future_return"] > 0.0).astype(int)
            merged_score = hold[["timestamp", "symbol"]].merge(
                ensemble[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left"
            )["score"].astype(float).clip(1e-6, 1 - 1e-6)
            valid = np.isfinite(merged_score.to_numpy()) & np.isfinite(yy.to_numpy())
            m.update({
                "model": model_name,
                "variant": variant,
                "seed_count": len(cfg.seeds),
                "auc": float(roc_auc_score(yy.to_numpy()[valid], merged_score.to_numpy()[valid])) if valid.sum() and yy.to_numpy()[valid].min() != yy.to_numpy()[valid].max() else None,
                "brier": float(brier_score_loss(yy.to_numpy()[valid], merged_score.to_numpy()[valid])) if valid.sum() else None,
            })
            holdout_summaries.append(m)

    planned = [
        ("price_ichimoku", "price_only"),
        ("price_derivatives", "price_only"),
        ("full", "price_only"),
        ("full", "price_ichimoku"),
        ("full", "price_derivatives"),
    ]
    pairwise: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        for candidate, baseline in planned:
            ck = f"{model_name}:{candidate}"
            bk = f"{model_name}:{baseline}"
            a = holdout_bt[ck].set_index("timestamp")["net_return"].rename("candidate")
            b = holdout_bt[bk].set_index("timestamp")["net_return"].rename("baseline")
            pair = pd.concat([a, b], axis=1).dropna()
            key = f"{model_name}:{candidate}__vs__{baseline}"
            if pair.empty:
                pairwise[key] = {"status": "NO_COMMON_COVERAGE", "n": 0}
                continue
            inf = moving_block_bootstrap_paired(
                pair["candidate"],
                pair["baseline"],
                samples=cfg.bootstrap_samples,
                block_length=cfg.block_length,
                alpha=cfg.alpha,
                random_state=cfg.random_state + sum(ord(ch) for ch in key),
                timeframe=cfg.timeframe,
            )
            pairwise[key] = inf
            p = _safe_float(inf.get("one_sided_p_mean_edge"))
            if p is not None:
                p_values[key] = p

    q = benjamini_hochberg(p_values)
    for key, value in q.items():
        pairwise[key]["fdr_q_value"] = float(value)
        pairwise[key]["fdr_significant"] = bool(value <= cfg.fdr_alpha)

    regime_diagnostics = {key: _regime_metrics(bt, regime, cfg) for key, bt in holdout_bt.items()}

    # Promotion is deliberately tied to FULL vs price-only on the final holdout.
    promotion = "NO_INCREMENTAL_DERIVATIVES_EVIDENCE"
    reasons: list[str] = []
    best_full: tuple[str, dict] | None = None
    for model_name in MODEL_NAMES:
        row = next(x for x in holdout_summaries if x["model"] == model_name and x["variant"] == "full")
        if best_full is None or (_safe_float(row.get("sharpe")) or -1e9) > (_safe_float(best_full[1].get("sharpe")) or -1e9):
            best_full = (model_name, row)

    if best_full is None:
        reasons.append("MISSING_FULL_MODEL")
        best_model = None
    else:
        best_model, full = best_full
        price = next(x for x in holdout_summaries if x["model"] == best_model and x["variant"] == "price_only")
        inf_key = f"{best_model}:full__vs__price_only"
        inf = pairwise.get(inf_key, {})
        ci = inf.get("mean_return_diff_ci") or [None, None]
        if (full.get("net_total_return") or 0.0) <= 0.0:
            reasons.append("NON_POSITIVE_FULL_NET_RETURN")
        if (_safe_float(full.get("sharpe")) or -1e9) <= (_safe_float(price.get("sharpe")) or -1e9) + 0.10:
            reasons.append("NO_CLEAR_SHARPE_EDGE_OVER_PRICE_ONLY")
        if _safe_float(full.get("max_drawdown")) is None or float(full["max_drawdown"]) < cfg.max_allowed_drawdown:
            reasons.append("DRAWDOWN_TOO_LARGE")
        if ci[0] is None or float(ci[0]) <= 0.0:
            reasons.append("BOOTSTRAP_EDGE_CI_INCLUDES_ZERO")
        if _safe_float(inf.get("fdr_q_value")) is None or float(inf["fdr_q_value"]) > cfg.fdr_alpha:
            reasons.append("NO_FDR_SIGNIFICANT_EDGE_OVER_PRICE_ONLY")
        reg = [r for r in regime_diagnostics[f"{best_model}:full"] if r.get("status") == "OK"]
        positive_regimes = sum(1 for r in reg if (r.get("total_return") or 0.0) > 0.0)
        if positive_regimes < 2:
            reasons.append("INSUFFICIENT_REGIME_STABILITY")
        if not reasons:
            promotion = "PROVISIONAL_DERIVATIVES_EDGE_NEEDS_FORWARD_PAPER_REPLICATION"

    return {
        "research_status": "V12_DERIVATIVES_EXTERNAL_HOLDOUT_NOT_TRADING_SIGNAL",
        "config": asdict(cfg),
        "feature_families": families,
        "feature_family_availability_rows": family_availability,
        "panel_rows": int(len(x)),
        "panel_timestamps": int(x["timestamp"].nunique()),
        "symbols": sorted(x["symbol"].unique().tolist()),
        "coverage_start": x["timestamp"].min().isoformat(),
        "coverage_end": x["timestamp"].max().isoformat(),
        "development_start": pd.Timestamp(dev_ts[0]).isoformat(),
        "development_end": pd.Timestamp(dev_ts[-1]).isoformat(),
        "final_holdout_start": pd.Timestamp(hold_ts[0]).isoformat(),
        "final_holdout_end": pd.Timestamp(hold_ts[-1]).isoformat(),
        "development_predictions_rows": int(len(development_predictions)),
        "holdout_summary": holdout_summaries,
        "seed_stability": seed_stability,
        "pairwise_bootstrap_inference": pairwise,
        "multiple_testing": {
            "method": "BENJAMINI_HOCHBERG_FDR",
            "planned_tests": len(MODEL_NAMES) * len(planned),
            "realized_tests": len(q),
            "fdr_alpha": cfg.fdr_alpha,
        },
        "regime_diagnostics": regime_diagnostics,
        "promotion_status": promotion,
        "provisional_best_full_model": best_model,
        "promotion_reasons": reasons,
        "limitations": [
            "Frozen modern perpetual cohort is an external replication cohort, not historical-universe coverage proof.",
            "Premium index is treated as a basis/crowding proxy, not independently reconstructed spot-perpetual basis.",
            "Kline taker-buy imbalance is coarse order flow rather than L2/L3 microstructure.",
            "Daily OI is conservatively available next UTC day and is lower-frequency than the 8h decision clock.",
            "No v0.12 output authorizes paper, testnet or live execution.",
        ],
    }
