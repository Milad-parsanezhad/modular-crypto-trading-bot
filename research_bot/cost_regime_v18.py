from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import performance_metrics
from .features import FEATURE_COLUMNS, add_features
from .regime import add_regime_features


@dataclass(frozen=True)
class V18Config:
    timeframe: str = "4h"
    train_fraction: float = 0.70
    horizon_bars: int = 1
    one_way_cost_bps: float = 12.0
    uncertainty_mad_multiplier: float = 0.25
    ridge_alpha: float = 5.0
    bootstrap_resamples: int = 500
    bootstrap_block: int = 12
    min_holdout_periods: int = 120
    seed: int = 42


def _safe(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _bt_from_position(frame: pd.DataFrame, position: pd.Series, cost_bps: float) -> pd.DataFrame:
    z = frame[["timestamp", "future_return"]].copy().reset_index(drop=True)
    pos = pd.Series(position, index=z.index, dtype=float).fillna(0.0).clip(0.0, 1.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * float(cost_bps) / 10_000.0
    z["position"] = pos
    z["turnover"] = turnover
    z["gross_return"] = pos * z["future_return"].astype(float)
    z["cost"] = cost
    z["net_return"] = z["gross_return"] - z["cost"]
    return z


def _metrics(bt: pd.DataFrame, timeframe: str) -> dict:
    if bt.empty:
        return {"n": 0}
    m = performance_metrics(bt["net_return"], timeframe=timeframe)
    m.update({
        "n": int(len(bt)),
        "gross_total_return": float((1.0 + bt["gross_return"]).prod() - 1.0),
        "net_total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
        "turnover_sum": float(bt["turnover"].sum()),
        "cost_sum": float(bt["cost"].sum()),
        "exposure": float(bt["position"].mean()),
        "trade_transitions": int((bt["turnover"] > 0).sum()),
    })
    return m


def moving_block_ci(diff: pd.Series, *, resamples: int, block: int, seed: int) -> dict:
    x = pd.Series(diff).dropna().astype(float).to_numpy()
    n = len(x)
    if n < max(20, block * 2):
        return {"n": n, "mean": _safe(np.mean(x)) if n else None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block + 1))
    means = np.empty(resamples, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    for i in range(resamples):
        sample = []
        for _ in range(blocks_needed):
            s = int(rng.choice(starts))
            sample.extend(x[s:s + block])
        means[i] = float(np.mean(sample[:n]))
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def run_cost_aware_conversion(symbol_bars: Mapping[str, pd.DataFrame], config: V18Config | None = None) -> dict:
    """Freeze one Ridge forecast per asset, then compare naive sign trading vs cost-aware abstention.

    All threshold ingredients are estimated from the training segment only. The holdout is never used
    to tune the model, uncertainty penalty or execution hurdle.
    """
    cfg = config or V18Config()
    per_symbol = []
    naive_parts = []
    aware_parts = []

    for symbol, bars in sorted(symbol_bars.items()):
        if bars.empty:
            continue
        x = bars.copy().sort_values("timestamp").reset_index(drop=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
        x = add_features(x)
        x["future_return"] = x["close"].shift(-cfg.horizon_bars) / x["close"] - 1.0
        x = x.dropna(subset=["timestamp", "future_return"]).reset_index(drop=True)
        features = [c for c in FEATURE_COLUMNS if c in x.columns]
        split = int(len(x) * cfg.train_fraction)
        if split < 180 or len(x) - split < cfg.min_holdout_periods:
            continue
        train = x.iloc[: max(1, split - cfg.horizon_bars)].copy()
        test = x.iloc[split:].copy().reset_index(drop=True)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=cfg.ridge_alpha)),
        ])
        model.fit(train[features], train["future_return"].astype(float))
        train_pred = model.predict(train[features])
        residual = train["future_return"].to_numpy(dtype=float) - train_pred
        residual_mad = float(np.median(np.abs(residual - np.median(residual))))
        uncertainty_penalty = cfg.uncertainty_mad_multiplier * residual_mad
        round_trip_cost = 2.0 * cfg.one_way_cost_bps / 10_000.0
        hurdle = round_trip_cost + uncertainty_penalty

        pred = model.predict(test[features])
        test["predicted_return"] = pred
        naive_pos = pd.Series((pred > 0.0).astype(float))
        aware_pos = pd.Series((pred > hurdle).astype(float))
        naive = _bt_from_position(test, naive_pos, cfg.one_way_cost_bps)
        aware = _bt_from_position(test, aware_pos, cfg.one_way_cost_bps)
        naive["symbol"] = symbol
        aware["symbol"] = symbol
        naive_parts.append(naive)
        aware_parts.append(aware)
        per_symbol.append({
            "symbol": symbol,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(test)),
            "train_residual_mad": residual_mad,
            "round_trip_cost_hurdle": round_trip_cost,
            "uncertainty_penalty": uncertainty_penalty,
            "total_hurdle": hurdle,
            "naive": _metrics(naive, cfg.timeframe),
            "cost_aware": _metrics(aware, cfg.timeframe),
        })

    if not naive_parts:
        raise ValueError("No symbol had enough data for v0.18 cost-aware evaluation")

    def aggregate(parts: list[pd.DataFrame]) -> pd.DataFrame:
        allx = pd.concat(parts, ignore_index=True)
        out = allx.groupby("timestamp", as_index=False).agg(
            gross_return=("gross_return", "mean"),
            net_return=("net_return", "mean"),
            turnover=("turnover", "mean"),
            cost=("cost", "mean"),
            position=("position", "mean"),
        )
        return out.sort_values("timestamp").reset_index(drop=True)

    naive_port = aggregate(naive_parts)
    aware_port = aggregate(aware_parts)
    common = naive_port[["timestamp", "net_return"]].merge(
        aware_port[["timestamp", "net_return"]], on="timestamp", suffixes=("_naive", "_aware")
    )
    infer = moving_block_ci(
        common["net_return_aware"] - common["net_return_naive"],
        resamples=cfg.bootstrap_resamples,
        block=cfg.bootstrap_block,
        seed=cfg.seed,
    )
    naive_m = _metrics(naive_port, cfg.timeframe)
    aware_m = _metrics(aware_port, cfg.timeframe)
    support = (
        int(aware_m.get("n", 0)) >= cfg.min_holdout_periods
        and (aware_m.get("net_total_return") or 0.0) > 0.0
        and (aware_m.get("turnover_sum") or 0.0) < (naive_m.get("turnover_sum") or 0.0)
        and infer.get("ci_low") is not None
        and float(infer["ci_low"]) > 0.0
    )
    return {
        "experiment": "A_COST_AWARE_ALPHA_CONVERSION",
        "config": asdict(cfg),
        "symbols": [x["symbol"] for x in per_symbol],
        "per_symbol": per_symbol,
        "naive_portfolio": naive_m,
        "cost_aware_portfolio": aware_m,
        "paired_block_bootstrap": infer,
        "decision": "COST_AWARE_CONVERSION_SUPPORTED" if support else "NO_COST_AWARE_CONVERSION_EVIDENCE",
        "claim_scope": "execution-policy comparison only; not live authorization",
    }


def run_external_regime_replication(symbol_bars: Mapping[str, pd.DataFrame], config: V18Config | None = None) -> dict:
    """Replicate the frozen v0.11 Ichimoku-regime hypothesis on an external venue/source.

    Frozen rule: base Ichimoku score > 0; conditioned version is active only when the
    point-in-time regime is HIGH_VOL or TREND_DOWN. No thresholds are tuned on this sample.
    """
    cfg = config or V18Config()
    plain_parts = []
    conditioned_parts = []
    diagnostics = []
    for symbol, bars in sorted(symbol_bars.items()):
        if bars.empty:
            continue
        x = bars.copy().sort_values("timestamp").reset_index(drop=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
        x = add_features(x)
        x = add_regime_features(x)
        x["future_return"] = x["close"].shift(-cfg.horizon_bars) / x["close"] - 1.0
        x = x.dropna(subset=["future_return", "ichi_tenkan_kijun", "ichi_price_kijun", "ichi_cloud_width"]).reset_index(drop=True)
        score = x["ichi_tenkan_kijun"].fillna(0.0) + x["ichi_price_kijun"].fillna(0.0) - x["ichi_cloud_width"].fillna(0.0)
        base = score > 0.0
        favorable = x["regime_high_vol"].eq(1.0) | x["regime_trend_down"].eq(1.0)
        plain = _bt_from_position(x, base.astype(float), cfg.one_way_cost_bps)
        conditioned = _bt_from_position(x, (base & favorable).astype(float), cfg.one_way_cost_bps)
        plain["symbol"] = symbol
        conditioned["symbol"] = symbol
        plain_parts.append(plain)
        conditioned_parts.append(conditioned)
        diagnostics.append({
            "symbol": symbol,
            "rows": int(len(x)),
            "favorable_fraction": float(favorable.mean()),
            "base_signal_fraction": float(base.mean()),
            "plain": _metrics(plain, cfg.timeframe),
            "regime_conditioned": _metrics(conditioned, cfg.timeframe),
        })

    if not plain_parts:
        raise ValueError("No external-venue data available for regime replication")

    def aggregate(parts: list[pd.DataFrame]) -> pd.DataFrame:
        allx = pd.concat(parts, ignore_index=True)
        return allx.groupby("timestamp", as_index=False).agg(
            gross_return=("gross_return", "mean"),
            net_return=("net_return", "mean"),
            turnover=("turnover", "mean"),
            cost=("cost", "mean"),
            position=("position", "mean"),
        ).sort_values("timestamp").reset_index(drop=True)

    plain_port = aggregate(plain_parts)
    cond_port = aggregate(conditioned_parts)
    common = plain_port[["timestamp", "net_return"]].merge(
        cond_port[["timestamp", "net_return"]], on="timestamp", suffixes=("_plain", "_conditioned")
    )
    infer = moving_block_ci(
        common["net_return_conditioned"] - common["net_return_plain"],
        resamples=cfg.bootstrap_resamples,
        block=cfg.bootstrap_block,
        seed=cfg.seed + 1,
    )
    plain_m = _metrics(plain_port, cfg.timeframe)
    cond_m = _metrics(cond_port, cfg.timeframe)
    support = (
        int(cond_m.get("n", 0)) >= cfg.min_holdout_periods
        and (cond_m.get("net_total_return") or 0.0) > 0.0
        and infer.get("ci_low") is not None
        and float(infer["ci_low"]) > 0.0
    )
    return {
        "experiment": "B_EXTERNAL_REGIME_ICHIMOKU_REPLICATION",
        "config": asdict(cfg),
        "symbols": [x["symbol"] for x in diagnostics],
        "per_symbol": diagnostics,
        "plain_ichimoku": plain_m,
        "regime_conditioned_ichimoku": cond_m,
        "paired_block_bootstrap": infer,
        "decision": "EXTERNAL_REGIME_REPLICATION_SUPPORTED" if support else "NO_EXTERNAL_REGIME_REPLICATION_EVIDENCE",
        "frozen_favorable_regimes": ["HIGH_VOL", "TREND_DOWN"],
        "claim_scope": "external replication of a frozen v0.11 hypothesis; not live authorization",
    }
