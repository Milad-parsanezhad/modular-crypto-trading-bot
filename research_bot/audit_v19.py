from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import ceil
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import performance_metrics
from .features import FEATURE_COLUMNS, add_features
from .oos_tournament_v10 import TournamentConfig, build_tournament_panel
from .regime import REGIME_COLUMNS
from .robustness_v11 import _classify_regime


@dataclass(frozen=True)
class V19AuditConfig:
    timeframe: str = "4h"
    train_fraction: float = 0.70
    horizon_bars: int = 1
    one_way_cost_bps: float = 12.0
    uncertainty_mad_multiplier: float = 0.25
    ridge_alpha: float = 5.0
    calibration_folds: int = 3
    calibration_min_train: int = 300
    bootstrap_resamples: int = 1000
    bootstrap_block: int = 12
    min_holdout_periods: int = 120
    min_cross_section_assets: int = 6
    top_quantile: float = 0.25
    seed: int = 19042


def stable_frame_fingerprint(df: pd.DataFrame, columns: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")) -> str:
    """Stable SHA-256 over the point-in-time market frame used by an experiment."""
    use = [c for c in columns if c in df.columns]
    if not use:
        raise ValueError("no fingerprint columns available")
    x = df[use].copy()
    if "timestamp" in x:
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise").astype(str)
    raw = x.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    return sha256(raw).hexdigest()


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=float(alpha))),
    ])


def expanding_crossfit_residual_mad(train: pd.DataFrame, features: list[str], cfg: V19AuditConfig) -> dict:
    """Estimate uncertainty from training-only out-of-fold residuals.

    v0.18 estimated MAD on in-sample training residuals. That is optimistic because
    the same observations fit the model and calibrate uncertainty. v0.19 instead
    generates chronological expanding-window predictions entirely inside the
    training segment and pools only those out-of-fold residuals.
    """
    n = len(train)
    min_train = min(max(cfg.calibration_min_train, 60), max(60, n // 2))
    if n <= min_train + max(30, cfg.calibration_folds):
        raise ValueError("training segment too short for expanding cross-fit calibration")
    remaining = n - min_train
    fold_size = max(30, remaining // cfg.calibration_folds)
    residuals: list[float] = []
    fold_rows: list[dict] = []

    for fold in range(cfg.calibration_folds):
        val_start = min_train + fold * fold_size
        val_end = n if fold == cfg.calibration_folds - 1 else min(n, val_start + fold_size)
        if val_end <= val_start:
            continue
        safe_train_end = max(0, val_start - cfg.horizon_bars)
        tr = train.iloc[:safe_train_end]
        va = train.iloc[val_start:val_end]
        if len(tr) < 60 or len(va) < 10:
            continue
        model = _ridge(cfg.ridge_alpha)
        model.fit(tr[features], tr["future_return"].astype(float))
        pred = model.predict(va[features])
        resid = va["future_return"].to_numpy(dtype=float) - pred
        residuals.extend(resid[np.isfinite(resid)].tolist())
        fold_rows.append({
            "fold": fold + 1,
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "validation_start": pd.Timestamp(va["timestamp"].iloc[0]).isoformat(),
            "validation_end": pd.Timestamp(va["timestamp"].iloc[-1]).isoformat(),
        })

    if len(residuals) < 50:
        raise ValueError("insufficient out-of-fold residuals for uncertainty calibration")
    r = np.asarray(residuals, dtype=float)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return {
        "oof_residual_count": int(len(r)),
        "oof_residual_median": med,
        "oof_residual_mad": mad,
        "folds": fold_rows,
    }


def _backtest_long_only(frame: pd.DataFrame, position: pd.Series, cost_bps: float, *, force_terminal_close: bool = True) -> pd.DataFrame:
    """One-bar forward-return backtest with explicit transition and terminal costs."""
    z = frame[["timestamp", "future_return"]].copy().reset_index(drop=True)
    pos = pd.Series(position, index=z.index, dtype=float).fillna(0.0).clip(0.0, 1.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    terminal_close = 0.0
    if force_terminal_close and len(pos) and float(pos.iloc[-1]) != 0.0:
        terminal_close = abs(float(pos.iloc[-1]))
        turnover.iloc[-1] += terminal_close
    cost = turnover * float(cost_bps) / 10_000.0
    z["position"] = pos
    z["turnover"] = turnover
    z["gross_return"] = pos * z["future_return"].astype(float)
    z["cost"] = cost
    z["net_return"] = z["gross_return"] - z["cost"]
    z.attrs["terminal_close_turnover"] = terminal_close
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
        "transition_periods": int((bt["turnover"] > 0).sum()),
        "terminal_close_turnover": float(bt.attrs.get("terminal_close_turnover", 0.0)),
    })
    return m


def moving_block_ci(diff: pd.Series, *, resamples: int, block: int, seed: int) -> dict:
    x = pd.Series(diff).replace([np.inf, -np.inf], np.nan).dropna().astype(float).to_numpy()
    n = len(x)
    if n < max(30, 2 * block):
        return {"n": n, "mean": float(np.mean(x)) if n else None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    b = max(1, min(int(block), n))
    starts = np.arange(n)
    means = np.empty(int(resamples), dtype=float)
    blocks_needed = int(ceil(n / b))
    for i in range(int(resamples)):
        sample: list[float] = []
        for _ in range(blocks_needed):
            s = int(rng.choice(starts))
            sample.extend(x[(s + np.arange(b)) % n].tolist())
        means[i] = float(np.mean(sample[:n]))
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "one_sided_p_positive": float((1.0 + np.sum(means <= 0.0)) / (len(means) + 1.0)),
    }


def audit_cost_aware_conversion(symbol_bars: Mapping[str, pd.DataFrame], config: V19AuditConfig | None = None) -> dict:
    """Audit v0.18 Experiment A without treating the spent holdout as fresh evidence."""
    cfg = config or V19AuditConfig()
    per_symbol: list[dict] = []
    naive_parts: list[pd.DataFrame] = []
    aware_parts: list[pd.DataFrame] = []

    for symbol, bars in sorted(symbol_bars.items()):
        if bars.empty:
            continue
        x = bars.copy().sort_values("timestamp").reset_index(drop=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
        fingerprint = stable_frame_fingerprint(x)
        x = add_features(x)
        x["future_return"] = x["close"].shift(-cfg.horizon_bars) / x["close"] - 1.0
        x = x.dropna(subset=["timestamp", "future_return"]).reset_index(drop=True)
        features = [c for c in FEATURE_COLUMNS if c in x.columns]
        split = int(len(x) * cfg.train_fraction)
        if split < 360 or len(x) - split < cfg.min_holdout_periods:
            continue
        # Purge the labels immediately before the holdout boundary.
        train = x.iloc[: max(1, split - cfg.horizon_bars)].copy().reset_index(drop=True)
        test = x.iloc[split:].copy().reset_index(drop=True)

        calibration = expanding_crossfit_residual_mad(train, features, cfg)
        uncertainty_penalty = cfg.uncertainty_mad_multiplier * float(calibration["oof_residual_mad"])
        round_trip_cost = 2.0 * cfg.one_way_cost_bps / 10_000.0
        hurdle = round_trip_cost + uncertainty_penalty

        final_model = _ridge(cfg.ridge_alpha)
        final_model.fit(train[features], train["future_return"].astype(float))
        pred = final_model.predict(test[features])
        test["predicted_return"] = pred

        naive = _backtest_long_only(test, pd.Series((pred > 0.0).astype(float)), cfg.one_way_cost_bps)
        aware = _backtest_long_only(test, pd.Series((pred > hurdle).astype(float)), cfg.one_way_cost_bps)
        naive["symbol"] = symbol
        aware["symbol"] = symbol
        naive_parts.append(naive)
        aware_parts.append(aware)
        per_symbol.append({
            "symbol": symbol,
            "dataset_fingerprint": fingerprint,
            "coverage_start": pd.Timestamp(bars["timestamp"].iloc[0]).isoformat(),
            "coverage_end": pd.Timestamp(bars["timestamp"].iloc[-1]).isoformat(),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(test)),
            "calibration": calibration,
            "round_trip_cost_hurdle": round_trip_cost,
            "uncertainty_penalty": uncertainty_penalty,
            "total_hurdle": hurdle,
            "naive": _metrics(naive, cfg.timeframe),
            "cost_aware": _metrics(aware, cfg.timeframe),
        })

    if not naive_parts:
        raise ValueError("no symbol had enough data for v0.19 audit")

    def aggregate(parts: list[pd.DataFrame]) -> pd.DataFrame:
        z = pd.concat(parts, ignore_index=True)
        return z.groupby("timestamp", as_index=False).agg(
            gross_return=("gross_return", "mean"),
            net_return=("net_return", "mean"),
            turnover=("turnover", "mean"),
            cost=("cost", "mean"),
            position=("position", "mean"),
        ).sort_values("timestamp").reset_index(drop=True)

    naive_port = aggregate(naive_parts)
    aware_port = aggregate(aware_parts)
    common = naive_port[["timestamp", "net_return"]].merge(
        aware_port[["timestamp", "net_return"]], on="timestamp", suffixes=("_naive", "_aware"), validate="one_to_one"
    )
    inference = moving_block_ci(
        common["net_return_aware"] - common["net_return_naive"],
        resamples=cfg.bootstrap_resamples,
        block=cfg.bootstrap_block,
        seed=cfg.seed,
    )
    return {
        "experiment": "V18A_AUDIT_CORRECTED_REANALYSIS",
        "evidence_status": "SPENT_HOLDOUT_AUDIT_ONLY_NOT_NEW_EVIDENCE",
        "config": asdict(cfg),
        "bugs_corrected": [
            "UNCERTAINTY_MAD_WAS_IN_SAMPLE_NOW_EXPANDING_OOF",
            "TERMINAL_LIQUIDATION_COST_WAS_OMITTED_NOW_CHARGED",
            "DATASET_FINGERPRINT_AND_COVERAGE_WERE_MISSING_NOW_RECORDED",
        ],
        "per_symbol": per_symbol,
        "naive_portfolio": _metrics(naive_port, cfg.timeframe),
        "cost_aware_portfolio": _metrics(aware_port, cfg.timeframe),
        "paired_moving_block_bootstrap": inference,
        "promotion_authorized": False,
        "reason": "v0.18 holdout was already observed; corrected re-analysis is an audit, not fresh promotion evidence",
    }


def _cross_sectional_weights(panel: pd.DataFrame, cfg: V19AuditConfig, favorable_only: bool) -> pd.DataFrame:
    rows: list[dict] = []
    prev: dict[str, float] = {}
    regime_cols = [c for c in REGIME_COLUMNS if c in panel.columns]
    regime_by_ts = panel.groupby("timestamp")[regime_cols].mean().reset_index()
    regime_by_ts["market_regime"] = regime_by_ts.apply(_classify_regime, axis=1)
    regime_map = dict(zip(regime_by_ts["timestamp"], regime_by_ts["market_regime"]))

    for ts, group in panel.groupby("timestamp", sort=True):
        z = group.dropna(subset=["future_return", "ichimoku_score"]).copy()
        if len(z) < cfg.min_cross_section_assets:
            continue
        market_regime = str(regime_map.get(ts, "RANGE"))
        active = (not favorable_only) or market_regime in {"HIGH_VOL", "TREND_DOWN"}
        weights: dict[str, float] = {}
        if active:
            k = max(1, int(ceil(len(z) * cfg.top_quantile)))
            selected = z.nlargest(k, "ichimoku_score")
            w = 1.0 / len(selected)
            weights = {str(s): w for s in selected["symbol"]}
        gross = float(sum(weights.get(str(r.symbol), 0.0) * float(r.future_return) for r in z.itertuples()))
        names = set(prev) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - prev.get(s, 0.0)) for s in names))
        cost = turnover * cfg.one_way_cost_bps / 10_000.0
        rows.append({
            "timestamp": pd.Timestamp(ts),
            "market_regime": market_regime,
            "position": float(sum(abs(v) for v in weights.values())),
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "n_assets": int(len(z)),
            "n_selected": int(len(weights)),
        })
        prev = weights

    out = pd.DataFrame(rows)
    if not out.empty and prev:
        terminal_turnover = float(sum(abs(v) for v in prev.values()))
        out.loc[out.index[-1], "turnover"] += terminal_turnover
        extra = terminal_turnover * cfg.one_way_cost_bps / 10_000.0
        out.loc[out.index[-1], "cost"] += extra
        out.loc[out.index[-1], "net_return"] -= extra
        out.attrs["terminal_close_turnover"] = terminal_turnover
    return out


def audit_v11_regime_replication_construction(symbol_bars: Mapping[str, pd.DataFrame], config: V19AuditConfig | None = None) -> dict:
    """Correct the v0.18-B construction to match the v0.11 cross-sectional baseline.

    v0.18-B used a per-asset binary `score > 0` rule and per-asset regime gate,
    while v0.11's Ichimoku baseline was a cross-sectional top-quartile portfolio
    and its regime label was market-level. This function restores that geometry.
    It remains a retrospective audit unless run on a pre-registered unseen domain.
    """
    cfg = config or V19AuditConfig()
    tcfg = TournamentConfig(
        timeframe=cfg.timeframe,
        horizon_bars=cfg.horizon_bars,
        top_quantile=cfg.top_quantile,
        one_way_cost_bps=cfg.one_way_cost_bps,
        label_hurdle_bps=cfg.one_way_cost_bps,
        min_assets_per_timestamp=cfg.min_cross_section_assets,
        min_train_timestamps=120,
        min_test_timestamps=30,
    )
    panel, _ = build_tournament_panel(symbol_bars, tcfg)
    if panel.empty:
        raise ValueError("empty panel for v0.19 regime audit")
    panel["ichimoku_score"] = (
        panel["ichi_tenkan_kijun"].fillna(0.0)
        + panel["ichi_price_kijun"].fillna(0.0)
        - panel["ichi_cloud_width"].fillna(0.0)
    )
    counts = panel.groupby("timestamp")["symbol"].nunique()
    valid_ts = counts[counts >= cfg.min_cross_section_assets].index
    panel = panel[panel["timestamp"].isin(valid_ts)].copy()
    if panel["symbol"].nunique() < cfg.min_cross_section_assets:
        raise ValueError("insufficient cross-sectional assets")

    plain = _cross_sectional_weights(panel, cfg, favorable_only=False)
    conditioned = _cross_sectional_weights(panel, cfg, favorable_only=True)
    common = plain[["timestamp", "net_return"]].merge(
        conditioned[["timestamp", "net_return"]], on="timestamp", suffixes=("_plain", "_conditioned"), validate="one_to_one"
    )
    inference = moving_block_ci(
        common["net_return_conditioned"] - common["net_return_plain"],
        resamples=cfg.bootstrap_resamples,
        block=cfg.bootstrap_block,
        seed=cfg.seed + 1,
    )
    return {
        "experiment": "V18B_CONSTRUCTION_AUDIT_CORRECTED",
        "evidence_status": "RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION",
        "config": asdict(cfg),
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "panel_rows": int(len(panel)),
        "panel_timestamps": int(panel["timestamp"].nunique()),
        "bugs_corrected": [
            "V18B_BINARY_PER_ASSET_SIGNAL_DID_NOT_MATCH_V11_CROSS_SECTIONAL_TOP_QUARTILE",
            "V18B_PER_ASSET_REGIME_GATE_DID_NOT_MATCH_V11_MARKET_LEVEL_REGIME_DIAGNOSTIC",
            "TERMINAL_LIQUIDATION_COST_WAS_OMITTED_NOW_CHARGED",
        ],
        "frozen_favorable_market_regimes": ["HIGH_VOL", "TREND_DOWN"],
        "plain_cross_sectional_ichimoku": _metrics(plain, cfg.timeframe),
        "market_regime_conditioned_ichimoku": _metrics(conditioned, cfg.timeframe),
        "paired_moving_block_bootstrap": inference,
        "promotion_authorized": False,
        "reason": "this run repairs strategy construction but is not a pre-registered fresh replication domain",
    }


def audit_manifest(cost_audit: dict, regime_audit: dict) -> dict:
    payload = {
        "version": "v0.19",
        "stage": "AUDIT_CORRECTION_AND_FORWARD_FREEZE",
        "cost_audit": cost_audit,
        "regime_audit": regime_audit,
        "live_execution_authorized": False,
        "paper_strategy_replacement_authorized": False,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["manifest_sha256"] = sha256(raw).hexdigest()
    return payload
