from __future__ import annotations

"""v0.39R canonical reconstruction of the project's mother strategy.

This module intentionally reconstructs the *architecture* of prior research
without promoting it to paper/live trading. It is deterministic, closed-bar,
and designed for ablation and auditability.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .multitimeframe_strategies_v19 import build_features


ComponentState = Literal[
    "FEATURE_ONLY",
    "CANDIDATE_GENERATOR",
    "ACTIVE_IN_RECONSTRUCTION",
    "REJECTED_UNDER_PROTOCOL",
    "SUPERSEDED",
    "UNFORMALIZED",
]


@dataclass(frozen=True)
class CanonicalConfig:
    timeframe: str = "4h"
    # Structure/context thresholds are fixed here for reconstruction, not tuned.
    displacement_atr: float = 0.80
    mitigation_atr: float = 0.50
    brooks_body_fraction: float = 0.55
    brooks_close_location: float = 0.65
    min_score: int = 6
    require_event: bool = True
    # Research-only risk defaults; execution remains disabled.
    risk_per_trade: float = 0.0025
    stop_atr: float = 1.5
    max_asset_weight: float = 0.35
    max_portfolio_gross: float = 0.70
    max_drawdown: float = 0.05
    fee_bps: float = 10.0
    slippage_bps: float = 2.0


CANONICAL_MANIFEST: tuple[dict[str, str], ...] = (
    {"component": "ichimoku_regime", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "v0.17+"},
    {"component": "cusum_event_sampling", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "IRGC-S/v0.17"},
    {"component": "liquidity_sweep", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "ICT/SMC v0.19+"},
    {"component": "bos_choch_proxy", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "ICT/SMC v0.19+"},
    {"component": "fvg_imbalance", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "ICT/SMC v0.19+"},
    {"component": "orderblock_mitigation_proxy", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "ICT/SMC v0.19+"},
    {"component": "premium_discount", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "ICT/SMC v0.19+"},
    {"component": "brooks_bar_confirmation_proxy", "state": "ACTIVE_IN_RECONSTRUCTION", "lineage": "formalized in v0.39R"},
    {"component": "triple_barrier", "state": "CANDIDATE_GENERATOR", "lineage": "IRGC-S/v0.17"},
    {"component": "meta_labeling", "state": "FEATURE_ONLY", "lineage": "v0.17/v0.21+"},
    {"component": "expected_net_r", "state": "FEATURE_ONLY", "lineage": "v0.36"},
    {"component": "conformal_uncertainty", "state": "FEATURE_ONLY", "lineage": "v0.36"},
    {"component": "drawdown_firewall", "state": "FEATURE_ONLY", "lineage": "v0.31+"},
    {"component": "temporal_consensus", "state": "REJECTED_UNDER_PROTOCOL", "lineage": "v0.38"},
    {"component": "kraken_holdout", "state": "UNFORMALIZED", "lineage": "sealed/reserved"},
)


def canonical_manifest() -> pd.DataFrame:
    return pd.DataFrame(CANONICAL_MANIFEST)


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=list(required)).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if x.empty:
        return x
    bad = (x[["open", "high", "low", "close"]] <= 0).any(axis=1)
    bad |= x["high"] < x[["open", "close", "low"]].max(axis=1)
    bad |= x["low"] > x[["open", "close", "high"]].min(axis=1)
    if bad.any():
        raise ValueError(f"invalid OHLC rows: {int(bad.sum())}")
    return x


def _causal_cusum(close: pd.Series, atr_pct: pd.Series, threshold: float = 0.75) -> pd.Series:
    log_return = np.log(close).diff().fillna(0.0)
    normalized = log_return / atr_pct.replace(0, np.nan)
    pos_sum = 0.0
    neg_sum = 0.0
    events = np.zeros(len(close), dtype=bool)
    for i, value in enumerate(normalized.fillna(0.0).to_numpy(float)):
        pos_sum = max(0.0, pos_sum + value)
        neg_sum = min(0.0, neg_sum + value)
        if pos_sum >= threshold or neg_sum <= -threshold:
            events[i] = True
            pos_sum = 0.0
            neg_sum = 0.0
    return pd.Series(events, index=close.index)


def _brooks_confirmation(x: pd.DataFrame, cfg: CanonicalConfig) -> tuple[pd.Series, pd.Series]:
    """Objective Brooks-inspired bar confirmation proxy.

    This does not claim to encode Al Brooks' discretionary method in full.
    It formalizes two auditable bar-by-bar ideas: strong trend bar location and
    failed-breakout/reversal confirmation around the prior bar.
    """
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    body_fraction = body / rng
    close_loc = (x["close"] - x["low"]) / rng

    strong_bull = (
        (x["close"] > x["open"])
        & (body_fraction >= cfg.brooks_body_fraction)
        & (close_loc >= cfg.brooks_close_location)
    )
    strong_bear = (
        (x["close"] < x["open"])
        & (body_fraction >= cfg.brooks_body_fraction)
        & (close_loc <= 1.0 - cfg.brooks_close_location)
    )

    failed_bear_break = (x["low"] < x["low"].shift(1)) & (x["close"] > x["close"].shift(1))
    failed_bull_break = (x["high"] > x["high"].shift(1)) & (x["close"] < x["close"].shift(1))
    return (strong_bull | failed_bear_break).fillna(False), (strong_bear | failed_bull_break).fillna(False)


def build_canonical_features(frame: pd.DataFrame, config: CanonicalConfig | None = None) -> pd.DataFrame:
    cfg = config or CanonicalConfig()
    raw = _validate(frame)
    x = build_features(raw)
    if x.empty:
        return x

    # Causal event layer.
    x["canonical_cusum_event"] = _causal_cusum(x["close"], x["atr_pct"]).astype("int8")

    # Ichimoku regime/context.
    x["canonical_kumo_bull"] = ((x["close"] > x["cloud_top"]) & (x["tenkan"] > x["kijun"])).astype("int8")
    x["canonical_kumo_bear"] = ((x["close"] < x["cloud_bottom"]) & (x["tenkan"] < x["kijun"])).astype("int8")

    # Market-structure transition proxy (closed-bar only).
    structure = pd.Series(np.nan, index=x.index, dtype=float)
    structure.loc[x["bos_up"].fillna(False)] = 1.0
    structure.loc[x["bos_down"].fillna(False)] = -1.0
    previous_structure = structure.ffill().shift(1)
    x["canonical_choch_up"] = (x["bos_up"].fillna(False) & previous_structure.eq(-1)).astype("int8")
    x["canonical_choch_down"] = (x["bos_down"].fillna(False) & previous_structure.eq(1)).astype("int8")

    # Displacement makes the FVG/structure event less permissive.
    signed_body = x["close"] - x["open"]
    x["canonical_bull_displacement"] = (signed_body >= cfg.displacement_atr * x["atr"]).astype("int8")
    x["canonical_bear_displacement"] = (-signed_body >= cfg.displacement_atr * x["atr"]).astype("int8")

    # Mitigation/retest proxies around the causal OB midpoints from v0.19.
    bull_ob = x["bull_ob_mid"]
    bear_ob = x["bear_ob_mid"]
    x["canonical_bull_mitigation"] = (
        bull_ob.notna()
        & (x["low"] <= bull_ob + cfg.mitigation_atr * x["atr"])
        & (x["close"] > bull_ob)
    ).astype("int8")
    x["canonical_bear_mitigation"] = (
        bear_ob.notna()
        & (x["high"] >= bear_ob - cfg.mitigation_atr * x["atr"])
        & (x["close"] < bear_ob)
    ).astype("int8")

    # Dealing-range location inherited from v0.19.
    x["canonical_discount"] = x["bull_retracement"].gt(0.5).astype("int8")
    x["canonical_premium"] = x["bull_retracement"].lt(0.5).astype("int8")

    bull_brooks, bear_brooks = _brooks_confirmation(x, cfg)
    x["canonical_brooks_bull"] = bull_brooks.astype("int8")
    x["canonical_brooks_bear"] = bear_brooks.astype("int8")

    return x.replace([np.inf, -np.inf], np.nan)


def canonical_score(frame: pd.DataFrame, config: CanonicalConfig | None = None) -> pd.DataFrame:
    """Return auditable long/short component scores and {-1,0,+1} decisions."""
    cfg = config or CanonicalConfig()
    x = build_canonical_features(frame, cfg)
    if x.empty:
        return x

    long_components = pd.DataFrame(
        {
            "regime": x["canonical_kumo_bull"].eq(1),
            "sweep": x["sweep_down"].fillna(False),
            "structure": x["bos_up"].fillna(False) | x["canonical_choch_up"].eq(1),
            "displacement_fvg": x["canonical_bull_displacement"].eq(1) & x["bull_fvg"].fillna(False),
            "ob_mitigation": x["canonical_bull_mitigation"].eq(1),
            "location": x["canonical_discount"].eq(1),
            "bar_confirmation": x["canonical_brooks_bull"].eq(1),
        },
        index=x.index,
    )
    short_components = pd.DataFrame(
        {
            "regime": x["canonical_kumo_bear"].eq(1),
            "sweep": x["sweep_up"].fillna(False),
            "structure": x["bos_down"].fillna(False) | x["canonical_choch_down"].eq(1),
            "displacement_fvg": x["canonical_bear_displacement"].eq(1) & x["bear_fvg"].fillna(False),
            "ob_mitigation": x["canonical_bear_mitigation"].eq(1),
            "location": x["canonical_premium"].eq(1),
            "bar_confirmation": x["canonical_brooks_bear"].eq(1),
        },
        index=x.index,
    )

    x["canonical_long_score"] = long_components.sum(axis=1).astype(int)
    x["canonical_short_score"] = short_components.sum(axis=1).astype(int)
    event_ok = x["canonical_cusum_event"].eq(1) if cfg.require_event else pd.Series(True, index=x.index)

    long_ok = event_ok & (x["canonical_long_score"] >= cfg.min_score)
    short_ok = event_ok & (x["canonical_short_score"] >= cfg.min_score)

    direction = np.where(long_ok & ~short_ok, 1, np.where(short_ok & ~long_ok, -1, 0))
    x["canonical_direction"] = direction.astype("int8")
    x["canonical_conflict"] = (long_ok & short_ok).astype("int8")
    return x


def risk_weight_from_stop(features: pd.DataFrame, config: CanonicalConfig | None = None) -> pd.Series:
    """Research-only stop-distance sizing; does not place orders."""
    cfg = config or CanonicalConfig()
    if "atr_pct" not in features:
        raise ValueError("atr_pct is required")
    denom = cfg.stop_atr * features["atr_pct"].replace(0, np.nan)
    return (cfg.risk_per_trade / denom).clip(lower=0.0, upper=cfg.max_asset_weight).fillna(0.0)


def reconstruction_metadata(config: CanonicalConfig | None = None) -> dict:
    cfg = config or CanonicalConfig()
    return {
        "version": "v0.39R",
        "purpose": "canonical_strategy_reconstruction",
        "config": asdict(cfg),
        "live_execution": False,
        "paper_execution": False,
        "kraken_holdout": "SEALED",
        "promotion_status": "NOT_EVALUATED",
    }
