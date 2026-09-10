from __future__ import annotations

"""Causal price/volume/liquidity features for the v0.22d multimodal lab.

Scientific contract
-------------------
This module deliberately separates two evidence tiers:

* SUPPORTED_COMPONENT: measurable market-microstructure/technical-analysis
  quantities with peer-reviewed support at the component level (order clustering,
  support/resistance, order-flow/volume-price information). They are still not
  assumed to be profitable in this project until our own untouched tests pass.
* COURSE_HYPOTHESIS: algorithmic proxies inspired by the supplied Wyckoff and
  liquidity course notes (spring/upthrust/SOS/SOW/equal-high-low logic). These
  names are never treated as ground-truth institutional intent.

Every row t is computable from information available by the close of t. No
future backfill, centered rolling windows, or backward joins are permitted.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import build_features


@dataclass(frozen=True)
class ScientificLiquidityConfig:
    volume_window: int = 20
    compression_window: int = 20
    equal_level_atr_tolerance: float = 0.20
    effort_volume_z: float = 1.0
    low_result_atr: float = 0.35
    sos_volume_z: float = 0.5
    sos_spread_atr: float = 1.0


SUPPORTED_COMPONENT_FEATURES = (
    "distance_last_swing_high_atr",
    "distance_last_swing_low_atr",
    "distance_round_level_atr",
    "distance_prev_day_high_atr",
    "distance_prev_day_low_atr",
    "distance_prev_week_high_atr",
    "distance_prev_week_low_atr",
    "distance_prev_month_high_atr",
    "distance_prev_month_low_atr",
    "sweep_prev_day_high",
    "sweep_prev_day_low",
    "sweep_prev_week_high",
    "sweep_prev_week_low",
    "sweep_prev_month_high",
    "sweep_prev_month_low",
    "volume_z20",
    "spread_atr",
    "body_atr",
    "signed_volume_proxy_z20",
    "effort_result_stall",
    "range_compression_ratio",
)

COURSE_HYPOTHESIS_FEATURES = (
    "equal_high_proxy",
    "equal_low_proxy",
    "spring_proxy",
    "upthrust_proxy",
    "sos_proxy",
    "sow_proxy",
)

ALL_LIQUIDITY_WYCKOFF_FEATURES = SUPPORTED_COMPONENT_FEATURES + COURSE_HYPOTHESIS_FEATURES


def _completed_period_levels(x: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    """Return previous *completed* period H/L, aligned causally by timestamp."""
    indexed = x.set_index("timestamp")[["high", "low"]]
    agg = indexed.resample(rule, label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
    # Shift the values, not the timestamps: at the start of the current period
    # only the just-completed period is made available.
    agg[["high", "low"]] = agg[["high", "low"]].shift(1)
    agg = agg.rename(columns={"high": f"{prefix}_high", "low": f"{prefix}_low"}).reset_index()
    return agg


def _asof_period_levels(x: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    levels = _completed_period_levels(x, rule, prefix)
    return pd.merge_asof(
        x[["timestamp"]].sort_values("timestamp"),
        levels.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )


def _safe_z(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=max(5, n // 2)).mean()
    std = s.rolling(n, min_periods=max(5, n // 2)).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def build_scientific_liquidity_features(
    frame: pd.DataFrame,
    config: ScientificLiquidityConfig | None = None,
) -> pd.DataFrame:
    """Build causal evidence-tiered liquidity and Wyckoff-inspired proxies."""
    cfg = config or ScientificLiquidityConfig()
    f = build_features(frame).copy()
    if f.empty:
        return f

    atr = f["atr"].replace(0, np.nan)
    spread = (f["high"] - f["low"]).clip(lower=0)
    body = (f["close"] - f["open"]).abs()

    f["distance_last_swing_high_atr"] = (f["last_swing_high"] - f["close"]) / atr
    f["distance_last_swing_low_atr"] = (f["close"] - f["last_swing_low"]) / atr
    f["distance_round_level_atr"] = (f["close"] - f["round_level"]).abs() / atr

    for rule, prefix, label in (("1D", "prev_day", "day"), ("7D", "prev_week", "week"), ("MS", "prev_month", "month")):
        lev = _asof_period_levels(f, rule, prefix)
        f[f"{prefix}_high"] = lev[f"{prefix}_high"].to_numpy()
        f[f"{prefix}_low"] = lev[f"{prefix}_low"].to_numpy()
        f[f"distance_{prefix}_high_atr"] = (f[f"{prefix}_high"] - f["close"]) / atr
        f[f"distance_{prefix}_low_atr"] = (f["close"] - f[f"{prefix}_low"]) / atr
        f[f"sweep_{prefix}_high"] = ((f["high"] > f[f"{prefix}_high"]) & (f["close"] < f[f"{prefix}_high"])).astype(float)
        f[f"sweep_{prefix}_low"] = ((f["low"] < f[f"{prefix}_low"]) & (f["close"] > f[f"{prefix}_low"])).astype(float)

    f["volume_z20"] = _safe_z(f["volume"].astype(float), cfg.volume_window)
    f["spread_atr"] = spread / atr
    f["body_atr"] = body / atr
    # OHLCV-only signed-volume proxy; this is intentionally NOT called OFI.
    location = ((f["close"] - f["open"]) / spread.replace(0, np.nan)).clip(-1, 1).fillna(0.0)
    signed_volume = location * f["volume"].astype(float)
    f["signed_volume_proxy_z20"] = _safe_z(signed_volume, cfg.volume_window)
    f["effort_result_stall"] = (
        (f["volume_z20"] >= cfg.effort_volume_z) & (f["body_atr"] <= cfg.low_result_atr)
    ).astype(float)
    atr_mean = f["atr"].rolling(cfg.compression_window, min_periods=max(5, cfg.compression_window // 2)).mean().replace(0, np.nan)
    f["range_compression_ratio"] = f["atr"] / atr_mean

    # Course-hypothesis layer: algorithmic labels only, never institutional-intent truth.
    prior_high = f["last_swing_high"].shift(1)
    prior_low = f["last_swing_low"].shift(1)
    prev_high = prior_high.shift(1)
    prev_low = prior_low.shift(1)
    f["equal_high_proxy"] = ((prior_high - prev_high).abs() <= cfg.equal_level_atr_tolerance * atr).astype(float)
    f["equal_low_proxy"] = ((prior_low - prev_low).abs() <= cfg.equal_level_atr_tolerance * atr).astype(float)

    # Range boundary excludes the current bar, preserving causality.
    hi20 = f["high"].shift(1).rolling(20, min_periods=20).max()
    lo20 = f["low"].shift(1).rolling(20, min_periods=20).min()
    f["spring_proxy"] = ((f["low"] < lo20) & (f["close"] > lo20)).astype(float)
    f["upthrust_proxy"] = ((f["high"] > hi20) & (f["close"] < hi20)).astype(float)
    f["sos_proxy"] = (
        (f["close"] > hi20)
        & (f["volume_z20"] >= cfg.sos_volume_z)
        & (f["spread_atr"] >= cfg.sos_spread_atr)
    ).astype(float)
    f["sow_proxy"] = (
        (f["close"] < lo20)
        & (f["volume_z20"] >= cfg.sos_volume_z)
        & (f["spread_atr"] >= cfg.sos_spread_atr)
    ).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def feature_evidence_table() -> pd.DataFrame:
    rows = []
    for name in SUPPORTED_COMPONENT_FEATURES:
        rows.append({
            "feature": name,
            "evidence_tier": "SUPPORTED_COMPONENT",
            "interpretation": "component-level evidence only; project alpha remains unproven",
        })
    for name in COURSE_HYPOTHESIS_FEATURES:
        rows.append({
            "feature": name,
            "evidence_tier": "COURSE_HYPOTHESIS",
            "interpretation": "formalized course concept; no claim of market-maker intent or profitability",
        })
    return pd.DataFrame(rows)
