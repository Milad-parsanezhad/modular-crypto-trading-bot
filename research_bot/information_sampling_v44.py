from __future__ import annotations

"""Causal information-driven sampling primitives for preregistered v0.44.

This module is intentionally limited to a volatility-adaptive symmetric CUSUM
activity clock that can be computed from 4h OHLCV. It does not attempt to
reconstruct tick-level dollar/volume bars from aggregate candles.

Research only. No model fitting, order execution, paper trading or holdout access
exists in this module.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CusumSamplingPolicyV44:
    volatility_span_bars: int = 48
    minimum_volatility_bars: int = 48
    threshold_multiplier: float = 1.0
    threshold_floor: float = 1e-8
    label_horizon_bars: int = 30

    def __post_init__(self) -> None:
        if self.volatility_span_bars < 2:
            raise ValueError("volatility_span_bars must be >=2")
        if self.minimum_volatility_bars < 2:
            raise ValueError("minimum_volatility_bars must be >=2")
        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")
        if self.threshold_floor <= 0:
            raise ValueError("threshold_floor must be positive")
        if self.label_horizon_bars < 1:
            raise ValueError("label_horizon_bars must be positive")


def preregistration_manifest_v44() -> dict:
    p = CusumSamplingPolicyV44()
    return {
        "version": "v0.44",
        "experiment": "CAUSAL_CUSUM_INFORMATION_DRIVEN_SAMPLING_ABLATION",
        "sampling_arms": ["V44_CONTROL_V43_EVENTS", "V44_CUSUM_ACTIVITY_EVENTS"],
        "cusum_policy": asdict(p),
        "cusum_direction_usage": "activity_clock_only",
        "trade_direction_source": "research_candidate_side_v39",
        "entry_timing": "next_bar_open",
        "development_venues": ["coinex", "okx", "kucoin"],
        "reserved_holdout": "kraken",
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "threshold_search": False,
        "tick_bar_reconstruction": False,
        "empirical_execution_allowed": False,
        "execution_unlock_requirement": "frozen docs/V43_RESULTS_2026-09-12.md with exact tested commit/artifact",
    }


def causal_cusum_activity_v44(
    frame: pd.DataFrame,
    policy: CusumSamplingPolicyV44 | None = None,
    *,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
) -> pd.DataFrame:
    """Return causal symmetric-CUSUM activity flags for a single series.

    Threshold at bar t uses an EWM standard deviation shifted by one bar, so the
    scale contains returns only through t-1. The current return r_t may update the
    CUSUM accumulator and trigger an event at close t; any trade must enter no
    earlier than t+1 open.
    """
    p = policy or CusumSamplingPolicyV44()
    if timestamp_col not in frame.columns or close_col not in frame.columns:
        raise ValueError("timestamp and close columns are required")

    x = frame[[timestamp_col, close_col]].copy()
    x[timestamp_col] = pd.to_datetime(x[timestamp_col], utc=True, errors="raise")
    x[close_col] = pd.to_numeric(x[close_col], errors="coerce")
    if not np.isfinite(x[close_col].to_numpy(dtype=float)).all() or (x[close_col] <= 0).any():
        raise ValueError("close must be finite and positive")
    if x[timestamp_col].duplicated().any():
        raise ValueError("duplicate timestamps are not allowed")
    x = x.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    log_return = np.log(x[close_col]).diff()
    prior_sigma = (
        log_return.ewm(
            span=int(p.volatility_span_bars),
            min_periods=int(p.minimum_volatility_bars),
            adjust=False,
        )
        .std(bias=False)
        .shift(1)
    )
    threshold = (prior_sigma * float(p.threshold_multiplier)).clip(lower=float(p.threshold_floor))

    trigger = np.zeros(len(x), dtype=np.int8)
    sign = np.zeros(len(x), dtype=np.int8)
    s_pos = 0.0
    s_neg = 0.0
    returns = log_return.to_numpy(dtype=float)
    thresholds = threshold.to_numpy(dtype=float)

    for i in range(len(x)):
        r = returns[i]
        h = thresholds[i]
        if not (np.isfinite(r) and np.isfinite(h) and h > 0.0):
            # Warm-up is not allowed to accumulate hidden pre-threshold state.
            s_pos = 0.0
            s_neg = 0.0
            continue
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos >= h:
            trigger[i] = 1
            sign[i] = 1
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -h:
            trigger[i] = 1
            sign[i] = -1
            s_pos = 0.0
            s_neg = 0.0

    out = pd.DataFrame(
        {
            timestamp_col: x[timestamp_col],
            "log_return_v44": log_return.astype(float),
            "prior_sigma_v44": prior_sigma.astype(float),
            "cusum_threshold_v44": threshold.astype(float),
            "cusum_trigger_v44": trigger,
            "cusum_activity_sign_v44": sign,
        }
    )
    return out


def attach_cusum_activity_v44(
    features: pd.DataFrame,
    policy: CusumSamplingPolicyV44 | None = None,
    *,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
) -> pd.DataFrame:
    """Attach CUSUM activity to one already causally constructed feature series."""
    sampled = causal_cusum_activity_v44(features, policy, timestamp_col=timestamp_col, close_col=close_col)
    out = features.copy().sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    for col in (
        "log_return_v44",
        "prior_sigma_v44",
        "cusum_threshold_v44",
        "cusum_trigger_v44",
        "cusum_activity_sign_v44",
    ):
        out[col] = sampled[col].to_numpy()
    return out


def cusum_candidate_mask_v44(features: pd.DataFrame) -> pd.Series:
    """Novel-arm candidate mask: CUSUM activity + frozen mother-strategy side.

    CUSUM sign is deliberately ignored. It controls *when* an example is sampled,
    while trade direction remains solely the frozen mother-strategy candidate side.
    """
    required = {"cusum_trigger_v44", "research_candidate_side_v39"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"missing v0.44 candidate columns: {sorted(missing)}")
    side = pd.to_numeric(features["research_candidate_side_v39"], errors="coerce").fillna(0).astype(int)
    return features["cusum_trigger_v44"].eq(1) & side.isin((-1, 1))


def reserve_complete_label_horizon_v44(
    frame: pd.DataFrame,
    policy: CusumSamplingPolicyV44 | None = None,
) -> pd.DataFrame:
    """Reserve the final N actual bars before any forward label generation."""
    p = policy or CusumSamplingPolicyV44()
    x = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = int(p.label_horizon_bars)
    if len(x) <= n:
        return x.iloc[0:0].copy()
    return x.iloc[:-n].copy()
