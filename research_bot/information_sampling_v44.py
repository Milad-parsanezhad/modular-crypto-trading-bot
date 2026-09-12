from __future__ import annotations

"""Causal sampling clocks for preregistered v0.44.

This module changes event *sampling only*.  It does not fit a model, create
future-dependent labels, inspect PnL/outcomes, touch Kraken, or execute orders.

Variants:
- S0 CLOCK_MOTHER_BASELINE
- S1 CUSUM_LAGGED_VOL
- S2 DIRECTIONAL_CHANGE_LAGGED_ATR
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


V44_VARIANTS: tuple[str, ...] = (
    "S0_CLOCK_MOTHER_BASELINE",
    "S1_CUSUM_LAGGED_VOL",
    "S2_DIRECTIONAL_CHANGE_LAGGED_ATR",
)
V44_DEVELOPMENT_VENUES: tuple[str, ...] = ("coinex", "okx", "kucoin")
V44_RESERVED_HOLDOUT = "kraken"
V44_PERTURBATION_SEEDS: tuple[int, ...] = (314, 1618, 2718)


@dataclass(frozen=True)
class InformationSamplingPolicyV44:
    atr_window: int = 14
    cusum_threshold: float = 0.75
    dc_atr_multiplier: float = 1.0
    scale_floor: float = 0.002
    scale_cap: float = 0.08
    max_hold_bars: int = 30
    embargo_bars: int = 30
    purged_folds: int = 5

    def __post_init__(self) -> None:
        if self.atr_window < 2:
            raise ValueError("atr_window must be >=2")
        if self.cusum_threshold <= 0:
            raise ValueError("cusum_threshold must be positive")
        if self.dc_atr_multiplier <= 0:
            raise ValueError("dc_atr_multiplier must be positive")
        if not (0 < self.scale_floor < self.scale_cap < 1):
            raise ValueError("invalid scale bounds")
        if self.max_hold_bars < 2 or self.embargo_bars < 1 or self.purged_folds < 2:
            raise ValueError("invalid temporal validation settings")


def preregistration_manifest_v44() -> dict:
    p = InformationSamplingPolicyV44()
    return {
        "version": "v0.44",
        "experiment": "INFORMATION_DRIVEN_INTRINSIC_TIME_SAMPLING_ABLATION",
        "variants_in_frozen_order": list(V44_VARIANTS),
        "development_venues": list(V44_DEVELOPMENT_VENUES),
        "reserved_holdout": V44_RESERVED_HOLDOUT,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "source_strategy": "v0.39 mother strategy with v0.43 event-family correction",
        "source_target": "v0.41 competing-risk semantics",
        "source_learner": "v0.43 asset-specific cross-venue HistGB cause-specific learner",
        "sampling_policy": asdict(p),
        "cusum_scale": "ATR14(t-1)/close(t-1)",
        "directional_change_threshold": "1.0 * clipped ATR14(t-1)/close(t-1)",
        "true_tick_volume_dollar_bars_claimed": False,
        "threshold_relaxation": False,
        "outcome_based_variant_tuning": False,
        "outcome_based_asset_pruning": False,
        "transformer_or_rl_added": False,
    }


def _validate_sampling_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "mother_event_v39"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing sampling columns: {sorted(missing)}")

    # Fail closed if callers accidentally pass target/economic columns into this
    # layer.  The sampler must not have access to future audit outcomes.
    forbidden = {
        "outcome", "gross_r", "net_r", "stress_net_r", "exit_time",
        "expected_r", "lower_expected_r", "profit_factor", "candidate_pass",
    }
    leaked = sorted(forbidden & set(frame.columns))
    if leaked:
        raise ValueError(f"sampling layer received forbidden outcome/economic columns: {leaked}")

    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="raise")
    x["mother_event_v39"] = pd.to_numeric(x["mother_event_v39"], errors="raise")
    x = x.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if x["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps are not permitted in v0.44 sampling")
    values = x[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite OHLCV")
    if (x[["open", "high", "low", "close"]] <= 0).any().any() or (x["volume"] < 0).any():
        raise ValueError("invalid nonpositive OHLC or negative volume")
    if (x["high"] < x[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("invalid high")
    if (x["low"] > x[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid low")
    return x


def _atr14(frame: pd.DataFrame, window: int) -> pd.Series:
    prev = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev).abs(),
            (frame["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # This ATR is causal.  Sampling thresholds below shift it one more bar so
    # the threshold used at t is fully fixed before observing bar t.
    return tr.ewm(alpha=1.0 / float(window), adjust=False).mean()


def lagged_atr_scale_v44(
    frame: pd.DataFrame,
    policy: InformationSamplingPolicyV44 | None = None,
) -> pd.Series:
    p = policy or InformationSamplingPolicyV44()
    x = _validate_sampling_input(frame)
    atr = _atr14(x, p.atr_window)
    scale = atr.shift(1) / x["close"].shift(1)
    return scale.astype(float)


def cusum_information_events_v44(
    frame: pd.DataFrame,
    policy: InformationSamplingPolicyV44 | None = None,
) -> pd.Series:
    """Two-sided CUSUM using a volatility scale frozen at t-1."""
    p = policy or InformationSamplingPolicyV44()
    x = _validate_sampling_input(frame)
    scale = lagged_atr_scale_v44(x, p)
    log_return = np.log(x["close"]).diff()
    z = log_return / scale.replace(0.0, np.nan)

    pos_sum = 0.0
    neg_sum = 0.0
    events = np.zeros(len(x), dtype=bool)
    for i, value in enumerate(z.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue
        pos_sum = max(0.0, pos_sum + float(value))
        neg_sum = min(0.0, neg_sum + float(value))
        if pos_sum >= p.cusum_threshold or neg_sum <= -p.cusum_threshold:
            events[i] = True
            pos_sum = 0.0
            neg_sum = 0.0
    return pd.Series(events, index=x.index, name="v44_cusum_event")


def directional_change_events_v44(
    frame: pd.DataFrame,
    policy: InformationSamplingPolicyV44 | None = None,
) -> pd.DataFrame:
    """Causal close-path Directional Change with lagged ATR thresholds.

    Event direction is +1 for an upward change after a down mode and -1 for a
    downward change after an up mode.  No overshoot information is used.
    """
    p = policy or InformationSamplingPolicyV44()
    x = _validate_sampling_input(frame)
    raw_scale = lagged_atr_scale_v44(x, p) * p.dc_atr_multiplier
    theta = raw_scale.clip(lower=p.scale_floor, upper=p.scale_cap)
    close = x["close"].to_numpy(dtype=float)
    th = theta.to_numpy(dtype=float)

    event = np.zeros(len(x), dtype=bool)
    direction = np.zeros(len(x), dtype=np.int8)
    mode = 0  # 0 uninitialized, +1 tracking high, -1 tracking low
    high_extreme = float(close[0]) if len(close) else np.nan
    low_extreme = float(close[0]) if len(close) else np.nan

    for i in range(1, len(close)):
        threshold = th[i]
        if not np.isfinite(threshold):
            continue
        c = float(close[i])
        prev = float(close[i - 1])
        if mode == 0:
            mode = 1 if c >= prev else -1
            high_extreme = max(prev, c)
            low_extreme = min(prev, c)
            continue

        if mode > 0:
            high_extreme = max(high_extreme, c)
            drawdown = (high_extreme - c) / max(high_extreme, np.finfo(float).eps)
            if drawdown >= threshold:
                event[i] = True
                direction[i] = -1
                mode = -1
                low_extreme = c
        else:
            low_extreme = min(low_extreme, c)
            drawup = (c - low_extreme) / max(low_extreme, np.finfo(float).eps)
            if drawup >= threshold:
                event[i] = True
                direction[i] = 1
                mode = 1
                high_extreme = c

    return pd.DataFrame(
        {
            "v44_dc_event": event,
            "v44_dc_direction": direction,
            "v44_dc_threshold": theta.to_numpy(dtype=float),
        },
        index=x.index,
    )


def build_sampling_flags_v44(
    frame: pd.DataFrame,
    policy: InformationSamplingPolicyV44 | None = None,
) -> pd.DataFrame:
    """Return all frozen v0.44 sampling flags without reading any outcome."""
    p = policy or InformationSamplingPolicyV44()
    x = _validate_sampling_input(frame)
    cusum = cusum_information_events_v44(x, p)
    dc = directional_change_events_v44(x, p)
    mother = x["mother_event_v39"].eq(1).to_numpy(dtype=bool)

    out = pd.DataFrame(index=x.index)
    out["v44_clock_mother"] = mother
    out["v44_cusum_event"] = cusum.to_numpy(dtype=bool)
    out["v44_dc_event"] = dc["v44_dc_event"].to_numpy(dtype=bool)
    out["v44_dc_direction"] = dc["v44_dc_direction"].to_numpy(dtype=np.int8)
    out["v44_dc_threshold"] = dc["v44_dc_threshold"].to_numpy(dtype=float)
    out["v44_sample_s0"] = mother
    out["v44_sample_s1"] = mother & out["v44_cusum_event"].to_numpy(dtype=bool)
    out["v44_sample_s2"] = mother & out["v44_dc_event"].to_numpy(dtype=bool)
    return out


def variant_column_v44(variant: str) -> str:
    mapping = {
        "S0_CLOCK_MOTHER_BASELINE": "v44_sample_s0",
        "S1_CUSUM_LAGGED_VOL": "v44_sample_s1",
        "S2_DIRECTIONAL_CHANGE_LAGGED_ATR": "v44_sample_s2",
    }
    if variant not in mapping:
        raise ValueError(f"unknown frozen v0.44 variant: {variant}")
    return mapping[variant]
