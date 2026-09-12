from __future__ import annotations

"""v0.43 asset-specific cross-venue stability primitives.

Research-only helpers.  No order execution exists here.  v0.43 keeps the
v0.39 mother strategy and v0.41 competing-risk semantics frozen while replacing
nominal estimator-seed evidence with deterministic moving-block perturbations of
the *training data* and adding outcome-independent data-quality screening.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


V43_DEVELOPMENT_VENUES: tuple[str, ...] = ("coinex", "okx", "kucoin")
V43_RESERVED_HOLDOUT = "kraken"
V43_PERTURBATION_SEEDS: tuple[int, ...] = (314, 1618, 2718)


@dataclass(frozen=True)
class DataQualityPolicyV43:
    timeframe_hours: int = 4
    minimum_bars: int = 900
    max_missing_bar_fraction: float = 0.02
    max_stale_close_fraction: float = 0.05
    label_horizon_bars: int = 30
    minimum_asset_train_events: int = 600
    minimum_venue_train_events: int = 100
    minimum_training_venues: int = 2
    perturbation_block_events: int = 64
    minimum_perturbation_agreement: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if self.timeframe_hours <= 0:
            raise ValueError("timeframe_hours must be positive")
        if self.minimum_bars < 100:
            raise ValueError("minimum_bars is implausibly small")
        if not (0.0 <= self.max_missing_bar_fraction < 0.5):
            raise ValueError("invalid missing-bar fraction")
        if not (0.0 <= self.max_stale_close_fraction < 0.5):
            raise ValueError("invalid stale-close fraction")
        if self.perturbation_block_events < 2:
            raise ValueError("perturbation block must be >=2")


def preregistration_manifest_v43() -> dict:
    p = DataQualityPolicyV43()
    return {
        "version": "v0.43",
        "experiment": "ASSET_SPECIFIC_CROSSVENUE_TRUE_PERTURBATION_STABILITY",
        "development_venues": list(V43_DEVELOPMENT_VENUES),
        "reserved_holdout": V43_RESERVED_HOLDOUT,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "source_strategy": "v0.39 mother strategy",
        "source_target": "v0.41 discrete-time competing risk",
        "learner": "v0.41 HistGradientBoosting cause-specific, trained per asset across development venues",
        "perturbation_seeds": list(V43_PERTURBATION_SEEDS),
        "perturbation_rule": "moving-block resample of training events; row-wise median prediction; never choose best perturbation",
        "naive_skill_baseline": "training-only empirical hazard with family x side x regime and deterministic backoff",
        "data_quality_policy": asdict(p),
        "threshold_relaxation": False,
        "outcome_based_asset_pruning": False,
        "transformer_or_rl_added": False,
    }


def _ensure_utc_timestamp(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp column required")
    return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")


def data_quality_diagnostics_v43(
    frame: pd.DataFrame,
    policy: DataQualityPolicyV43 | None = None,
) -> dict:
    """Return outcome-independent OHLCV quality diagnostics and acceptance.

    The function never reads returns, labels, expectancy, PF, model output or
    holdout data.  Rejection reasons are deterministic from the supplied OHLCV.
    """
    p = policy or DataQualityPolicyV43()
    required = ("timestamp", "open", "high", "low", "close", "volume")
    missing_columns = [c for c in required if c not in frame.columns]
    if missing_columns:
        return {
            "accepted": False,
            "reason_codes": ["MISSING_REQUIRED_COLUMNS"],
            "missing_columns": missing_columns,
            "bars": int(len(frame)),
        }

    ts = _ensure_utc_timestamp(frame)
    x = frame.copy()
    x["timestamp"] = ts
    duplicate_timestamps = int(x["timestamp"].duplicated(keep=False).sum())
    bad_timestamp = int(x["timestamp"].isna().sum())
    x = x.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    numeric = x.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    nonfinite_rows = int((~finite_mask).sum())
    negative_volume_rows = int((numeric["volume"] < 0).fillna(False).sum())

    o = numeric["open"].to_numpy(dtype=float)
    h = numeric["high"].to_numpy(dtype=float)
    l = numeric["low"].to_numpy(dtype=float)
    c = numeric["close"].to_numpy(dtype=float)
    valid_ohlc = (l <= np.minimum(o, c)) & (np.maximum(o, c) <= h) & (l <= h)
    invalid_ohlc_rows = int((~valid_ohlc & finite_mask).sum())

    unique_valid_ts = x.loc[x["timestamp"].notna(), "timestamp"].drop_duplicates()
    if len(unique_valid_ts) >= 2:
        expected = pd.date_range(
            unique_valid_ts.iloc[0],
            unique_valid_ts.iloc[-1],
            freq=f"{p.timeframe_hours}h",
            tz="UTC",
        )
        present = pd.DatetimeIndex(unique_valid_ts)
        missing_expected_bars = int(len(expected.difference(present)))
        missing_bar_fraction = float(missing_expected_bars / max(1, len(expected)))
    else:
        missing_expected_bars = 0
        missing_bar_fraction = 1.0

    close = numeric["close"]
    stale_close_fraction = (
        float(close.eq(close.shift(1)).iloc[1:].mean()) if len(close) > 1 else 1.0
    )

    reasons: list[str] = []
    if len(x) < p.minimum_bars:
        reasons.append("INSUFFICIENT_BARS")
    if bad_timestamp:
        reasons.append("INVALID_TIMESTAMP")
    if duplicate_timestamps:
        reasons.append("DUPLICATE_TIMESTAMP")
    if nonfinite_rows:
        reasons.append("NONFINITE_OHLCV")
    if negative_volume_rows:
        reasons.append("NEGATIVE_VOLUME")
    if invalid_ohlc_rows:
        reasons.append("INVALID_OHLC_RELATION")
    if missing_bar_fraction > p.max_missing_bar_fraction:
        reasons.append("EXCESSIVE_MISSING_BARS")
    if stale_close_fraction > p.max_stale_close_fraction:
        reasons.append("EXCESSIVE_STALE_CLOSE")

    return {
        "accepted": not reasons,
        "reason_codes": reasons or ["PASS"],
        "bars": int(len(x)),
        "bad_timestamp_rows": bad_timestamp,
        "duplicate_timestamp_rows": duplicate_timestamps,
        "nonfinite_rows": nonfinite_rows,
        "negative_volume_rows": negative_volume_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "missing_expected_bars": missing_expected_bars,
        "missing_bar_fraction": missing_bar_fraction,
        "stale_close_fraction": stale_close_fraction,
        "labelable_rows_after_horizon_reserve": max(0, int(len(x) - p.label_horizon_bars)),
    }


def reserve_unsettled_horizon_v43(
    frame: pd.DataFrame,
    policy: DataQualityPolicyV43 | None = None,
) -> pd.DataFrame:
    """Remove the final horizon bars before label generation."""
    p = policy or DataQualityPolicyV43()
    x = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(x) <= p.label_horizon_bars:
        return x.iloc[0:0].copy()
    return x.iloc[: -p.label_horizon_bars].copy()


def moving_block_resample_positions_v43(
    n_rows: int,
    *,
    seed: int,
    block_length: int | None = None,
    policy: DataQualityPolicyV43 | None = None,
) -> np.ndarray:
    """Deterministic moving-block bootstrap positions of length n_rows."""
    p = policy or DataQualityPolicyV43()
    b = int(block_length or p.perturbation_block_events)
    n = int(n_rows)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if b > n:
        b = n
    rng = np.random.default_rng(int(seed))
    blocks: list[np.ndarray] = []
    total = 0
    max_start = n - b
    while total < n:
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        block = np.arange(start, start + b, dtype=np.int64)
        blocks.append(block)
        total += len(block)
    return np.concatenate(blocks)[:n]


def training_support_v43(
    training_events: pd.DataFrame,
    policy: DataQualityPolicyV43 | None = None,
) -> dict:
    """Fold-local, outcome-blind support gate for one asset."""
    p = policy or DataQualityPolicyV43()
    if "venue" not in training_events.columns:
        raise ValueError("venue column required")
    counts = training_events["venue"].astype(str).value_counts().to_dict()
    supported_venues = [
        v for v in V43_DEVELOPMENT_VENUES if int(counts.get(v, 0)) >= p.minimum_venue_train_events
    ]
    ok = len(training_events) >= p.minimum_asset_train_events and len(supported_venues) >= p.minimum_training_venues
    return {
        "supported": bool(ok),
        "total_training_events": int(len(training_events)),
        "venue_counts": {v: int(counts.get(v, 0)) for v in V43_DEVELOPMENT_VENUES},
        "supported_venues": supported_venues,
        "reason": "PASS" if ok else "INSUFFICIENT_TRAINING_SUPPORT",
    }


def _empirical_probability(table: pd.DataFrame, outcome_name: str) -> float:
    if table.empty:
        return 0.0
    return float(table["outcome"].astype(str).eq(outcome_name).mean())


def empirical_hazard_baseline_v43(
    training_events: pd.DataFrame,
    target_events: pd.DataFrame,
) -> pd.DataFrame:
    """Training-only hierarchical TARGET/STOP probability baseline.

    Backoff order is family x side x regime -> side x regime -> global.  No
    target/test outcome is read; the returned frame preserves target rows and
    only appends baseline probabilities.
    """
    required = {"event_family_v41", "side", "regime", "outcome"}
    missing = required - set(training_events.columns)
    if missing:
        raise ValueError(f"training events missing {sorted(missing)}")
    target_required = {"event_family_v41", "side", "regime"}
    missing_target = target_required - set(target_events.columns)
    if missing_target:
        raise ValueError(f"target events missing {sorted(missing_target)}")

    train = training_events.copy()
    out = target_events.copy().reset_index(drop=True)
    global_target = _empirical_probability(train, "TARGET")
    global_stop = _empirical_probability(train, "STOP")

    fine: dict[tuple[str, int, str], tuple[int, float, float]] = {}
    for key, g in train.groupby(["event_family_v41", "side", "regime"], dropna=False):
        fine[(str(key[0]), int(key[1]), str(key[2]))] = (
            int(len(g)), _empirical_probability(g, "TARGET"), _empirical_probability(g, "STOP")
        )
    coarse: dict[tuple[int, str], tuple[int, float, float]] = {}
    for key, g in train.groupby(["side", "regime"], dropna=False):
        coarse[(int(key[0]), str(key[1]))] = (
            int(len(g)), _empirical_probability(g, "TARGET"), _empirical_probability(g, "STOP")
        )

    p_target: list[float] = []
    p_stop: list[float] = []
    source: list[str] = []
    for row in out.itertuples(index=False):
        fk = (str(getattr(row, "event_family_v41")), int(getattr(row, "side")), str(getattr(row, "regime")))
        ck = (int(getattr(row, "side")), str(getattr(row, "regime")))
        if fk in fine and fine[fk][0] >= 50:
            _, pt, ps = fine[fk]
            src = "family_side_regime"
        elif ck in coarse and coarse[ck][0] >= 100:
            _, pt, ps = coarse[ck]
            src = "side_regime"
        else:
            pt, ps = global_target, global_stop
            src = "global"
        p_target.append(float(pt))
        p_stop.append(float(ps))
        source.append(src)

    out["baseline_p_target_v43"] = p_target
    out["baseline_p_stop_v43"] = p_stop
    out["baseline_source_v43"] = source
    return out


def brier_skill_v43(y_true: Sequence[float], model_probability: Sequence[float], baseline_probability: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    pm = np.asarray(model_probability, dtype=float)
    pb = np.asarray(baseline_probability, dtype=float)
    if not (len(y) == len(pm) == len(pb)) or len(y) == 0:
        raise ValueError("arrays must have equal non-zero length")
    model_brier = float(np.mean((pm - y) ** 2))
    baseline_brier = float(np.mean((pb - y) ** 2))
    if baseline_brier <= 0.0:
        return 0.0 if model_brier <= 0.0 else float("-inf")
    return float(1.0 - model_brier / baseline_brier)
