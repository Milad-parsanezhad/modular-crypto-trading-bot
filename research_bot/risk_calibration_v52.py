from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalRiskConfig:
    """Experimental v0.52 calibration controls.

    This module is research-only. It does not authorize paper or live execution.
    """

    alpha: float = 0.99
    decay: float = 0.97
    regime_bandwidth: float = 1.0
    min_observations: int = 40
    min_effective_sample_size: float = 15.0

    def __post_init__(self) -> None:
        if not 0.5 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0.5, 1.0)")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.regime_bandwidth <= 0:
            raise ValueError("regime_bandwidth must be positive")
        if self.min_observations < 10:
            raise ValueError("min_observations must be >= 10")
        if self.min_effective_sample_size <= 0:
            raise ValueError("min_effective_sample_size must be positive")


@dataclass(frozen=True)
class ConformalRiskResult:
    calibrated_var: float
    additive_buffer: float
    effective_sample_size: float
    observation_count: int
    approved_for_research_use: bool
    reason: str


def _finite_1d(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be aligned one-dimensional arrays")
    if len(values) == 0:
        raise ValueError("cannot compute weighted quantile of empty data")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive total")

    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w) / float(w.sum())
    idx = int(np.searchsorted(cumulative, q, side="left"))
    return float(v[min(idx, len(v) - 1)])


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    denom = float(np.square(weights).sum())
    if total <= 0 or denom <= 0:
        return 0.0
    return total * total / denom


def calibrate_one_sided_var(
    *,
    realized_losses: Sequence[float],
    predicted_var: Sequence[float],
    current_predicted_var: float,
    historical_regimes: Sequence[Sequence[float]] | None = None,
    current_regime: Sequence[float] | None = None,
    config: ConformalRiskConfig | None = None,
) -> ConformalRiskResult:
    """Calibrate a one-sided VaR forecast using past nonconformity scores.

    Score_t = realized_loss_t - predicted_var_t. Positive values mean the
    forecast understated loss. Recent observations receive exponentially larger
    weights. When regime vectors are supplied, an RBF similarity weight is
    multiplied into the time-decay weight.

    The returned VaR is always at least the base forecast. Fail-closed behavior
    is used when there are too few valid observations or insufficient effective
    sample size.
    """

    cfg = config or ConformalRiskConfig()
    if not np.isfinite(current_predicted_var) or current_predicted_var < 0:
        raise ValueError("current_predicted_var must be finite and non-negative")

    losses = np.asarray(realized_losses, dtype=float)
    forecasts = np.asarray(predicted_var, dtype=float)
    if losses.ndim != 1 or forecasts.ndim != 1 or len(losses) != len(forecasts):
        raise ValueError("realized_losses and predicted_var must be aligned 1D sequences")

    finite = np.isfinite(losses) & np.isfinite(forecasts) & (forecasts >= 0)
    losses = losses[finite]
    forecasts = forecasts[finite]
    n = len(losses)
    if n < cfg.min_observations:
        return ConformalRiskResult(
            calibrated_var=float(current_predicted_var),
            additive_buffer=0.0,
            effective_sample_size=0.0,
            observation_count=n,
            approved_for_research_use=False,
            reason="INSUFFICIENT_CALIBRATION_HISTORY",
        )

    age = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.power(cfg.decay, age)

    if historical_regimes is not None or current_regime is not None:
        if historical_regimes is None or current_regime is None:
            raise ValueError("historical_regimes and current_regime must be supplied together")
        regimes = np.asarray(historical_regimes, dtype=float)
        current = np.asarray(current_regime, dtype=float)
        if regimes.ndim != 2 or current.ndim != 1 or regimes.shape[0] != len(finite):
            raise ValueError("regime history must align with the unfiltered input history")
        regimes = regimes[finite]
        if regimes.shape[1] != current.shape[0]:
            raise ValueError("current_regime dimension must match historical regimes")
        if not np.all(np.isfinite(regimes)) or not np.all(np.isfinite(current)):
            raise ValueError("regime features must be finite")
        distances_sq = np.square(regimes - current).sum(axis=1)
        weights *= np.exp(-0.5 * distances_sq / (cfg.regime_bandwidth**2))

    ess = _effective_sample_size(weights)
    if ess < cfg.min_effective_sample_size:
        return ConformalRiskResult(
            calibrated_var=float(current_predicted_var),
            additive_buffer=0.0,
            effective_sample_size=float(ess),
            observation_count=n,
            approved_for_research_use=False,
            reason="INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        )

    scores = losses - forecasts
    correction = max(0.0, _weighted_quantile(scores, weights, cfg.alpha))
    return ConformalRiskResult(
        calibrated_var=float(current_predicted_var + correction),
        additive_buffer=float(correction),
        effective_sample_size=float(ess),
        observation_count=n,
        approved_for_research_use=True,
        reason="CALIBRATED_RESEARCH_ONLY",
    )
