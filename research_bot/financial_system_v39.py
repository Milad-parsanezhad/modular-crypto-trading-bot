from __future__ import annotations

"""v0.39 robust financial/learning control system.

This module does not place orders.  It defines the causal normalization,
regularization/optimizer policy, experiment loop guard and portfolio-risk
constraints that every v0.39 learning model must satisfy before it can be
considered for development qualification.

Design goals:
- reduce overfitting through purged temporal evaluation, fixed complexity ladder,
  seed ensembling and regularization;
- reduce underfitting by requiring each added model class to beat simpler frozen
  baselines out-of-sample rather than by increasing capacity blindly;
- reduce local-optimum dependence through a fixed multi-seed median ensemble;
- prevent tuning loops through an irreversible experiment state machine;
- keep all normalization causal and neural-network ready;
- preserve the historical portfolio limits and 5% hard drawdown firewall.

Kraken remains a sealed external holdout.  PAPER/LIVE execution is disabled.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEVELOPMENT_VENUES_V39 = ("coinex", "okx", "kucoin")
RESERVED_HOLDOUT_VENUE_V39 = "kraken"


@dataclass(frozen=True)
class NormalizationPolicyV39:
    """Causal robust scaler suitable for tabular or sequence neural inputs."""

    rolling_window: int = 180
    min_history: int = 60
    mad_scale: float = 1.4826
    clip_z: float = 5.0
    epsilon: float = 1e-8
    add_missing_indicators: bool = True


@dataclass(frozen=True)
class LearningPolicyV39:
    """Frozen capacity/optimizer policy; not a hyperparameter search space."""

    # Complexity is promoted only when the previous level fails to capture a
    # reproducible OOS relationship and the higher level adds robust OOS value.
    complexity_ladder: tuple[str, ...] = (
        "ridge",
        "hist_gradient_boosting",
        "shallow_mlp",
        "patchtst_or_cmamba",
    )
    ensemble_seeds: tuple[int, ...] = (314, 1618, 2718)
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.15
    max_epochs: int = 80
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
    warmup_fraction: float = 0.05
    max_restarts_per_seed: int = 1
    huber_delta_r: float = 1.0
    lower_quantile: float = 0.20
    duration_loss_weight: float = 0.15
    lower_quantile_loss_weight: float = 0.25


@dataclass(frozen=True)
class ValidationPolicyV39:
    """Frozen anti-overfit gates for model selection on consumed venues only."""

    minimum_purged_folds: int = 5
    embargo_bars: int = 30
    minimum_positive_fold_fraction: float = 0.60
    minimum_positive_seed_fraction: float = 2.0 / 3.0
    minimum_selected_events_per_venue: int = 200
    minimum_profit_factor: float = 1.05
    minimum_expectancy_r: float = 0.0
    minimum_positive_asset_fraction: float = 0.60
    minimum_block_ci_low: float = 0.0
    minimum_positive_quarter_fraction: float = 0.60
    stress_round_trip_bps: float = 36.0
    stress_min_profit_factor: float = 1.0
    group_dro_axes: tuple[str, ...] = ("venue", "quarter", "regime")
    holdout_retraining_allowed: bool = False
    post_result_threshold_relaxation: bool = False


@dataclass(frozen=True)
class FinancialRiskPolicyV39:
    """Portfolio capital policy inherited from the strongest prior risk lineage."""

    base_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.0050
    aggregate_open_risk_cap: float = 0.020
    directional_open_risk_cap: float = 0.015
    max_asset_weight: float = 0.35
    max_portfolio_gross: float = 0.70
    drawdown_warn_1: float = 0.020
    drawdown_warn_2: float = 0.035
    hard_drawdown_cap: float = 0.050
    drawdown_scale_1: float = 0.75
    drawdown_scale_2: float = 0.50
    minimum_uncertainty_scale: float = 0.25
    base_round_trip_bps: float = 24.0
    stress_round_trip_bps: float = 36.0

    def __post_init__(self) -> None:
        if not (0 < self.base_risk_per_trade <= self.max_risk_per_trade):
            raise ValueError("invalid per-trade risk")
        if not (
            0 < self.directional_open_risk_cap <= self.aggregate_open_risk_cap < 1
        ):
            raise ValueError("invalid portfolio risk caps")
        if not (0 < self.drawdown_warn_1 < self.drawdown_warn_2 < self.hard_drawdown_cap < 1):
            raise ValueError("invalid drawdown thresholds")


class ExperimentState(str, Enum):
    CREATED = "CREATED"
    TRAINED = "TRAINED"
    CALIBRATED = "CALIBRATED"
    VALIDATED = "VALIDATED"
    FROZEN = "FROZEN"
    DEVELOPMENT_TESTED = "DEVELOPMENT_TESTED"
    REJECTED = "REJECTED"


_ALLOWED_TRANSITIONS: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.CREATED: {ExperimentState.TRAINED, ExperimentState.REJECTED},
    ExperimentState.TRAINED: {ExperimentState.CALIBRATED, ExperimentState.REJECTED},
    ExperimentState.CALIBRATED: {ExperimentState.VALIDATED, ExperimentState.REJECTED},
    ExperimentState.VALIDATED: {ExperimentState.FROZEN, ExperimentState.REJECTED},
    ExperimentState.FROZEN: {ExperimentState.DEVELOPMENT_TESTED, ExperimentState.REJECTED},
    ExperimentState.DEVELOPMENT_TESTED: set(),
    ExperimentState.REJECTED: set(),
}


@dataclass
class ExperimentLoopGuardV39:
    """Finite-state guard that forbids result-driven retraining loops."""

    state: ExperimentState = ExperimentState.CREATED
    training_attempts: int = 0
    max_training_attempts: int = 6  # 3 fixed seeds x max 1 restart each.
    holdout_touched: bool = False

    def register_training_attempt(self) -> None:
        if self.state is not ExperimentState.CREATED:
            raise RuntimeError("training attempts are only allowed before first TRAINED transition")
        self.training_attempts += 1
        if self.training_attempts > self.max_training_attempts:
            raise RuntimeError("v0.39 training-attempt budget exhausted")

    def touch_holdout(self) -> None:
        raise RuntimeError("Kraken is sealed; v0.39 development code cannot touch the holdout")

    def transition(self, new_state: ExperimentState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"illegal v0.39 experiment transition: {self.state} -> {new_state}")
        self.state = new_state


def preregistration_manifest_v39() -> dict:
    """Machine-readable policy block for v0.39 preregistration."""

    return {
        "version": "v0.39",
        "experiment": "ROBUST_FINANCIAL_LEARNING_SYSTEM",
        "development_venues": list(DEVELOPMENT_VENUES_V39),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V39,
        "normalization": asdict(NormalizationPolicyV39()),
        "learning": asdict(LearningPolicyV39()),
        "validation": asdict(ValidationPolicyV39()),
        "financial_risk": asdict(FinancialRiskPolicyV39()),
        "normalization_is_causal": True,
        "neural_input_contract": "float32 finite matrix + missingness indicators",
        "seed_selection_rule": "median ensemble; never cherry-pick best seed",
        "complexity_rule": "higher capacity requires incremental OOS evidence over frozen simpler baseline",
        "group_robustness_rule": "minimize/qualify worst venue-quarter-regime groups, not only global mean",
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
    }


def _rolling_mad(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda a: float(np.median(np.abs(a - np.median(a)))), raw=True
    )


def causal_robust_normalize(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    policy: NormalizationPolicyV39 | None = None,
) -> pd.DataFrame:
    """Return leakage-safe robust-z features using *prior* observations only.

    For each symbol and feature, location/scale at row t are estimated from rows
    strictly before t via shifted rolling median/MAD.  The current observation is
    therefore never used to normalize itself.  Missing normalized values are
    filled with zero after an optional missingness indicator is created.
    """

    p = policy or NormalizationPolicyV39()
    required = {timestamp_col, group_col, *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing normalization columns: {sorted(missing)}")

    x = frame.copy()
    x[timestamp_col] = pd.to_datetime(x[timestamp_col], utc=True, errors="raise")
    x = x.sort_values([group_col, timestamp_col], kind="mergesort").reset_index(drop=True)

    out = x[[timestamp_col, group_col]].copy()
    for column in feature_columns:
        values = pd.to_numeric(x[column], errors="coerce")
        normalized = pd.Series(np.nan, index=x.index, dtype=float)

        for _, idx in x.groupby(group_col, sort=False).groups.items():
            loc = pd.Index(idx)
            s = values.loc[loc]
            history = s.shift(1)
            median = history.rolling(p.rolling_window, min_periods=p.min_history).median()
            mad = _rolling_mad(history, p.rolling_window, p.min_history)
            robust_scale = p.mad_scale * mad
            fallback_std = history.rolling(p.rolling_window, min_periods=p.min_history).std(ddof=0)
            scale = robust_scale.where(robust_scale > p.epsilon, fallback_std)
            scale = scale.where(scale > p.epsilon, np.nan)
            z = (s - median) / scale
            normalized.loc[loc] = z.clip(-p.clip_z, p.clip_z)

        if p.add_missing_indicators:
            out[f"{column}__missing"] = normalized.isna().astype("float32")
        out[column] = normalized.fillna(0.0).astype("float32")

    return out


def tensor_ready_matrix(
    normalized: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    include_missing_indicators: bool = True,
) -> np.ndarray:
    """Create a finite float32 matrix consumable by PyTorch/NumPy models."""

    columns: list[str] = []
    for feature in feature_columns:
        columns.append(feature)
        missing_col = f"{feature}__missing"
        if include_missing_indicators and missing_col in normalized:
            columns.append(missing_col)
    matrix = normalized.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite value reached v0.39 neural input contract")
    return matrix


def drawdown_risk_scale(
    equity: float,
    peak: float,
    policy: FinancialRiskPolicyV39 | None = None,
) -> float:
    p = policy or FinancialRiskPolicyV39()
    if not np.isfinite(equity) or not np.isfinite(peak) or equity <= 0 or peak <= 0:
        return 0.0
    drawdown = max(0.0, 1.0 - equity / peak)
    if drawdown >= p.hard_drawdown_cap:
        return 0.0
    if drawdown >= p.drawdown_warn_2:
        return p.drawdown_scale_2
    if drawdown >= p.drawdown_warn_1:
        return p.drawdown_scale_1
    return 1.0


def uncertainty_risk_scale(
    uncertainty_width_r: float,
    policy: FinancialRiskPolicyV39 | None = None,
) -> float:
    """Monotonically shrink risk as predictive uncertainty widens."""

    p = policy or FinancialRiskPolicyV39()
    if not np.isfinite(uncertainty_width_r) or uncertainty_width_r < 0:
        return p.minimum_uncertainty_scale
    return float(np.clip(1.0 / (1.0 + uncertainty_width_r), p.minimum_uncertainty_scale, 1.0))


def proposed_trade_risk_fraction(
    *,
    lower_expected_r: float,
    uncertainty_width_r: float,
    equity: float,
    peak: float,
    policy: FinancialRiskPolicyV39 | None = None,
) -> float:
    """Risk only positive lower-bound economic edge; never risk from point mean alone."""

    p = policy or FinancialRiskPolicyV39()
    if not np.isfinite(lower_expected_r) or lower_expected_r <= 0:
        return 0.0
    dd_scale = drawdown_risk_scale(equity, peak, p)
    u_scale = uncertainty_risk_scale(uncertainty_width_r, p)
    raw = p.base_risk_per_trade * dd_scale * u_scale
    return float(np.clip(raw, 0.0, p.max_risk_per_trade))


def allocate_portfolio_risk(
    proposals: pd.DataFrame,
    *,
    equity: float,
    peak: float,
    open_total_risk_fraction: float = 0.0,
    open_long_risk_fraction: float = 0.0,
    open_short_risk_fraction: float = 0.0,
    policy: FinancialRiskPolicyV39 | None = None,
) -> pd.DataFrame:
    """Allocate stop-risk under aggregate, directional, drawdown and asset caps.

    Required proposal columns:
      symbol, side {-1,+1}, lower_expected_r, uncertainty_width_r, stop_fraction.

    The returned ``allocated_risk_fraction`` is account equity at risk at stop.
    ``position_weight`` converts that stop-risk budget into nominal exposure and
    is capped at max_asset_weight.  This is still research-only sizing.
    """

    p = policy or FinancialRiskPolicyV39()
    required = {"symbol", "side", "lower_expected_r", "uncertainty_width_r", "stop_fraction"}
    missing = required - set(proposals.columns)
    if missing:
        raise ValueError(f"missing risk proposal columns: {sorted(missing)}")

    x = proposals.copy().reset_index(drop=True)
    if x.empty:
        x["allocated_risk_fraction"] = pd.Series(dtype=float)
        x["position_weight"] = pd.Series(dtype=float)
        return x

    x["side"] = pd.to_numeric(x["side"], errors="raise").astype(int)
    if bool((~x["side"].isin([-1, 1])).any()):
        raise ValueError("side must be -1 or +1")

    x["proposed_risk_fraction"] = [
        proposed_trade_risk_fraction(
            lower_expected_r=float(row.lower_expected_r),
            uncertainty_width_r=float(row.uncertainty_width_r),
            equity=equity,
            peak=peak,
            policy=p,
        )
        for row in x.itertuples(index=False)
    ]

    # First enforce same-direction risk budgets.
    allocated = x["proposed_risk_fraction"].to_numpy(dtype=float)
    for side, open_risk in ((1, open_long_risk_fraction), (-1, open_short_risk_fraction)):
        mask = x["side"].to_numpy() == side
        wanted = float(allocated[mask].sum())
        available = max(0.0, p.directional_open_risk_cap - float(open_risk))
        if wanted > available and wanted > 0:
            allocated[mask] *= available / wanted

    # Then enforce aggregate open-risk and drawdown headroom.
    aggregate_available = max(0.0, p.aggregate_open_risk_cap - float(open_total_risk_fraction))
    dd_scale = drawdown_risk_scale(equity, peak, p)
    if dd_scale <= 0:
        aggregate_available = 0.0
    if peak > 0 and equity > 0:
        floor_equity = peak * (1.0 - p.hard_drawdown_cap)
        headroom_fraction = max(0.0, (equity - floor_equity) / equity)
        aggregate_available = min(aggregate_available, headroom_fraction)

    wanted_total = float(allocated.sum())
    if wanted_total > aggregate_available and wanted_total > 0:
        allocated *= aggregate_available / wanted_total

    x["allocated_risk_fraction"] = np.clip(allocated, 0.0, p.max_risk_per_trade)
    stop = pd.to_numeric(x["stop_fraction"], errors="coerce").replace(0.0, np.nan)
    nominal_weight = x["allocated_risk_fraction"] / stop.abs()
    x["position_weight"] = nominal_weight.clip(lower=0.0, upper=p.max_asset_weight).fillna(0.0)

    gross = float(x["position_weight"].sum())
    if gross > p.max_portfolio_gross and gross > 0:
        x["position_weight"] *= p.max_portfolio_gross / gross

    return x


def seed_ensemble_median(predictions: Iterable[np.ndarray]) -> np.ndarray:
    """Fixed median ensemble; intentionally no best-seed selection."""

    arrays = [np.asarray(a, dtype=float) for a in predictions]
    if len(arrays) != len(LearningPolicyV39().ensemble_seeds):
        raise ValueError("v0.39 requires exactly the preregistered three seed predictions")
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("seed prediction shapes do not match")
    stack = np.stack(arrays, axis=0)
    if not np.isfinite(stack).all():
        raise ValueError("non-finite seed predictions")
    return np.median(stack, axis=0)


def generalization_gate_v39(
    *,
    fold_expectancies_r: Sequence[float],
    seed_expectancies_r: Sequence[float],
    selected_events_by_venue: Sequence[int],
    profit_factors_by_venue: Sequence[float],
    expectancy_by_venue: Sequence[float],
    breadth_by_venue: Sequence[float],
    block_ci_low_by_venue: Sequence[float],
    positive_quarter_fraction_by_venue: Sequence[float],
    stress_pf_by_venue: Sequence[float],
    policy: ValidationPolicyV39 | None = None,
) -> dict:
    """One frozen gate for overfit/underfit/robustness diagnostics.

    Passing this gate does not authorize the Kraken holdout or execution.  It only
    indicates that the development evidence is strong enough to continue.
    """

    p = policy or ValidationPolicyV39()
    folds = np.asarray(fold_expectancies_r, dtype=float)
    seeds = np.asarray(seed_expectancies_r, dtype=float)
    if folds.size < p.minimum_purged_folds:
        return {"pass": False, "reason": "INSUFFICIENT_PURGED_FOLDS"}
    if seeds.size != 3:
        return {"pass": False, "reason": "SEED_ENSEMBLE_NOT_FROZEN_THREE"}

    positive_fold_fraction = float(np.mean(folds > 0)) if folds.size else 0.0
    positive_seed_fraction = float(np.mean(seeds > 0)) if seeds.size else 0.0

    venue_arrays = [
        np.asarray(selected_events_by_venue, dtype=float),
        np.asarray(profit_factors_by_venue, dtype=float),
        np.asarray(expectancy_by_venue, dtype=float),
        np.asarray(breadth_by_venue, dtype=float),
        np.asarray(block_ci_low_by_venue, dtype=float),
        np.asarray(positive_quarter_fraction_by_venue, dtype=float),
        np.asarray(stress_pf_by_venue, dtype=float),
    ]
    if any(len(a) != len(DEVELOPMENT_VENUES_V39) for a in venue_arrays):
        return {"pass": False, "reason": "VENUE_VECTOR_LENGTH_MISMATCH"}

    checks = {
        "positive_fold_fraction": positive_fold_fraction >= p.minimum_positive_fold_fraction,
        "positive_seed_fraction": positive_seed_fraction >= p.minimum_positive_seed_fraction,
        "selected_events": bool(np.all(venue_arrays[0] >= p.minimum_selected_events_per_venue)),
        "profit_factor": bool(np.all(venue_arrays[1] >= p.minimum_profit_factor)),
        "expectancy": bool(np.all(venue_arrays[2] > p.minimum_expectancy_r)),
        "breadth": bool(np.all(venue_arrays[3] >= p.minimum_positive_asset_fraction)),
        "block_ci_low": bool(np.all(venue_arrays[4] > p.minimum_block_ci_low)),
        "positive_quarters": bool(np.all(venue_arrays[5] >= p.minimum_positive_quarter_fraction)),
        "stress_profit_factor": bool(np.all(venue_arrays[6] >= p.stress_min_profit_factor)),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "positive_fold_fraction": positive_fold_fraction,
        "positive_seed_fraction": positive_seed_fraction,
        "kraken_authorized": False,
        "paper_execution_authorized": False,
        "live_execution_authorized": False,
    }
