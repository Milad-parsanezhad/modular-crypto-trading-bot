from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfigV52:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    embargo_bars: int = 6

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0,1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0,1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train+validation must leave a test set")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")


@dataclass(frozen=True)
class CostConfigV52:
    round_trip_bps: float = 24.0
    stress_round_trip_bps: float = 36.0

    def __post_init__(self) -> None:
        if self.round_trip_bps < 0 or self.stress_round_trip_bps < self.round_trip_bps:
            raise ValueError("invalid cost configuration")


@dataclass(frozen=True)
class PerformanceV52:
    observations: int
    cumulative_return: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown: float
    calmar: float
    profit_factor: float
    turnover: float
    cvar_95: float

    def to_dict(self) -> dict:
        return asdict(self)


def chronological_split_v52(frame: pd.DataFrame, *, time_col: str = "timestamp", config: SplitConfigV52 | None = None) -> dict[str, pd.DataFrame]:
    cfg = config or SplitConfigV52()
    if time_col not in frame:
        raise ValueError(f"missing {time_col}")
    x = frame.copy()
    x[time_col] = pd.to_datetime(x[time_col], utc=True, errors="coerce")
    if x[time_col].isna().any():
        raise ValueError("invalid timestamps")
    x = x.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    if x[time_col].duplicated().any():
        raise ValueError("duplicate timestamps are forbidden")
    n = len(x)
    if n < 100:
        raise ValueError("v0.52 benchmark requires at least 100 chronological rows")
    train_end = int(n * cfg.train_fraction)
    val_end = int(n * (cfg.train_fraction + cfg.validation_fraction))
    e = cfg.embargo_bars
    train = x.iloc[: max(0, train_end - e)].copy()
    validation = x.iloc[min(n, train_end + e): max(min(n, train_end + e), val_end - e)].copy()
    test = x.iloc[min(n, val_end + e):].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("embargo leaves an empty split")
    if not (train[time_col].max() < validation[time_col].min() < test[time_col].min()):
        raise ValueError("chronological split invariant violated")
    return {"train": train, "validation": validation, "test": test}


def positions_from_scores_v52(scores: Iterable[float], *, threshold: float = 0.0) -> np.ndarray:
    s = np.asarray(list(scores), dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("scores must be finite 1D")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return np.where(s > threshold, 1.0, np.where(s < -threshold, -1.0, 0.0))


def apply_transaction_costs_v52(returns: Iterable[float], positions: Iterable[float], *, round_trip_bps: float) -> tuple[np.ndarray, float]:
    r = np.asarray(list(returns), dtype=float)
    p = np.asarray(list(positions), dtype=float)
    if r.shape != p.shape or r.ndim != 1 or not np.isfinite(r).all() or not np.isfinite(p).all():
        raise ValueError("returns and positions must be aligned finite vectors")
    if np.any(r <= -1.0):
        raise ValueError("simple returns <= -100% are invalid for this benchmark")
    if round_trip_bps < 0:
        raise ValueError("round_trip_bps must be non-negative")
    if np.any(np.abs(p) > 1):
        raise ValueError("position magnitude cannot exceed 1 in v0.52 benchmark")
    previous = np.concatenate([[0.0], p[:-1]])
    turnover = np.abs(p - previous)
    one_way_cost = round_trip_bps / 20_000.0
    net = p * r - turnover * one_way_cost
    if np.any(net <= -1.0):
        raise ValueError("net return <= -100% violates benchmark equity assumptions")
    return net, float(turnover.sum())


def performance_metrics_v52(net_returns: Iterable[float], *, turnover: float = 0.0, periods_per_year: float = 6 * 365.25) -> PerformanceV52:
    r = np.asarray(list(net_returns), dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("net_returns must contain at least two finite values")
    if np.any(r <= -1.0) or periods_per_year <= 0:
        raise ValueError("invalid return or annualization domain")
    log_growth = np.log1p(r)
    equity = np.exp(np.cumsum(log_growth))
    cumulative = float(equity[-1] - 1.0)
    std = float(r.std(ddof=1))
    sharpe = float(r.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    annual_log_growth = float(log_growth.mean() * periods_per_year)
    cagr = float(np.exp(np.clip(annual_log_growth, -50.0, 50.0)) - 1.0)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = float(gains / losses) if losses > 0 else (1e12 if gains > 0 else 0.0)
    cutoff = float(np.quantile(r, 0.05))
    tail = r[r <= cutoff]
    cvar95 = float(-tail.mean()) if len(tail) else 0.0
    values = np.asarray([cumulative, sharpe, sortino, mdd, calmar, pf, turnover, cvar95], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite performance metric")
    return PerformanceV52(len(r), cumulative, sharpe, sortino, mdd, calmar, pf, float(turnover), cvar95)


def evaluate_scores_v52(scores: Iterable[float], future_returns: Iterable[float], *, threshold: float = 0.0, costs: CostConfigV52 | None = None) -> dict[str, dict]:
    cfg = costs or CostConfigV52()
    positions = positions_from_scores_v52(scores, threshold=threshold)
    base_net, base_turnover = apply_transaction_costs_v52(future_returns, positions, round_trip_bps=cfg.round_trip_bps)
    stress_net, stress_turnover = apply_transaction_costs_v52(future_returns, positions, round_trip_bps=cfg.stress_round_trip_bps)
    return {
        "base": performance_metrics_v52(base_net, turnover=base_turnover).to_dict(),
        "stress": performance_metrics_v52(stress_net, turnover=stress_turnover).to_dict(),
    }


def summarize_seed_runs_v52(rows: pd.DataFrame, *, metric: str = "annualized_sharpe") -> dict:
    required = {"model", "seed", metric}
    if not required.issubset(rows.columns):
        raise ValueError(f"missing columns: {sorted(required.difference(rows.columns))}")
    x = rows.copy()
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    if x[metric].isna().any() or not np.isfinite(x[metric]).all():
        raise ValueError("seed metrics must be finite")
    out = {}
    for model, g in x.groupby("model", sort=True):
        if g["seed"].duplicated().any():
            raise ValueError(f"duplicate seed for {model}")
        vals = g[metric].to_numpy(dtype=float)
        out[str(model)] = {
            "n_seeds": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out
