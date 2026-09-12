from __future__ import annotations

"""v0.42 breadth-expansion and cluster-robust validation utilities.

v0.42 is a new preregistered experiment motivated by v0.41. It does not relax
v0.41 admission thresholds and does not use Kraken. The v0.41 HistGB
cause-specific competing-risk architecture is carried forward unchanged while
the development universe is expanded under an outcome-independent availability
rule. Validation adds a hierarchical symbol x moving-block bootstrap so that
cross-asset correlation cannot create an artificial sample-size pass.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .event_competing_risk_v41 import (
    CompetingRiskPolicyV41,
    V41_EVENT_FAMILIES,
    V41_FEATURES,
    V41_SEEDS,
)
from .financial_system_v39 import FinancialRiskPolicyV39, ValidationPolicyV39


V42_CANDIDATE = "V42_HISTGB_CAUSE_SPECIFIC"
V42_SOURCE_MODEL = "V41_HISTGB_CAUSE_SPECIFIC"
V42_DEVELOPMENT_VENUES: tuple[str, ...] = ("coinex", "okx", "kucoin")
V42_RESERVED_HOLDOUT = "kraken"
V42_SYMBOL_CANDIDATES: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "DOT/USDT",
    "TRX/USDT",
    "ATOM/USDT",
    "NEAR/USDT",
    "ETC/USDT",
    "FIL/USDT",
    "UNI/USDT",
    "AAVE/USDT",
    "SUI/USDT",
    "TON/USDT",
    "XLM/USDT",
    "ALGO/USDT",
    "ICP/USDT",
    "ARB/USDT",
    "OP/USDT",
)
V42_MIN_COMMON_SYMBOLS = 18
V42_MIN_BARS_PER_VENUE_SYMBOL = 900
V42_CLUSTER_BOOTSTRAP_SAMPLES = 1000
V42_CLUSTER_BLOCK_TRADES = 10
V42_CLUSTER_ALPHA = 0.05


@dataclass(frozen=True)
class BreadthClusterPolicyV42:
    minimum_common_symbols: int = V42_MIN_COMMON_SYMBOLS
    minimum_bars_per_venue_symbol: int = V42_MIN_BARS_PER_VENUE_SYMBOL
    cluster_bootstrap_samples: int = V42_CLUSTER_BOOTSTRAP_SAMPLES
    cluster_block_trades: int = V42_CLUSTER_BLOCK_TRADES
    cluster_alpha: float = V42_CLUSTER_ALPHA
    minimum_executed_per_venue: int = 200
    minimum_positive_asset_fraction: float = 0.60
    minimum_profit_factor: float = 1.05
    minimum_stress_profit_factor: float = 1.00
    minimum_positive_quarter_fraction: float = 0.60
    maximum_drawdown: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_common_symbols < 5:
            raise ValueError("minimum_common_symbols must be >=5")
        if self.minimum_bars_per_venue_symbol < 300:
            raise ValueError("minimum_bars_per_venue_symbol must be >=300")
        if self.cluster_bootstrap_samples < 200:
            raise ValueError("cluster_bootstrap_samples must be >=200")
        if self.cluster_block_trades < 2:
            raise ValueError("cluster_block_trades must be >=2")
        if not (0.0 < self.cluster_alpha < 0.2):
            raise ValueError("cluster_alpha must be in (0,0.2)")


def preregistration_manifest_v42() -> dict:
    return {
        "version": "v0.42",
        "experiment": "BREADTH_EXPANSION_CLUSTER_ROBUST_VALIDATION",
        "hypothesis": (
            "The frozen v0.41 HistGB cause-specific competing-risk edge generalizes "
            "across a broader outcome-independently eligible development universe."
        ),
        "candidate": V42_CANDIDATE,
        "source_model": V42_SOURCE_MODEL,
        "model_capacity_change": False,
        "threshold_relaxation": False,
        "family_pruning": False,
        "event_families": list(V41_EVENT_FAMILIES),
        "feature_columns": list(V41_FEATURES),
        "seeds": list(V41_SEEDS),
        "seed_rule": "row-wise median; never select best seed",
        "candidate_symbol_universe": list(V42_SYMBOL_CANDIDATES),
        "eligibility_rule": (
            "symbol must exist on all three consumed development venues and provide "
            f">={V42_MIN_BARS_PER_VENUE_SYMBOL} 4h bars in the frozen request window; "
            "eligibility is determined before any event outcome or model score is inspected"
        ),
        "minimum_common_symbols": V42_MIN_COMMON_SYMBOLS,
        "development_venues": list(V42_DEVELOPMENT_VENUES),
        "reserved_holdout": V42_RESERVED_HOLDOUT,
        "cluster_validation": {
            "method": "hierarchical symbol-resampling x within-symbol moving-block bootstrap",
            "samples": V42_CLUSTER_BOOTSTRAP_SAMPLES,
            "block_trades": V42_CLUSTER_BLOCK_TRADES,
            "lower_quantile": V42_CLUSTER_ALPHA / 2.0,
            "gate": "cluster_bootstrap_expectancy_lower_bound > 0",
        },
        "competing_risk_policy": asdict(CompetingRiskPolicyV41()),
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
        "post_result_threshold_relaxation": False,
        "post_result_symbol_pruning_by_performance": False,
    }


def select_common_symbols_v42(
    bars_by_venue: Mapping[str, Mapping[str, int]],
    policy: BreadthClusterPolicyV42 | None = None,
) -> tuple[str, ...]:
    """Outcome-independent eligibility from availability and bar count only."""
    p = policy or BreadthClusterPolicyV42()
    eligible: list[str] = []
    for symbol in V42_SYMBOL_CANDIDATES:
        if all(
            int(bars_by_venue.get(venue, {}).get(symbol, 0)) >= p.minimum_bars_per_venue_symbol
            for venue in V42_DEVELOPMENT_VENUES
        ):
            eligible.append(symbol)
    if len(eligible) < p.minimum_common_symbols:
        raise RuntimeError(
            f"DATA_UNAVAILABLE: only {len(eligible)} common symbols; "
            f"need >= {p.minimum_common_symbols}"
        )
    return tuple(eligible)


def _moving_block_sample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x
    b = max(1, min(int(block), n))
    sampled: list[float] = []
    while len(sampled) < n:
        start = int(rng.integers(0, n))
        sampled.extend(x[(start + np.arange(b)) % n].tolist())
    return np.asarray(sampled[:n], dtype=float)


def hierarchical_symbol_block_bootstrap_low_v42(
    trades: pd.DataFrame,
    *,
    value_col: str = "net_r",
    symbol_col: str = "symbol",
    time_col: str = "exit_time",
    seed: int = 314,
    policy: BreadthClusterPolicyV42 | None = None,
) -> float | None:
    """Lower CI for expectancy while preserving symbol and temporal clustering.

    Each bootstrap replicate samples symbols with replacement. For every sampled
    symbol, its time-ordered trade outcomes are resampled with a circular moving
    block bootstrap. This intentionally makes breadth expansion harder to pass
    than treating every trade as independent.
    """
    p = policy or BreadthClusterPolicyV42()
    if trades.empty or symbol_col not in trades or value_col not in trades:
        return None
    x = trades.copy()
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x[np.isfinite(x[value_col])]
    if len(x) < 20:
        return None
    symbols = tuple(sorted(x[symbol_col].astype(str).unique()))
    if len(symbols) < 3:
        return None
    series: dict[str, np.ndarray] = {}
    for symbol in symbols:
        g = x[x[symbol_col].astype(str) == symbol].copy()
        if time_col in g:
            g = g.sort_values(time_col)
        series[symbol] = g[value_col].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    means = np.empty(p.cluster_bootstrap_samples, dtype=float)
    for i in range(p.cluster_bootstrap_samples):
        sampled_symbols = rng.choice(np.asarray(symbols, dtype=object), size=len(symbols), replace=True)
        pieces: list[np.ndarray] = []
        for symbol in sampled_symbols.tolist():
            arr = series[str(symbol)]
            if len(arr):
                pieces.append(_moving_block_sample(arr, p.cluster_block_trades, rng))
        means[i] = float(np.mean(np.concatenate(pieces))) if pieces else np.nan
    means = means[np.isfinite(means)]
    if len(means) < max(100, p.cluster_bootstrap_samples // 2):
        return None
    return float(np.quantile(means, p.cluster_alpha / 2.0))


def v42_gate_from_metrics(metrics: Mapping[str, object], policy: BreadthClusterPolicyV42 | None = None) -> dict:
    p = policy or BreadthClusterPolicyV42()

    def finite(v: object) -> bool:
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    def ge(v: object, threshold: float) -> bool:
        try:
            f = float(v)
            return bool(np.isinf(f) or f >= threshold)
        except (TypeError, ValueError):
            return False

    checks = {
        "n_ge_200": int(metrics.get("n", 0) or 0) >= p.minimum_executed_per_venue,
        "pf_ge_1_05": ge(metrics.get("profit_factor"), p.minimum_profit_factor),
        "expectancy_gt_0": finite(metrics.get("expectancy_r")) and float(metrics["expectancy_r"]) > 0.0,
        "breadth_ge_0_60": ge(metrics.get("positive_asset_fraction"), p.minimum_positive_asset_fraction),
        "block_ci_low_gt_0": finite(metrics.get("block_ci_low")) and float(metrics["block_ci_low"]) > 0.0,
        "cluster_ci_low_gt_0": finite(metrics.get("cluster_ci_low")) and float(metrics["cluster_ci_low"]) > 0.0,
        "positive_quarters_ge_0_60": ge(metrics.get("positive_quarter_fraction"), p.minimum_positive_quarter_fraction),
        "stress_pf_ge_1_00": ge(metrics.get("stress_profit_factor"), p.minimum_stress_profit_factor),
        "max_dd_le_0_05": finite(metrics.get("max_account_drawdown")) and float(metrics["max_account_drawdown"]) >= -p.maximum_drawdown - 1e-12,
    }
    return {**checks, "venue_pass": bool(all(checks.values()))}
