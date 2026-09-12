from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_bot.breadth_cluster_v42 import (
    BreadthClusterPolicyV42,
    V42_CANDIDATE,
    V42_DEVELOPMENT_VENUES,
    V42_SOURCE_MODEL,
    V42_SYMBOL_CANDIDATES,
    hierarchical_symbol_block_bootstrap_low_v42,
    preregistration_manifest_v42,
    select_common_symbols_v42,
    v42_gate_from_metrics,
)


def test_v42_manifest_is_fail_closed_and_not_threshold_rescue() -> None:
    m = preregistration_manifest_v42()
    assert m["candidate"] == V42_CANDIDATE
    assert m["source_model"] == V42_SOURCE_MODEL == "V41_HISTGB_CAUSE_SPECIFIC"
    assert m["model_capacity_change"] is False
    assert m["threshold_relaxation"] is False
    assert m["family_pruning"] is False
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["reserved_holdout"] == "kraken"


def test_v42_candidate_universe_contains_original_five_and_is_unique() -> None:
    original = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"}
    assert original.issubset(set(V42_SYMBOL_CANDIDATES))
    assert len(V42_SYMBOL_CANDIDATES) == len(set(V42_SYMBOL_CANDIDATES))
    assert len(V42_SYMBOL_CANDIDATES) == 25


def test_common_symbol_eligibility_uses_only_bar_counts() -> None:
    p = BreadthClusterPolicyV42(minimum_common_symbols=5, minimum_bars_per_venue_symbol=900)
    bars = {v: {s: 1000 for s in V42_SYMBOL_CANDIDATES[:6]} for v in V42_DEVELOPMENT_VENUES}
    bars["coinex"][V42_SYMBOL_CANDIDATES[5]] = 899
    eligible = select_common_symbols_v42(bars, p)
    assert eligible == V42_SYMBOL_CANDIDATES[:5]


def test_common_symbol_eligibility_fails_closed_if_too_few() -> None:
    p = BreadthClusterPolicyV42(minimum_common_symbols=5, minimum_bars_per_venue_symbol=900)
    bars = {v: {s: 1000 for s in V42_SYMBOL_CANDIDATES[:4]} for v in V42_DEVELOPMENT_VENUES}
    with pytest.raises(RuntimeError, match="DATA_UNAVAILABLE"):
        select_common_symbols_v42(bars, p)


def _synthetic_trades(mean: float, seed: int = 314) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for j, symbol in enumerate(("A", "B", "C", "D", "E", "F")):
        values = rng.normal(loc=mean, scale=0.12, size=60)
        for i, value in enumerate(values):
            rows.append(
                {
                    "symbol": symbol,
                    "exit_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=4 * (i + 70 * j)),
                    "net_r": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_deterministic_and_positive_for_strong_edge() -> None:
    p = BreadthClusterPolicyV42(cluster_bootstrap_samples=300, cluster_block_trades=8)
    trades = _synthetic_trades(0.35)
    a = hierarchical_symbol_block_bootstrap_low_v42(trades, seed=314, policy=p)
    b = hierarchical_symbol_block_bootstrap_low_v42(trades, seed=314, policy=p)
    assert a is not None and b is not None
    assert a == b
    assert a > 0.0


def test_v42_gate_requires_cluster_ci_and_sample_size() -> None:
    metrics = {
        "n": 250,
        "profit_factor": 1.4,
        "expectancy_r": 0.2,
        "positive_asset_fraction": 0.7,
        "block_ci_low": 0.03,
        "cluster_ci_low": 0.02,
        "positive_quarter_fraction": 0.8,
        "stress_profit_factor": 1.2,
        "max_account_drawdown": -0.02,
    }
    assert v42_gate_from_metrics(metrics)["venue_pass"] is True
    bad_cluster = dict(metrics, cluster_ci_low=-0.001)
    assert v42_gate_from_metrics(bad_cluster)["venue_pass"] is False
    bad_n = dict(metrics, n=199)
    assert v42_gate_from_metrics(bad_n)["venue_pass"] is False
