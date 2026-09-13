import numpy as np
import pandas as pd
import pytest

from research_bot.sequence_benchmark_v52 import (
    CostConfigV52,
    SplitConfigV52,
    apply_transaction_costs_v52,
    chronological_split_v52,
    evaluate_scores_v52,
    performance_metrics_v52,
    summarize_seed_runs_v52,
)
from research_bot.deep_models_v52 import DeepModelConfigV52


def _frame(n=200):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "x": np.arange(n, dtype=float),
    })


def test_chronological_split_has_embargo_and_no_overlap():
    parts = chronological_split_v52(_frame(), config=SplitConfigV52(0.6, 0.2, embargo_bars=3))
    assert parts["train"]["timestamp"].max() < parts["validation"]["timestamp"].min()
    assert parts["validation"]["timestamp"].max() < parts["test"]["timestamp"].min()
    assert set(parts["train"].index).isdisjoint(parts["validation"].index)


def test_duplicate_time_fails_closed():
    x = _frame()
    x.loc[1, "timestamp"] = x.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        chronological_split_v52(x)


def test_costs_charge_position_changes_and_stress_is_worse():
    returns = [0.01, 0.01, 0.01, 0.01]
    positions = [1, -1, 1, -1]
    base, turnover = apply_transaction_costs_v52(returns, positions, round_trip_bps=24)
    stress, _ = apply_transaction_costs_v52(returns, positions, round_trip_bps=36)
    assert turnover == 7.0
    assert stress.sum() < base.sum()


def test_performance_reports_drawdown_and_cvar():
    m = performance_metrics_v52([0.01, -0.02, 0.005, -0.01, 0.02])
    assert m.max_drawdown < 0
    assert m.cvar_95 > 0
    assert np.isfinite(m.annualized_sharpe)


def test_end_to_end_score_evaluation_has_base_and_stress():
    out = evaluate_scores_v52(
        [1.0, 1.0, -1.0, -1.0, 0.2],
        [0.01, -0.005, -0.01, 0.004, 0.002],
        costs=CostConfigV52(24, 36),
    )
    assert set(out) == {"base", "stress"}
    assert out["stress"]["cumulative_return"] <= out["base"]["cumulative_return"]


def test_seed_summary_rejects_duplicate_seed():
    rows = pd.DataFrame({"model": ["x", "x"], "seed": [1, 1], "annualized_sharpe": [0.2, 0.3]})
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_seed_runs_v52(rows)


def test_deep_config_rejects_invalid_transformer_shape():
    with pytest.raises(ValueError, match="divisible"):
        DeepModelConfigV52(input_dim=10, d_model=62, nhead=4)
