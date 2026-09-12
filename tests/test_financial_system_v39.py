from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_bot.financial_system_v39 import (
    ExperimentLoopGuardV39,
    ExperimentState,
    FinancialRiskPolicyV39,
    allocate_portfolio_risk,
    causal_robust_normalize,
    drawdown_risk_scale,
    generalization_gate_v39,
    preregistration_manifest_v39,
    seed_ensemble_median,
    tensor_ready_matrix,
)


def _panel(n: int = 260) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for symbol_i, symbol in enumerate(("BTC/USDT", "ETH/USDT")):
        base = np.linspace(0.0, 4.0, n) + symbol_i
        cycle = np.sin(np.linspace(0.0, 8.0 * np.pi, n))
        for i in range(n):
            rows.append(
                {
                    "timestamp": ts[i],
                    "symbol": symbol,
                    "f1": base[i] + cycle[i],
                    "f2": 10.0 + 0.1 * base[i] - cycle[i],
                }
            )
    return pd.DataFrame(rows)


def test_manifest_keeps_kraken_sealed_and_execution_disabled() -> None:
    m = preregistration_manifest_v39()
    assert m["reserved_holdout_venue"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["seed_selection_rule"].startswith("median ensemble")


def test_causal_normalizer_is_invariant_to_future_perturbation() -> None:
    panel = _panel()
    base = causal_robust_normalize(panel, ["f1", "f2"])

    perturbed = panel.copy()
    cutoff = pd.Timestamp("2025-01-31", tz="UTC")
    future = perturbed["timestamp"] > cutoff
    perturbed.loc[future, "f1"] *= 1_000.0
    perturbed.loc[future, "f2"] -= 100_000.0
    changed = causal_robust_normalize(perturbed, ["f1", "f2"])

    early_base = base.loc[base["timestamp"] <= cutoff, ["f1", "f2"]].reset_index(drop=True)
    early_changed = changed.loc[changed["timestamp"] <= cutoff, ["f1", "f2"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_base, early_changed, check_exact=True)


def test_tensor_contract_is_finite_float32() -> None:
    normalized = causal_robust_normalize(_panel(), ["f1", "f2"])
    matrix = tensor_ready_matrix(normalized, ["f1", "f2"])
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert matrix.shape[1] == 4  # two features + two missingness indicators


def test_drawdown_scaling_and_hard_kill() -> None:
    p = FinancialRiskPolicyV39()
    assert drawdown_risk_scale(1.00, 1.00, p) == 1.0
    assert drawdown_risk_scale(0.975, 1.00, p) == p.drawdown_scale_1
    assert drawdown_risk_scale(0.96, 1.00, p) == p.drawdown_scale_2
    assert drawdown_risk_scale(0.95, 1.00, p) == 0.0


def test_risk_allocator_respects_portfolio_and_directional_caps() -> None:
    p = FinancialRiskPolicyV39()
    proposals = pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
            "side": [1, 1, -1, -1],
            "lower_expected_r": [0.4, 0.3, 0.5, 0.2],
            "uncertainty_width_r": [0.1, 0.2, 0.1, 0.3],
            "stop_fraction": [0.02, 0.03, 0.025, 0.02],
        }
    )
    out = allocate_portfolio_risk(proposals, equity=1.0, peak=1.0, policy=p)
    assert float(out["allocated_risk_fraction"].sum()) <= p.aggregate_open_risk_cap + 1e-12
    assert float(out.loc[out["side"] == 1, "allocated_risk_fraction"].sum()) <= p.directional_open_risk_cap + 1e-12
    assert float(out.loc[out["side"] == -1, "allocated_risk_fraction"].sum()) <= p.directional_open_risk_cap + 1e-12
    assert float(out["position_weight"].max()) <= p.max_asset_weight + 1e-12
    assert float(out["position_weight"].sum()) <= p.max_portfolio_gross + 1e-12


def test_nonpositive_lower_bound_gets_zero_risk() -> None:
    proposals = pd.DataFrame(
        {
            "symbol": ["BTC/USDT"],
            "side": [1],
            "lower_expected_r": [0.0],
            "uncertainty_width_r": [0.1],
            "stop_fraction": [0.02],
        }
    )
    out = allocate_portfolio_risk(proposals, equity=1.0, peak=1.0)
    assert float(out.loc[0, "allocated_risk_fraction"]) == 0.0
    assert float(out.loc[0, "position_weight"]) == 0.0


def test_seed_ensemble_uses_fixed_median_not_best_seed() -> None:
    p1 = np.array([1.0, -2.0, 3.0])
    p2 = np.array([2.0, -1.0, 2.0])
    p3 = np.array([100.0, 5.0, -50.0])
    got = seed_ensemble_median([p1, p2, p3])
    np.testing.assert_allclose(got, np.array([2.0, -1.0, 2.0]))


def test_loop_guard_is_irreversible_after_development_test() -> None:
    g = ExperimentLoopGuardV39()
    for _ in range(3):
        g.register_training_attempt()
    g.transition(ExperimentState.TRAINED)
    g.transition(ExperimentState.CALIBRATED)
    g.transition(ExperimentState.VALIDATED)
    g.transition(ExperimentState.FROZEN)
    g.transition(ExperimentState.DEVELOPMENT_TESTED)
    with pytest.raises(RuntimeError):
        g.transition(ExperimentState.TRAINED)
    with pytest.raises(RuntimeError):
        g.register_training_attempt()
    with pytest.raises(RuntimeError):
        g.touch_holdout()


def test_generalization_gate_requires_all_venues_and_seed_stability() -> None:
    result = generalization_gate_v39(
        fold_expectancies_r=[0.10, 0.05, 0.03, -0.01, 0.02],
        seed_expectancies_r=[0.04, 0.03, 0.02],
        selected_events_by_venue=[250, 260, 240],
        profit_factors_by_venue=[1.10, 1.08, 1.12],
        expectancy_by_venue=[0.03, 0.02, 0.04],
        breadth_by_venue=[0.8, 0.6, 0.8],
        block_ci_low_by_venue=[0.001, 0.002, 0.001],
        positive_quarter_fraction_by_venue=[0.8, 0.6, 0.8],
        stress_pf_by_venue=[1.03, 1.01, 1.05],
    )
    assert result["pass"] is True
    assert result["kraken_authorized"] is False

    unstable = generalization_gate_v39(
        fold_expectancies_r=[0.10, 0.05, 0.03, -0.01, 0.02],
        seed_expectancies_r=[0.04, -0.03, -0.02],
        selected_events_by_venue=[250, 260, 240],
        profit_factors_by_venue=[1.10, 1.08, 1.12],
        expectancy_by_venue=[0.03, 0.02, 0.04],
        breadth_by_venue=[0.8, 0.6, 0.8],
        block_ci_low_by_venue=[0.001, 0.002, 0.001],
        positive_quarter_fraction_by_venue=[0.8, 0.6, 0.8],
        stress_pf_by_venue=[1.03, 1.01, 1.05],
    )
    assert unstable["pass"] is False
