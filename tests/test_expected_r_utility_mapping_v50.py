import numpy as np
import pandas as pd

from research_bot.expected_r_utility_mapping_v50 import (
    broad_harmful_stage_v50,
    ranking_diagnostic_v50,
    route_decision_v50,
    state_transport_metrics_v50,
)


def _ranking_frame(expected, realized):
    n = len(expected)
    return pd.DataFrame({
        "expected_r_v47": expected,
        "net_r": realized,
        "signal_time": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
        "venue": ["coinex"] * n,
    })


def test_state_transport_excess_exact():
    fit = {"TARGET": 3.0, "STOP": -1.0, "TIME_POSITIVE": 0.5, "TIME_NONPOSITIVE": -0.4}
    cal = {"TARGET": 2.9, "STOP": -1.1, "TIME_POSITIVE": 0.4, "TIME_NONPOSITIVE": -0.5}
    test = {"TARGET": 2.5, "STOP": -1.3, "TIME_POSITIVE": 0.1, "TIME_NONPOSITIVE": -0.9}
    d = state_transport_metrics_v50(fit, cal, test, fit.keys())
    assert d["supported"] is True
    assert np.isclose(d["cal_state_mae"], 0.1)
    assert np.isclose(d["test_state_mae"], 0.425)
    assert np.isclose(d["state_transport_excess"], 0.325)


def test_state_transport_requires_three_states():
    d = state_transport_metrics_v50({"A": 1, "B": 2}, {"A": 1, "B": 2}, {"A": 1, "B": 2}, ["A", "B"])
    assert d["supported"] is False
    assert d["state_transport_excess"] is None


def test_ranking_perfect_order_is_positive():
    x = np.linspace(-0.2, 0.8, 100)
    d = ranking_diagnostic_v50(_ranking_frame(x, 2.0 * x + 0.1))
    assert d.spearman_rho is not None and d.spearman_rho > 0.999
    assert d.top_bottom_spread is not None and d.top_bottom_spread > 0
    assert d.ols_slope is not None and np.isclose(d.ols_slope, 2.0)


def test_ranking_inverse_order_is_negative():
    x = np.linspace(-0.2, 0.8, 100)
    d = ranking_diagnostic_v50(_ranking_frame(x, -x))
    assert d.spearman_rho is not None and d.spearman_rho < -0.999
    assert d.top_bottom_spread is not None and d.top_bottom_spread < 0


def test_broad_harmful_stage_requires_three_negative_and_negative_median():
    assert broad_harmful_stage_v50([-0.1, -0.2, -0.3, 0.1, 0.2]) is True
    assert broad_harmful_stage_v50([-0.1, -0.2, 0.1, 0.2, 0.3]) is False


def test_decision_routing_specific_mixed_and_inconclusive():
    specific = route_decision_v50(
        state_failure=False, ranking_failure=True, admission_failure=False,
        nonoverlap_failure=False, governor_failure=False,
    )
    assert specific == "V50_EXPECTED_R_RANKING_FAILURE_SUPPORTED"
    mixed = route_decision_v50(
        state_failure=True, ranking_failure=True, admission_failure=False,
        nonoverlap_failure=False, governor_failure=False,
    )
    assert mixed == "V50_MIXED_UTILITY_PIPELINE_FAILURE_SUPPORTED"
    none = route_decision_v50(
        state_failure=False, ranking_failure=False, admission_failure=False,
        nonoverlap_failure=False, governor_failure=False,
    )
    assert none == "V50_UTILITY_MAPPING_EVIDENCE_INCONCLUSIVE"
