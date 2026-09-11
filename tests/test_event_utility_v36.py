import numpy as np
import pandas as pd

from research_bot.event_utility_v36 import (
    BASE_D1_STRATEGIES_V36,
    DEVELOPMENT_VENUES_V36,
    TOTAL_EFFECTIVE_TRIALS_V36,
    V36_CANDIDATES,
    add_fixed_strategy_features_v36,
    preregistration_manifest_v36,
    screen_three_venues_v36,
    v36_decision,
)


def test_v36_preregistration_is_fail_closed_and_kraken_sealed():
    m = preregistration_manifest_v36()
    assert m["candidate_count"] == 3
    assert m["total_effective_trials"] == 123 == TOTAL_EFFECTIVE_TRIALS_V36
    assert m["selection_fraction"] == 0.50
    assert m["development_venues"] == list(DEVELOPMENT_VENUES_V36)
    assert m["reserved_holdout_venue"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["portfolio_drawdown_gate_applied_here"] is False
    assert m["paper_replacement_authorized"] is False
    assert m["live_execution_authorized"] is False
    assert len(BASE_D1_STRATEGIES_V36) == 6
    assert [x.model_family for x in V36_CANDIDATES] == ["ridge", "huber", "histgb"]


def test_strategy_and_calendar_features_are_deterministic():
    x = pd.DataFrame({
        "signal_time": pd.to_datetime(["2025-01-01", "2025-07-01"], utc=True),
        "side": [1, -1],
        "base_strategy_v36": [BASE_D1_STRATEGIES_V36[0], BASE_D1_STRATEGIES_V36[-1]],
    })
    y = add_fixed_strategy_features_v36(x)
    assert y["f_strategy_0_v36"].tolist() == [1.0, 0.0]
    assert y[f"f_strategy_{len(BASE_D1_STRATEGIES_V36)-1}_v36"].tolist() == [0.0, 1.0]
    assert y["f_side_v36"].tolist() == [1, -1]
    assert np.isfinite(y[["f_month_sin_v36", "f_month_cos_v36"]].to_numpy()).all()


def _metrics(*, n=250, pf=1.2, exp=0.1, breadth=0.7, ci=0.0001, stress=1.1):
    return {
        "selected_events": n,
        "profit_factor": pf,
        "expectancy_r": exp,
        "positive_asset_fraction": breadth,
        "block_ci_low": ci,
        "stress_36bps_profit_factor": stress,
    }


def test_three_venue_gate_requires_every_frozen_condition():
    good = {v: _metrics() for v in DEVELOPMENT_VENUES_V36}
    assert screen_three_venues_v36(good)["development_eligible_v36"] is True
    for key, bad_value in [
        ("selected_events", 199),
        ("profit_factor", 1.049),
        ("expectancy_r", 0.0),
        ("positive_asset_fraction", 0.59),
        ("block_ci_low", 0.0),
        ("stress_36bps_profit_factor", 0.99),
    ]:
        bad = {v: _metrics() for v in DEVELOPMENT_VENUES_V36}
        bad[DEVELOPMENT_VENUES_V36[0]][key] = bad_value
        assert screen_three_venues_v36(bad)["development_eligible_v36"] is False


def test_v36_cannot_authorize_execution_even_with_winner():
    d = v36_decision({"strategy": "X"})
    assert d["decision"] == "V36_EVENT_UTILITY_CANDIDATE_REQUIRES_PORTFOLIO_ARBITRATION"
    assert d["kraken_touched"] is False
    assert d["portfolio_gate_complete"] is False
    assert d["paper_replacement_authorized"] is False
    assert d["live_execution_authorized"] is False
