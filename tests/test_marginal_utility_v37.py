import pandas as pd

from research_bot.marginal_utility_v37 import (
    TOTAL_EFFECTIVE_TRIALS_V37,
    V37_POLICIES,
    preregistration_manifest_v37,
    screen_three_venues_v37,
    v37_decision,
)


def test_v37_preregistered_before_v36_result_and_fail_closed():
    m = preregistration_manifest_v37()
    assert m["preregistered_before_v36_result"] is True
    assert m["candidate_count"] == 9
    assert m["total_effective_trials"] == 132 == TOTAL_EFFECTIVE_TRIALS_V37
    assert m["reserved_holdout_venue"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["parent_v36_must_pass"] is True
    assert m["one_open_position_per_symbol"] is True
    assert m["paper_replacement_authorized"] is False
    assert m["live_execution_authorized"] is False
    assert [p.name for p in V37_POLICIES] == ["BALANCED", "TAIL_DEFENSIVE", "DIVERSITY_FIRST"]


def _m(n=250, pf=1.2, exp=0.1, breadth=0.7, dd=-0.04, ci=0.0001):
    return {"trades": n, "profit_factor": pf, "expectancy_r": exp, "positive_asset_fraction": breadth, "max_drawdown": dd, "block_ci_low": ci}


def test_v37_cannot_rescue_parent_event_failure():
    metrics = {v: _m() for v in ("coinex_consumed", "okx_consumed", "kucoin_consumed")}
    s = screen_three_venues_v37(metrics, parent_event_eligible=False)
    assert s["portfolio_eligible_v37"] is True
    assert s["joint_eligible_v37"] is False


def test_v37_requires_all_portfolio_gates_and_parent_pass():
    venues = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
    good = {v: _m() for v in venues}
    assert screen_three_venues_v37(good, parent_event_eligible=True)["joint_eligible_v37"] is True
    bad = {v: _m() for v in venues}
    bad[venues[0]]["max_drawdown"] = -0.051
    assert screen_three_venues_v37(bad, parent_event_eligible=True)["joint_eligible_v37"] is False


def test_v37_pass_still_does_not_authorize_execution():
    d = v37_decision({"strategy": "PARENT__BALANCED"})
    assert d["decision"] == "V37_JOINT_CANDIDATE_LOCKED_FOR_UNTOUCHED_KRAKEN"
    assert d["kraken_touched"] is False
    assert d["holdout_authorized_to_run"] is True
    assert d["paper_replacement_authorized"] is False
    assert d["live_execution_authorized"] is False
