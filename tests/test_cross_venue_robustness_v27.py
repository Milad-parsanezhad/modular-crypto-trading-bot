from __future__ import annotations

from research_bot.cross_venue_robustness_v27 import (
    V27RobustnessPolicy,
    development_screen,
    holdout_failures,
    select_development_winner,
    v27_decision,
)


def venue(*, trades=250, pf=1.20, exp=0.10, dd=-0.03, breadth=0.70, ci=0.0002):
    return {
        "venue": "x",
        "trades": trades,
        "profit_factor": pf,
        "expectancy_r": exp,
        "max_drawdown": dd,
        "positive_asset_fraction": breadth,
        "block_ci_low": ci,
    }


def test_development_screen_requires_sample_economics_and_hard_dd_safety_on_both_venues():
    good = development_screen(venue(), venue())
    assert good["development_eligible_v27"] is True

    low_sample = development_screen(venue(trades=199), venue())
    assert low_sample["development_eligible_v27"] is False
    assert low_sample["development_sample_ok_v27"] is False

    weak_economics = development_screen(venue(pf=1.049), venue())
    assert weak_economics["development_eligible_v27"] is False
    assert weak_economics["development_economics_ok_v27"] is False

    hard_dd = development_screen(venue(dd=-0.05001), venue())
    assert hard_dd["development_eligible_v27"] is False
    assert hard_dd["development_safety_ok_v27"] is False


def test_selection_uses_worst_venue_block_ci_before_pf_or_expectancy():
    rows = [
        {
            "strategy": "HIGH_PF_WEAKER_CI",
            "timeframe": "4h",
            "development_eligible_v27": True,
            "robust_floor_block_ci_low_v27": -0.0004,
            "robust_floor_breadth_v27": 0.8,
            "robust_floor_profit_factor_v27": 2.5,
            "robust_floor_expectancy_r_v27": 0.8,
            "robust_min_trades_v27": 400,
            "robust_worst_drawdown_v27": 0.02,
        },
        {
            "strategy": "BETTER_CI",
            "timeframe": "1d",
            "development_eligible_v27": True,
            "robust_floor_block_ci_low_v27": -0.0001,
            "robust_floor_breadth_v27": 0.6,
            "robust_floor_profit_factor_v27": 1.1,
            "robust_floor_expectancy_r_v27": 0.05,
            "robust_min_trades_v27": 220,
            "robust_worst_drawdown_v27": 0.04,
        },
    ]
    winner = select_development_winner(rows)
    assert winner is not None
    assert winner["strategy"] == "BETTER_CI"


def test_selection_is_deterministic_on_exact_metric_tie():
    base = {
        "timeframe": "4h",
        "development_eligible_v27": True,
        "robust_floor_block_ci_low_v27": 0.0,
        "robust_floor_breadth_v27": 0.6,
        "robust_floor_profit_factor_v27": 1.1,
        "robust_floor_expectancy_r_v27": 0.1,
        "robust_min_trades_v27": 250,
        "robust_worst_drawdown_v27": 0.04,
    }
    winner = select_development_winner([
        {**base, "strategy": "ZZZ"},
        {**base, "strategy": "AAA"},
    ])
    assert winner is not None
    assert winner["strategy"] == "AAA"


def test_kucoin_holdout_uses_full_unchanged_gate():
    good = venue()
    assert holdout_failures(good) == []

    failures = holdout_failures(venue(trades=199, pf=1.0, exp=-0.1, dd=-0.051, breadth=0.5, ci=-0.0001))
    assert failures == ["TRADES", "PROFIT_FACTOR", "EXPECTANCY", "BREADTH", "MAX_DRAWDOWN", "BLOCK_CI"]


def test_holdout_hard_dd_rejects_immediately_even_before_200_trades():
    winner = {"strategy": "LOCKED", "timeframe": "4h"}
    decision = v27_decision(winner, venue(trades=35, dd=-0.051))
    assert decision["decision"] == "ROBUST_CANDIDATE_REJECTED_HOLDOUT"
    assert decision["live_execution_authorized"] is False
    assert decision["forward_paper_candidate_authorized"] is False


def test_holdout_pass_still_requires_new_temporal_oos():
    winner = {"strategy": "LOCKED", "timeframe": "4h"}
    decision = v27_decision(winner, venue())
    assert decision["decision"] == "ROBUST_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
    assert decision["forward_paper_candidate_authorized"] is False
    assert decision["live_execution_authorized"] is False


def test_no_development_candidate_never_uses_holdout():
    decision = v27_decision(None, None, policy=V27RobustnessPolicy())
    assert decision["decision"] == "NO_ROBUST_DEVELOPMENT_CANDIDATE"
    assert decision["holdout_used"] is False
