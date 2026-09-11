from __future__ import annotations

from research_bot.qualification_v31 import select_v31_winner, v31_decision


def row(name: str, ci: float, *, eligible: bool = True) -> dict:
    return {
        "strategy": name,
        "timeframe": "1d",
        "development_eligible_v27": eligible,
        "robust_floor_block_ci_low_v27": ci,
        "robust_floor_breadth_v27": 0.65,
        "robust_floor_profit_factor_v27": 1.20,
        "robust_floor_expectancy_r_v27": 0.12,
        "robust_min_trades_v27": 250,
        "robust_worst_drawdown_v27": 0.04,
    }


def test_no_winner_keeps_holdout_untouched() -> None:
    assert select_v31_winner([row("A", 0.001, eligible=False)]) is None
    d = v31_decision(None, None)
    assert d["decision"] == "NO_V31_FIREWALL_ROBUST_CANDIDATE"
    assert d["holdout_used"] is False
    assert d["final_holdout_venue"] == "kucoin"
    assert d["live_execution_authorized"] is False


def test_winner_is_selected_without_holdout_information() -> None:
    winner = select_v31_winner([row("A", 0.001), row("B", 0.002)])
    assert winner is not None
    assert winner["strategy"] == "B"


def test_hard_drawdown_rejects_before_minimum_holdout_sample() -> None:
    winner = row("A", 0.002)
    holdout = {
        "trades": 15,
        "profit_factor": 1.4,
        "expectancy_r": 0.2,
        "positive_asset_fraction": 0.8,
        "max_drawdown": -0.051,
        "block_ci_low": 0.001,
    }
    d = v31_decision(winner, holdout)
    assert d["decision"] == "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT"
    assert "MAX_DRAWDOWN" in d["holdout_failures"]


def test_complete_holdout_pass_only_requires_fresh_temporal_oos() -> None:
    winner = row("A", 0.002)
    holdout = {
        "trades": 240,
        "profit_factor": 1.2,
        "expectancy_r": 0.1,
        "positive_asset_fraction": 0.7,
        "max_drawdown": -0.04,
        "block_ci_low": 0.0002,
    }
    d = v31_decision(winner, holdout)
    assert d["decision"] == "V31_FIREWALL_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
    assert d["forward_paper_candidate_authorized"] is False
    assert d["paper_replacement_authorized"] is False
    assert d["live_execution_authorized"] is False
