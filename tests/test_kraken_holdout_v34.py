from research_bot.kraken_holdout_v34 import holdout_failures, v34_decision


def test_blocked_v33_never_uses_kraken():
    d = v34_decision({"decision":"NO_V33_ROBUST_DEVELOPMENT_CANDIDATE","winner":None,"kraken_touched":False,"threshold_relaxation":False,"base_alpha_retuning":False}, {"winner":None,"kraken_touched":False}, None)
    assert d["decision"] == "BLOCKED_BY_V33"
    assert d["holdout_used"] is False
    assert d["live_execution_authorized"] is False


def test_full_gate_is_unchanged():
    good = {"trades":250,"profit_factor":1.2,"expectancy_r":0.1,"positive_asset_fraction":0.65,"max_drawdown":-0.04,"block_ci_low":0.0001}
    assert holdout_failures(good) == []
    bad = dict(good); bad["positive_asset_fraction"] = 0.55
    assert "BREADTH" in holdout_failures(bad)


def test_pass_still_does_not_authorize_live():
    v33 = {"decision":"V33_BREADTH_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT","winner":"X","timeframe":"1d","kraken_touched":False,"threshold_relaxation":False,"base_alpha_retuning":False}
    lock = {"winner":"X","kraken_touched":False}
    good = {"trades":250,"profit_factor":1.2,"expectancy_r":0.1,"positive_asset_fraction":0.65,"max_drawdown":-0.04,"block_ci_low":0.0001}
    d = v34_decision(v33, lock, good)
    assert d["decision"] == "V34_CANDIDATE_PASSED_KRAKEN_REQUIRES_FRESH_TEMPORAL_OOS"
    assert d["forward_paper_candidate_authorized"] is False
    assert d["live_execution_authorized"] is False
