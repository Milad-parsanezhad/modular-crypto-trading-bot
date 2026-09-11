from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.asymmetric_meta_v35 import (
    DEVELOPMENT_VENUES_V35,
    RESERVED_HOLDOUT_VENUE_V35,
    TOTAL_EFFECTIVE_TRIALS_V35,
    V35_CANDIDATES,
    attach_event_context_v35,
    causal_empirical_bayes_meta_score_v35,
    preregistration_manifest_v35,
    screen_three_venues_v35,
    select_candidate_events_v35,
    v35_decision,
)


def test_v35_preregistration_contract_is_frozen_and_kraken_sealed():
    manifest = preregistration_manifest_v35()
    assert len(V35_CANDIDATES) == 6
    assert TOTAL_EFFECTIVE_TRIALS_V35 == 120
    assert tuple(manifest["development_venues"]) == DEVELOPMENT_VENUES_V35
    assert RESERVED_HOLDOUT_VENUE_V35 == "kraken"
    assert manifest["reserved_holdout_venue"] == "kraken"
    assert manifest["kraken_touched"] is False
    assert manifest["threshold_relaxation"] is False
    assert manifest["rejected_candidate_retuning"] is False
    assert manifest["paper_replacement_authorized"] is False
    assert manifest["live_execution_authorized"] is False
    assert manifest["strict_meta_availability"] == "historical exit_time < current signal_time"


def test_attach_event_context_preserves_actual_regime_not_side_proxy():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    t1 = pd.Timestamp("2026-01-02", tz="UTC")
    events = pd.DataFrame({
        "signal_time": [t0, t1],
        "entry_time": [t0 + pd.Timedelta(days=1), t1 + pd.Timedelta(days=1)],
        "exit_time": [t0 + pd.Timedelta(days=2), t1 + pd.Timedelta(days=2)],
        "symbol": ["AAA/USDT", "AAA/USDT"],
        "side": [-1, -1],
        "r_multiple": [1.0, -1.0],
    })
    context = pd.DataFrame({
        "timestamp": [t0, t1],
        "symbol": ["AAA/USDT", "AAA/USDT"],
        "rank20_v35": [0.2, 0.3],
        "rank60_v35": [0.25, 0.35],
        "rs_score_v35": [0.225, 0.325],
        "dispersion_ratio_v35": [0.8, 1.2],
        "market_regime_v35": [-1, 0],
    })
    out = attach_event_context_v35(events, context)
    assert out["regime_group_v35"].tolist() == [-1, 0]
    assert out["side"].tolist() == [-1, -1]


def _meta_panel(n: int = 26) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    for i in range(n):
        signal = start + pd.Timedelta(days=i)
        rows.append({
            "signal_time": signal,
            "entry_time": signal + pd.Timedelta(hours=1),
            "exit_time": signal + pd.Timedelta(days=1),
            "symbol": f"S{i % 3}/USDT",
            "side": 1,
            "r_multiple": 1.0 if i % 2 == 0 else -1.0,
            "rs_bucket_v35": 2.0,
            "dispersion_bucket_v35": 0,
            "regime_group_v35": 1,
            "rs_score_v35": 0.8,
        })
    return pd.DataFrame(rows)


def test_meta_score_excludes_same_timestamp_exit_strictly():
    panel = _meta_panel(26)
    base = causal_empirical_bayes_meta_score_v35(panel)
    target_signal = panel.iloc[25]["signal_time"]
    base_score = float(base.loc[base["signal_time"].eq(target_signal), "meta_score_v35"].iloc[0])

    equal_exit_mutation = panel.copy()
    # Event 24 exits exactly at event 25's signal time and therefore must be unavailable.
    assert equal_exit_mutation.iloc[24]["exit_time"] == target_signal
    equal_exit_mutation.loc[24, "r_multiple"] = -99.0
    equal_scored = causal_empirical_bayes_meta_score_v35(equal_exit_mutation)
    equal_score = float(equal_scored.loc[equal_scored["signal_time"].eq(target_signal), "meta_score_v35"].iloc[0])
    assert np.isclose(base_score, equal_score)

    strictly_prior_mutation = panel.copy()
    assert strictly_prior_mutation.iloc[23]["exit_time"] < target_signal
    strictly_prior_mutation.loc[23, "r_multiple"] = 99.0
    prior_scored = causal_empirical_bayes_meta_score_v35(strictly_prior_mutation)
    prior_score = float(prior_scored.loc[prior_scored["signal_time"].eq(target_signal), "meta_score_v35"].iloc[0])
    assert not np.isclose(base_score, prior_score)


def test_side_specific_rank_and_meta_selection_is_asymmetric():
    candidate = V35_CANDIDATES[0]
    t = pd.date_range("2026-01-01", periods=4, tz="UTC")
    events = pd.DataFrame({
        "entry_time": t,
        "side": [1, 1, -1, -1],
        "rs_score_v35": [0.60, 0.50, 0.40, 0.50],
        "meta_score_v35": [0.60, 0.60, 0.60, 0.60],
        "meta_history_n_v35": [25, 25, 25, 25],
        "symbol": ["A", "B", "C", "D"],
    })
    selected = select_candidate_events_v35(events, candidate)
    assert selected["symbol"].tolist() == ["A", "C"]


def _passing_metrics() -> dict:
    return {
        "trades": 220,
        "profit_factor": 1.10,
        "expectancy_r": 0.10,
        "positive_asset_fraction": 0.65,
        "max_drawdown": -0.04,
        "block_ci_low": 0.0001,
    }


def test_three_venue_gate_is_unchanged_and_ci_is_mandatory():
    metrics = {venue: _passing_metrics() for venue in DEVELOPMENT_VENUES_V35}
    passed = screen_three_venues_v35(metrics)
    assert passed["development_eligible_v35"] is True

    failed = {venue: dict(values) for venue, values in metrics.items()}
    failed["okx_consumed"]["block_ci_low"] = -1e-6
    rejected = screen_three_venues_v35(failed)
    assert rejected["development_eligible_v35"] is False


def test_no_winner_keeps_kraken_untouched_and_all_execution_fail_closed():
    decision = v35_decision(None)
    assert decision["decision"] == "NO_V35_ROBUST_DEVELOPMENT_CANDIDATE"
    assert decision["winner"] is None
    assert decision["reserved_holdout_venue"] == "kraken"
    assert decision["kraken_touched"] is False
    assert decision["forward_paper_candidate_authorized"] is False
    assert decision["paper_replacement_authorized"] is False
    assert decision["live_execution_authorized"] is False
