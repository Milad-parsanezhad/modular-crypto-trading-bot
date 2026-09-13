import pandas as pd
import pytest

from research_bot.predictor_identity_v51 import (
    PredictorVerificationV51,
    V51_PRODUCTION_FOLD,
    V51_PRODUCTION_SYMBOLS,
    V51_REPAIRED_PROSPECTIVE_START,
    assert_verified_manifest,
    verification_manifest,
)
from research_bot.v51_integrity import (
    AMENDED_PROSPECTIVE_START_V51,
    assert_common_candidate_set_v51,
    candidate_identity_digest_v51,
    safe_route_decision_v51,
    validate_venue_block_support_v51,
)


def _candidate(series_id: str):
    entry = pd.Timestamp("2026-09-14T00:00:00Z")
    return {
        "venue": "coinex",
        "symbol": "BTCUSDT",
        "series_id": series_id,
        "signal_time": entry - pd.Timedelta(hours=4),
        "entry_time": entry,
    }


def test_amended_boundary_is_clean_four_hour_utc_boundary():
    assert AMENDED_PROSPECTIVE_START_V51 == pd.Timestamp(V51_REPAIRED_PROSPECTIVE_START)
    assert AMENDED_PROSPECTIVE_START_V51 == pd.Timestamp("2026-09-13T12:00:00Z")
    assert AMENDED_PROSPECTIVE_START_V51.hour % 4 == 0


def test_candidate_identity_digest_is_order_invariant_and_paired():
    a = pd.DataFrame([_candidate("a"), _candidate("b")])
    b = a.iloc[::-1].copy()
    assert candidate_identity_digest_v51(a) == candidate_identity_digest_v51(b)
    assert_common_candidate_set_v51(a, b)
    c = b.copy()
    c.loc[c.index[0], "series_id"] = "changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        assert_common_candidate_set_v51(a, c)


def _support_matrix():
    rows = []
    for venue in ("coinex", "okx", "kucoin"):
        for block in range(1, 6):
            rows.append({
                "venue": venue,
                "prospective_block_v51": block,
                "evidence_present": True,
                "conflict_cohorts": 0,
            })
    return pd.DataFrame(rows)


def test_support_matrix_requires_all_15_cells_and_explicit_evidence():
    good = _support_matrix()
    validate_venue_block_support_v51(good)
    with pytest.raises(ValueError, match="all frozen venue/block cells"):
        validate_venue_block_support_v51(good.iloc[:-1].copy())
    absent = good.copy()
    absent.loc[0, "evidence_present"] = False
    with pytest.raises(ValueError, match="missing venue/block evidence"):
        validate_venue_block_support_v51(absent)


def test_safe_router_fails_closed_on_infinite_metric():
    decision = safe_route_decision_v51(
        conflict_counts=[20] * 5,
        expectancy_deltas=[0.01] * 5,
        aggregate_expectancy_a0=0.01,
        aggregate_expectancy_a1=float("inf"),
        profit_factor_a0=1.0,
        profit_factor_a1=1.1,
        stress_profit_factor_a0=0.95,
        stress_profit_factor_a1=1.0,
        worst_drawdown_a1=-0.04,
        causal_checks_passed=True,
    )
    assert decision == "V51_ARBITRATION_NOT_SUPPORTED"


def test_predictor_manifest_is_fail_closed():
    rows = [
        PredictorVerificationV51(
            symbol=symbol,
            fold=V51_PRODUCTION_FOLD,
            rows=100,
            canonical_temperature=1.5,
            max_abs_expected_r_error=1e-8,
            mean_abs_expected_r_error=1e-9,
            selected_model_match_fraction=1.0,
            identity_match=True,
        )
        for symbol in V51_PRODUCTION_SYMBOLS
    ]
    manifest = verification_manifest(rows)
    assert_verified_manifest(manifest)
    broken = dict(manifest)
    broken["all_verified"] = False
    with pytest.raises(ValueError, match="not fully verified"):
        assert_verified_manifest(broken)
