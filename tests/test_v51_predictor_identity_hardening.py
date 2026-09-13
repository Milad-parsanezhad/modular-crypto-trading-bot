import copy

import pandas as pd
import pytest

from research_bot.overlap_arbitration_v51 import (
    ALLOWED_VENUES_V51,
    N_BLOCKS_V51,
    PROSPECTIVE_START_V51,
    assert_common_candidate_set_v51,
    candidate_identity_digest_v51,
    route_decision_v51,
    validate_venue_block_support_v51,
)
from research_bot.predictor_identity_v51 import (
    PredictorVerificationV51,
    V51_PRODUCTION_FOLD,
    V51_PRODUCTION_SYMBOLS,
    V51_REPAIRED_PROSPECTIVE_START,
    assert_verified_manifest,
    canonical_json_sha256,
    verification_manifest,
)


def _candidate(series_id="a", *, score=0.2):
    entry = pd.Timestamp("2026-09-14T00:00:00Z")
    return {
        "venue": "coinex",
        "symbol": "BTCUSDT",
        "series_id": series_id,
        "signal_time": entry - pd.Timedelta(hours=4),
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=8),
        "expected_r_v47": score,
        "selected_model": True,
    }


def test_repaired_boundary_is_frozen_to_clean_4h_utc_decision():
    assert PROSPECTIVE_START_V51 == pd.Timestamp(V51_REPAIRED_PROSPECTIVE_START)
    assert PROSPECTIVE_START_V51 == pd.Timestamp("2026-09-13T12:00:00Z")
    assert PROSPECTIVE_START_V51.hour % 4 == 0


def test_candidate_digest_is_order_invariant_and_arm_pairing_is_exact():
    a = pd.DataFrame([_candidate("a"), _candidate("b")])
    b = a.iloc[::-1].copy()
    assert candidate_identity_digest_v51(a) == candidate_identity_digest_v51(b)
    assert_common_candidate_set_v51(a, b)
    c = b.copy()
    c.loc[c.index[0], "series_id"] = "changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        assert_common_candidate_set_v51(a, c)


def test_candidate_digest_rejects_duplicate_identity():
    row = _candidate("a")
    with pytest.raises(ValueError, match="duplicates"):
        candidate_identity_digest_v51(pd.DataFrame([row, row]))


def _support_matrix():
    rows = []
    for venue in sorted(ALLOWED_VENUES_V51):
        for block in range(1, N_BLOCKS_V51 + 1):
            rows.append({
                "venue": venue,
                "prospective_block_v51": block,
                "evidence_present": True,
                "conflict_cohorts": 0,
            })
    return pd.DataFrame(rows)


def test_support_matrix_requires_all_venue_block_cells():
    good = _support_matrix()
    validate_venue_block_support_v51(good)
    with pytest.raises(ValueError, match="all frozen venue/block cells"):
        validate_venue_block_support_v51(good.iloc[:-1].copy())
    missing = good.copy()
    missing.loc[0, "evidence_present"] = False
    with pytest.raises(ValueError, match="missing venue/block evidence"):
        validate_venue_block_support_v51(missing)


def test_support_matrix_rejects_duplicates_and_nonfinite_counts():
    good = _support_matrix()
    duplicated = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_venue_block_support_v51(duplicated)
    bad = good.copy()
    bad["conflict_cohorts"] = bad["conflict_cohorts"].astype(float)
    bad.loc[0, "conflict_cohorts"] = float("inf")
    with pytest.raises(ValueError, match="invalid conflict"):
        validate_venue_block_support_v51(bad)


def test_route_fails_closed_on_infinite_metric():
    decision = route_decision_v51(
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


def _valid_manifest():
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
    return verification_manifest(rows)


def _rehash(payload):
    payload = copy.deepcopy(payload)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def test_predictor_verification_manifest_is_fail_closed():
    manifest = _valid_manifest()
    assert manifest["all_verified"] is True
    assert_verified_manifest(manifest)
    bad = dict(manifest)
    bad["all_verified"] = False
    with pytest.raises(ValueError):
        assert_verified_manifest(bad)


def test_manifest_rejects_tamper_even_if_all_verified_remains_true():
    forged = _valid_manifest()
    forged["canonical_v47_head"] = "0" * 40
    with pytest.raises(ValueError, match="manifest field"):
        assert_verified_manifest(forged)


def test_manifest_rejects_rehashed_fake_metrics():
    forged = _valid_manifest()
    forged["verifications"][0]["max_abs_expected_r_error"] = 0.25
    forged["verifications"][0]["passed"] = True
    forged["all_verified"] = True
    forged = _rehash(forged)
    with pytest.raises(ValueError, match="metrics fail closed"):
        assert_verified_manifest(forged)


def test_manifest_rejects_duplicate_symbol_with_valid_hash():
    forged = _valid_manifest()
    forged["verifications"][1]["symbol"] = V51_PRODUCTION_SYMBOLS[0]
    forged = _rehash(forged)
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        assert_verified_manifest(forged)


def test_manifest_rejects_missing_or_extra_row_fields():
    missing = _valid_manifest()
    missing["verifications"][0].pop("mean_abs_expected_r_error")
    missing = _rehash(missing)
    with pytest.raises(ValueError, match="row schema"):
        assert_verified_manifest(missing)

    extra = _valid_manifest()
    extra["verifications"][0]["trusted"] = True
    extra = _rehash(extra)
    with pytest.raises(ValueError, match="row schema"):
        assert_verified_manifest(extra)
