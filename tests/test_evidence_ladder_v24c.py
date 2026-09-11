from __future__ import annotations

import pytest

from research_bot.evidence_ladder_v24c import (
    EvidenceSnapshot,
    EvidenceState,
    MLTrack,
    PlanName,
    assert_evidence_integrity,
    eligible_tracks,
    next_plan,
    promotion_ready,
    protocol_manifest,
)


def test_blocked_external_routes_to_future_time_not_same_test_retune():
    snap = EvidenceSnapshot(
        plan=PlanName.EXTERNAL_UNTOUCHED,
        state=EvidenceState.BLOCKED,
        test_seen=False,
        model_or_threshold_changed_after_test=False,
        source_is_fresh=True,
    )
    assert next_plan(snap) == PlanName.FUTURE_TIME_FORWARD


def test_failed_external_routes_to_research_redesign():
    snap = EvidenceSnapshot(
        plan=PlanName.EXTERNAL_UNTOUCHED,
        state=EvidenceState.FAIL,
        test_seen=True,
        model_or_threshold_changed_after_test=False,
        source_is_fresh=True,
    )
    assert next_plan(snap) == PlanName.RESEARCH_REDESIGN


def test_post_test_retuning_is_forbidden():
    snap = EvidenceSnapshot(
        plan=PlanName.EXTERNAL_UNTOUCHED,
        state=EvidenceState.FAIL,
        test_seen=True,
        model_or_threshold_changed_after_test=True,
        source_is_fresh=True,
    )
    with pytest.raises(ValueError, match="POST_TEST_RETUNING_FORBIDDEN"):
        assert_evidence_integrity(snap)


def test_nonfresh_pass_is_forbidden():
    snap = EvidenceSnapshot(
        plan=PlanName.EXTERNAL_UNTOUCHED,
        state=EvidenceState.PASS,
        test_seen=True,
        model_or_threshold_changed_after_test=False,
        source_is_fresh=False,
    )
    with pytest.raises(ValueError, match="PASS_REQUIRES_FRESH_EVIDENCE"):
        assert_evidence_integrity(snap)


def test_live_is_fail_closed():
    snap = EvidenceSnapshot(
        plan=PlanName.EXTERNAL_UNTOUCHED,
        state=EvidenceState.PASS,
        test_seen=True,
        model_or_threshold_changed_after_test=False,
        source_is_fresh=True,
        live_execution_authorized=True,
    )
    with pytest.raises(ValueError, match="V24C_LIVE_MUST_REMAIN_FAIL_CLOSED"):
        assert_evidence_integrity(snap)


def test_rl_requires_portfolio_allocator_pass():
    status = {
        MLTrack.TABULAR_META: EvidenceState.PASS,
        MLTrack.TEMPORAL_SEQUENCE: EvidenceState.PASS,
        MLTrack.VISION_MULTIMODAL: EvidenceState.PASS,
        MLTrack.PORTFOLIO_ALLOCATOR: EvidenceState.FAIL,
    }
    assert MLTrack.REINFORCEMENT_LEARNING not in eligible_tracks(status)
    status[MLTrack.PORTFOLIO_ALLOCATOR] = EvidenceState.PASS
    assert MLTrack.REINFORCEMENT_LEARNING in eligible_tracks(status)


def test_ensemble_requires_independent_upstream_tracks():
    status = {
        MLTrack.TABULAR_META: EvidenceState.PASS,
        MLTrack.TEMPORAL_SEQUENCE: EvidenceState.PASS,
        MLTrack.VISION_MULTIMODAL: EvidenceState.FAIL,
    }
    assert MLTrack.ENSEMBLE_STACK not in eligible_tracks(status)
    status[MLTrack.VISION_MULTIMODAL] = EvidenceState.PASS
    assert MLTrack.ENSEMBLE_STACK in eligible_tracks(status)


def test_promotion_gate_is_strict_and_fresh():
    valid = {
        "trades": 250,
        "profit_factor": 1.20,
        "mean_r": 0.12,
        "mtm_drawdown": -0.035,
        "positive_symbol_fraction": 0.75,
        "bootstrap_uplift_low": 0.0001,
        "pbo": 0.30,
        "dsr_probability": 0.97,
        "stress_36bps_pass": True,
        "fresh_external_or_forward": True,
    }
    ok, failures = promotion_ready(validation=valid)
    assert ok and not failures
    valid["fresh_external_or_forward"] = False
    ok, failures = promotion_ready(validation=valid)
    assert not ok and "FRESH_EVIDENCE" in failures


def test_protocol_manifest_never_authorizes_live():
    m = protocol_manifest()
    assert m["forward_paper_authorized"] is False
    assert m["live_execution_authorized"] is False
    assert len(m["plans"]) == 4
    assert len(m["tracks"]) >= 7
