from __future__ import annotations

import pytest

from research_bot.evidence_governance_v29 import (
    V27_PASS,
    evidence_record,
    governance_bundle,
    hash_chain,
    sha256_payload,
    validate_v27_v28_chain,
)


def v27_payload(state: str = V27_PASS) -> dict:
    return {
        "version": "v0.27",
        "decision": state,
        "winner": "H4_TEST" if state == V27_PASS else None,
        "timeframe": "4h" if state == V27_PASS else None,
        "holdout_used": state == V27_PASS,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "holdout_winner_reselection": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def v28_payload(state: str = "FRESH_TEMPORAL_OOS_ACCUMULATING") -> dict:
    return {
        "version": "v0.28",
        "decision": state,
        "winner": "H4_TEST",
        "timeframe": "4h",
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "winner_reselection": False,
        "historical_test_recycling": False,
        "forward_paper_candidate_authorized": state == "FORWARD_PAPER_CANDIDATE",
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def test_canonical_hash_is_order_invariant() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})


def test_v27_pass_without_v28_is_awaiting_temporal_evidence() -> None:
    out = validate_v27_v28_chain(v27_payload(), None)
    assert out["promotion_state"] == "AWAITING_V28"
    assert out["forward_paper_candidate_authorized"] is False
    assert out["live_execution_authorized"] is False


def test_v27_nonpass_can_only_block_v28() -> None:
    v27 = v27_payload("ROBUST_CANDIDATE_REJECTED_HOLDOUT")
    blocked = {
        **v28_payload("BLOCKED_BY_V27"),
        "winner": None,
        "timeframe": None,
    }
    out = validate_v27_v28_chain(v27, blocked)
    assert out["promotion_state"] == "STOPPED_AT_V28"

    illegal = {
        **v28_payload("FORWARD_PAPER_CANDIDATE"),
        "winner": None,
        "timeframe": None,
    }
    with pytest.raises(ValueError, match="only transition"):
        validate_v27_v28_chain(v27, illegal)


def test_winner_drift_is_rejected() -> None:
    changed = {**v28_payload(), "winner": "OTHER"}
    with pytest.raises(ValueError, match="winner/timeframe drifted"):
        validate_v27_v28_chain(v27_payload(), changed)


def test_live_authorization_is_rejected_even_after_forward_pass() -> None:
    bad = {**v28_payload("FORWARD_PAPER_CANDIDATE"), "live_execution_authorized": True}
    with pytest.raises(ValueError, match="live execution"):
        validate_v27_v28_chain(v27_payload(), bad)


def test_forward_candidate_does_not_authorize_paper_replacement_or_live() -> None:
    bundle = governance_bundle(v27_payload(), v28_payload("FORWARD_PAPER_CANDIDATE"))
    assert bundle["decision_chain"]["promotion_state"] == "FORWARD_PAPER_CANDIDATE"
    assert bundle["claim_policy"]["fresh_temporal_oos_passed"] is True
    assert bundle["claim_policy"]["stable_alpha_proven"] is False
    assert bundle["paper_replacement_authorized"] is False
    assert bundle["live_execution_authorized"] is False


def test_hash_chain_is_order_sensitive_and_linked() -> None:
    r27 = evidence_record(stage="holdout", decision_payload=v27_payload(), workflow_run_id=27, commit_sha="a" * 40)
    r28 = evidence_record(stage="fresh_oos", decision_payload=v28_payload(), workflow_run_id=28, commit_sha="b" * 40)
    chain = hash_chain([r27, r28])
    assert len(chain) == 2
    assert chain[0]["previous_chain_sha256"] == "0" * 64
    assert chain[1]["previous_chain_sha256"] == chain[0]["chain_sha256"]
    assert hash_chain([r28, r27])[-1]["chain_sha256"] != chain[-1]["chain_sha256"]
