from __future__ import annotations

from research_bot.readiness_gate_v42 import decision_v42, preregistration_manifest_v42, readiness_matrix_v42


def _d(decision, **extra):
    x={'decision':decision,'kraken_touched':False}
    x.update(extra); return x


def test_manifest_never_authorizes_real_money():
    m=preregistration_manifest_v42()
    assert m['manual_human_approval_required'] is True
    assert m['live_execution'] is False
    assert m['real_money_trading_authorized'] is False


def test_failed_science_blocks_supervised_paper_even_if_engine_is_valid():
    m=readiness_matrix_v42(
        d38=_d('NO_V38_ROBUST_SOFT_PORTFOLIO_CANDIDATE'),
        d39=_d('NO_V39_TEMPORALLY_STABLE_CANDIDATE'),
        d40=_d('NO_V40_EXECUTION_ROBUST_CANDIDATE'),
        d41=_d('V41_SHADOW_ENGINE_VALIDATED_RESEARCH_BLOCKED',engine_validated=True,paper_path_ready=False,live_execution_authorized=False,real_order_transmission=False),
    )
    d=decision_v42(m)
    assert m['scientific_parent_valid'] is False
    assert d['supervised_paper_ready'] is False
    assert d['real_money_trading_authorized'] is False
    assert d['live_execution_authorized'] is False


def test_even_hypothetical_ready_state_still_requires_human_and_disables_live():
    m=readiness_matrix_v42(
        d38=_d('V38_SOFT_PORTFOLIO_CANDIDATE_LOCKED_FOR_WALK_FORWARD'),
        d39=_d('V39_TEMPORALLY_STABLE_CANDIDATE_LOCKED_FOR_EXECUTION_STRESS'),
        d40=_d('V40_EXECUTION_ROBUST_CANDIDATE_LOCKED_FOR_SHADOW'),
        d41=_d('V41_SUPERVISED_PAPER_PATH_READY',engine_validated=True,paper_path_ready=True,live_execution_authorized=False,real_order_transmission=False),
    )
    d=decision_v42(m)
    assert m['all_controls_pass'] is True
    assert d['supervised_paper_ready'] is True
    assert d['manual_human_approval_required'] is True
    assert d['real_money_trading_authorized'] is False
    assert d['live_execution_authorized'] is False
