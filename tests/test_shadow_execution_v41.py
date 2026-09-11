from __future__ import annotations

import pytest

from research_bot.shadow_execution_v41 import (
    LIVE_EXECUTION,
    LiveExecutionForbidden,
    OrderIntentV41,
    RiskRejected,
    ShadowExecutionEngineV41,
    decision_v41,
    preregistration_manifest_v41,
)


def intent():
    return OrderIntentV41('BTC/USDT','buy',0.01,50000.0,'2026-09-11T20:00:00Z','V41_TEST')


def test_live_is_structurally_forbidden():
    assert LIVE_EXECUTION is False
    e=ShadowExecutionEngineV41()
    with pytest.raises(LiveExecutionForbidden):
        e.submit_live(intent())


def test_idempotency_rejects_duplicate():
    e=ShadowExecutionEngineV41()
    e.submit_shadow(intent())
    with pytest.raises(RiskRejected, match='DUPLICATE_CLIENT_ORDER_ID'):
        e.submit_shadow(intent())


def test_kill_switch_blocks_new_orders():
    e=ShadowExecutionEngineV41(); e.activate_kill_switch('test')
    with pytest.raises(RiskRejected, match='KILL_SWITCH_ACTIVE'):
        e.submit_shadow(intent())


def test_reconciliation_mismatch_activates_kill_switch():
    e=ShadowExecutionEngineV41(); e.submit_paper(intent())
    r=e.reconcile({})
    assert r['ok'] is False
    assert e.kill_switch is True


def test_manifest_and_decision_never_authorize_live():
    m=preregistration_manifest_v41(); assert m['live_execution'] is False
    d=decision_v41(True,True)
    assert d['paper_path_ready'] is True
    assert d['live_execution_authorized'] is False
    assert d['real_order_transmission'] is False
