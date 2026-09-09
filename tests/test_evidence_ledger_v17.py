from research_bot.evidence_ledger_v17 import CLAIM_POLICY, MILESTONES, ledger, thesis_conclusion, validate_ledger


def test_milestone_sequence_and_fail_closed_execution():
    validate_ledger()
    assert [x.version for x in MILESTONES] == [f"v0.{n}" for n in range(10, 17)]
    assert all(x.execution_authorized is False for x in MILESTONES)


def test_claim_policy_does_not_overstate_science():
    assert CLAIM_POLICY["engineering_operational"] is True
    assert CLAIM_POLICY["forward_paper_operational"] is True
    assert CLAIM_POLICY["profitable_strategy"] is False
    assert CLAIM_POLICY["statistically_significant_alpha"] is False
    assert CLAIM_POLICY["stable_sharpe"] is False
    assert CLAIM_POLICY["live_ready"] is False
    assert CLAIM_POLICY["real_money_execution_authorized"] is False


def test_ledger_serializes_all_stages():
    rows = ledger()
    assert len(rows) == 7
    assert rows[0]["decision"] == "NO_MODEL_PROMOTED"
    assert rows[2]["decision"] == "NO_INCREMENTAL_DERIVATIVES_EVIDENCE"
    assert rows[-1]["decision"] == "INSUFFICIENT_FORWARD_SAMPLE"


def test_thesis_conclusion_is_conservative():
    text = thesis_conclusion().lower()
    assert "operational" in text
    assert "does not justify" in text
    assert "profitability" in text
    assert "live readiness" in text
