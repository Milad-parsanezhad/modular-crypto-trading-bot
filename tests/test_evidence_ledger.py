import json

import pytest

from research_bot.evidence_ledger import validate_ledger, validate_record


def test_repository_evidence_ledger_is_valid():
    assert validate_ledger()


def test_execution_flags_fail_closed(tmp_path):
    record = json.loads(validate_ledger()[0].read_text(encoding="utf-8"))
    record["safety"]["live_execution"] = True
    with pytest.raises(ValueError, match="safety"):
        validate_record(record, tmp_path / "bad.json")
