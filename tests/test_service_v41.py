from __future__ import annotations

import pytest
from fastapi import HTTPException

from research_bot.service import (
    evaluate_forecast,
    health,
    paper_observations,
    paper_run_once,
    paper_status,
    research_manifest,
    research_status,
    root,
)


def test_service_reports_research_only_and_sealed_holdout() -> None:
    r = root()
    h = health()
    s = research_status()

    assert r["mode"] == "RESEARCH_ONLY"
    assert r["paper_execution"] is False
    assert r["live_execution"] is False
    assert r["kraken_holdout"] == "SEALED"

    assert h["execution_mode"] == "RESEARCH_ONLY"
    assert h["paper_execution"] is False
    assert h["live_execution"] is False
    assert h["kraken_holdout"] == "SEALED"

    assert s["current_experiment"] == "v0.41"
    assert s["reserved_holdout"] == "kraken"
    assert s["kraken_holdout"] == "SEALED"
    assert s["paper_execution"] is False
    assert s["live_execution"] is False


def test_service_v41_manifest_is_fail_closed() -> None:
    m = research_manifest("v41")
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False


def test_paper_and_decision_endpoints_cannot_execute() -> None:
    status = paper_status()
    assert status["status"] == "DISABLED_BY_SCIENTIFIC_GATE"
    assert status["paper_execution_enabled"] is False
    assert paper_observations()["items"] == []

    with pytest.raises(HTTPException) as paper_exc:
        paper_run_once()
    assert paper_exc.value.status_code == 423

    with pytest.raises(HTTPException) as decision_exc:
        evaluate_forecast({"anything": "ignored"})
    assert decision_exc.value.status_code == 423
