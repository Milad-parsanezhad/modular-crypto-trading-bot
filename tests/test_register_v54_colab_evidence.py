import argparse
import json
from pathlib import Path

import pytest

from scripts.register_v54_colab_evidence import register


def make_args(tmp_path: Path, status: dict, **overrides):
    status_path = tmp_path / "run_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    values = {
        "run_status": status_path,
        "tests": None,
        "log": None,
        "artifact": [],
        "output": tmp_path / "evidence",
        "run_id": "fixed-run",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def safe_status():
    return {
        "status": "BLOCKED_AT_COINEX_CONNECTIVITY",
        "started_at": "2026-09-14T12:00:00Z",
        "scientific_promotion_authorized": False,
        "live_execution": False,
        "paper_execution": False,
    }


def test_registers_append_only_manifest(tmp_path):
    destination = register(make_args(tmp_path, safe_status()))
    manifest = json.loads(
        (destination / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    assert destination.name == "fixed-run"
    assert manifest["research_only"] is True
    assert manifest["live_execution"] is False
    assert manifest["files"][0]["path"] == "run_status.json"

    with pytest.raises(FileExistsError):
        register(make_args(tmp_path, safe_status()))


def test_rejects_live_execution_evidence(tmp_path):
    status = safe_status()
    status["LIVE_EXECUTION"] = True

    with pytest.raises(ValueError, match="LIVE_EXECUTION"):
        register(make_args(tmp_path, status))


def test_rejects_log_containing_secret(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text("API_KEY=abcdef1234567890\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Potential secret"):
        register(make_args(tmp_path, safe_status(), log=log))
