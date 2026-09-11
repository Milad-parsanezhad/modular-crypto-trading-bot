from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

import research_bot.frozen_snapshot_v24d as frozen
from research_bot.frozen_snapshot_v24d import (
    FrozenSnapshotContract,
    assert_runtime_compatible,
    file_sha256,
    verify_snapshot_files,
)


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def test_contract_pins_immutable_source_and_never_authorizes_live():
    c = FrozenSnapshotContract()
    assert c.source_run_id == 34578494059
    assert c.source_artifact_id == 10190676664
    assert c.model_sha256 == "ec4a81b7d708b0ffc7a80238668fa1bb34cccdac0408c51e4aff082075a5a22a"
    assert c.dataset_sha256 == "fe1a48a8854b9550283db41cc43821ba635b7a63b07eb48df894dd06f337e338"
    assert c.forward_paper_authorized is False
    assert c.live_execution_authorized is False


def test_verify_snapshot_files_is_byte_exact_and_detects_tamper(tmp_path):
    payloads = {
        "model.bin": b"trusted-model-bytes",
        "dataset.csv": b"a,b\n1,2\n",
        "leaderboard.csv": b"model,score\nx,1\n",
        "decision.json": b'{"decision":"frozen"}',
    }
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    c = replace(
        FrozenSnapshotContract(),
        model_file="model.bin",
        model_sha256=_digest(payloads["model.bin"]),
        dataset_file="dataset.csv",
        dataset_sha256=_digest(payloads["dataset.csv"]),
        leaderboard_file="leaderboard.csv",
        leaderboard_sha256=_digest(payloads["leaderboard.csv"]),
        decision_file="decision.json",
        decision_sha256=_digest(payloads["decision.json"]),
    )
    observed = verify_snapshot_files(tmp_path, c)
    assert observed["model.bin"] == c.model_sha256

    (tmp_path / "dataset.csv").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="FROZEN_SNAPSHOT_SHA_MISMATCH dataset.csv"):
        verify_snapshot_files(tmp_path, c)


def test_file_sha256_is_stable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_runtime_version_mismatch_fails_before_model_load(monkeypatch):
    values = {
        "scikit-learn": "0.0-bad",
        "numpy": "2.5.3",
        "pandas": "3.0.5",
        "joblib": "1.6.0",
    }
    monkeypatch.setattr(frozen, "package_version", lambda name: values[name])
    with pytest.raises(RuntimeError, match="FROZEN_SNAPSHOT_RUNTIME_MISMATCH"):
        assert_runtime_compatible(FrozenSnapshotContract())


def test_runtime_exact_versions_pass(monkeypatch):
    values = {
        "scikit-learn": "1.9.1",
        "numpy": "2.5.3",
        "pandas": "3.0.5",
        "joblib": "1.6.0",
    }
    monkeypatch.setattr(frozen, "package_version", lambda name: values[name])
    assert assert_runtime_compatible(FrozenSnapshotContract()) == values
