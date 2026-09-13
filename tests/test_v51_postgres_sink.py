from __future__ import annotations

import pandas as pd
import pytest

from scripts.collect_v51_to_postgres import (
    ineligibility_reason,
    resolve_source_commit,
    row_hash,
)


def _row(**overrides):
    data = {
        "venue": "coinex",
        "symbol": "BTCUSDT",
        "bar_open_time": "2026-09-13T12:00:00+00:00",
        "bar_close_time": "2026-09-13T16:00:00+00:00",
        "first_seen_at": "2026-09-13T16:05:00+00:00",
        "prospective_eligible_v51": True,
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 12.0,
    }
    data.update(overrides)
    return pd.Series(data)


def test_source_commit_fails_closed_when_missing(monkeypatch):
    for key in ("V51_SOURCE_COMMIT", "RAILWAY_GIT_COMMIT_SHA", "GITHUB_SHA"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="provenance fail-closed"):
        resolve_source_commit()


def test_source_commit_accepts_real_sha_and_normalizes_case(monkeypatch):
    sha = "A" * 40
    monkeypatch.setenv("V51_SOURCE_COMMIT", sha)
    assert resolve_source_commit() == sha.lower()


def test_source_commit_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("V51_SOURCE_COMMIT", "UNKNOWN")
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    with pytest.raises(RuntimeError):
        resolve_source_commit()


def test_timely_in_window_row_has_no_ineligibility_reason():
    assert ineligibility_reason(_row()) is None


def test_late_row_is_permanently_labeled_late():
    row = _row(
        first_seen_at="2026-09-13T17:00:01+00:00",
        prospective_eligible_v51=False,
    )
    assert ineligibility_reason(row) == "LATE_FIRST_SEEN_GT_60M"


def test_pre_start_row_is_labeled_context_only():
    row = _row(
        bar_open_time="2026-09-13T04:00:00+00:00",
        bar_close_time="2026-09-13T08:00:00+00:00",
        first_seen_at="2026-09-13T08:05:00+00:00",
        prospective_eligible_v51=False,
    )
    assert ineligibility_reason(row) == "PRE_PROSPECTIVE_START"


def test_hash_changes_when_ohlcv_changes():
    a = row_hash(_row())
    b = row_hash(_row(close=102.01, high=103.01))
    assert a != b
    assert len(a) == 64
