import json

import pytest

from research_bot.trial_registry_v17 import TrialRecord, append_trial, load_trials, validate_trial


def _record(**overrides):
    payload = dict(
        trial_id="trial-001",
        git_commit="abc123",
        dataset_fingerprint="data-sha",
        feature_family="price_ichimoku",
        model_family="logistic",
        hyperparameters={"C": 1.0},
        random_seed=42,
        forecast_horizon="4h",
        decision_rule="top_quartile_long_only",
        cost_model={"one_way_bps": 12},
        train_window="2025-01-01/2025-12-31",
        validation_window="2026-01-01/2026-03-31",
        holdout_window="2026-04-01/2026-06-30",
        return_path_artifact="artifacts/trial-001-returns.csv",
        status="VALIDATED_OOS",
    )
    payload.update(overrides)
    return TrialRecord(**payload)


def test_registry_is_deterministic_and_deduplicated(tmp_path):
    path = tmp_path / "trials.jsonl"
    record = _record()
    fp1 = append_trial(path, record)
    fp2 = append_trial(path, record)
    assert fp1 == fp2
    rows = load_trials(path)
    assert len(rows) == 1
    assert rows[0]["fingerprint"] == fp1
    assert rows[0]["status"] == "VALIDATED_OOS"


def test_advanced_label_requires_holdout():
    with pytest.raises(ValueError):
        validate_trial(_record(status="SEARCH_AWARE_SURVIVOR", holdout_window=None))


def test_unsupported_status_fails_closed():
    with pytest.raises(ValueError):
        validate_trial(_record(status="LIVE_PROFITABLE"))
