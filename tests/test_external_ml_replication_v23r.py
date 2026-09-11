from __future__ import annotations

from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from research_bot.external_ml_replication_v23r import (
    ExternalReplicationContract,
    FrozenChampion,
    evaluate_frozen_external,
    load_frozen_champion,
)


class ScoreOnlyModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("external replication must never call fit")

    def predict_proba(self, frame):
        x = np.asarray(frame["f_x"], dtype=float)
        p = np.clip(x, 0.0, 1.0)
        return np.c_[1.0 - p, p]


def champion(threshold=0.7):
    schema = json.dumps({"numeric": ["f_x"], "categorical": ["timeframe", "side"]}, sort_keys=True).encode("utf-8")
    return FrozenChampion(
        model=ScoreOnlyModel(), champion_family="score_only", champion_seed=314,
        threshold=threshold, numeric_features=("f_x",), categorical_features=("timeframe", "side"),
        feature_schema_sha256=sha256(schema).hexdigest(),
    )


def external_frame(n=240):
    score = np.linspace(0.0, 1.0, n)
    # High-score rows are strongly favorable but retain deterministic losing
    # observations so profit factor is finite rather than +inf.
    idx = np.arange(n)
    selected_ret = np.where(idx % 5 == 0, -0.002, 0.004)
    ret = np.where(score >= 0.7, selected_ret, -0.003)
    symbols = np.array(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT"])
    return pd.DataFrame({
        "signal_time": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
        "label_end_time": pd.date_range("2025-01-01 08:00", periods=n, freq="4h", tz="UTC"),
        "symbol": symbols[idx % len(symbols)],
        "timeframe": "4h", "side": 1, "f_x": score,
        "label_positive_net": (ret > 0).astype(int), "label_future_net_return": ret,
    })


def test_external_evaluation_never_refits_and_keeps_frozen_threshold():
    c = champion(0.7)
    pred, decision = evaluate_frozen_external(
        c, external_frame(),
        contract=ExternalReplicationContract(min_selected=50, min_symbols=5),
    )
    assert c.model.fit_calls == 0
    assert decision["frozen_threshold"] == 0.7
    assert decision["threshold_tuned_on_external"] is False
    assert decision["model_refit_on_external"] is False
    assert (pred["frozen_threshold"] == 0.7).all()
    assert decision["decision"] == "EXTERNAL_ML_REPLICATION_PASS"
    assert np.isfinite(decision["profit_factor"])


def test_external_result_can_never_authorize_paper_or_live():
    _, decision = evaluate_frozen_external(
        champion(0.7), external_frame(),
        contract=ExternalReplicationContract(min_selected=50, min_symbols=5),
    )
    assert decision["forward_paper_authorized"] is False
    assert decision["paper_replacement_authorized"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["requires_strategy_level_risk_backtest"] is True


def test_external_schema_mismatch_fails_closed():
    frame = external_frame().drop(columns=["f_x"])
    with pytest.raises(ValueError, match="schema mismatch"):
        evaluate_frozen_external(champion(), frame)


def test_load_frozen_champion_rejects_schema_hash_tamper(tmp_path):
    joblib.dump(ScoreOnlyModel(), tmp_path / "validation_champion.joblib")
    manifest = {
        "champion_family": "score_only", "champion_seed": 314,
        "frozen_threshold": 0.7, "numeric_features": ["f_x"],
        "categorical_features": ["timeframe", "side"],
        "feature_schema_sha256": "0" * 64,
        "fit_segment": "development_only", "selection_segment": "validation_only",
        "test_used_for_refit": False, "live_execution_authorized": False,
    }
    (tmp_path / "validation_champion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema hash mismatch"):
        load_frozen_champion(tmp_path)


def test_load_frozen_champion_rejects_test_refit_flag(tmp_path):
    joblib.dump(ScoreOnlyModel(), tmp_path / "validation_champion.joblib")
    numeric, categorical = ["f_x"], ["timeframe", "side"]
    digest = sha256(json.dumps({"numeric": numeric, "categorical": categorical}, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "champion_family": "score_only", "champion_seed": 314,
        "frozen_threshold": 0.7, "numeric_features": numeric,
        "categorical_features": categorical, "feature_schema_sha256": digest,
        "fit_segment": "development_only", "selection_segment": "validation_only",
        "test_used_for_refit": True, "live_execution_authorized": False,
    }
    (tmp_path / "validation_champion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="test contamination"):
        load_frozen_champion(tmp_path)
