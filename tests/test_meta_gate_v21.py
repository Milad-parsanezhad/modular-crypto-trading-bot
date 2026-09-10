import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from research_bot.meta_gate_v21 import MetaLabelGate


def _write_manifest(path):
    (path / "dataset_manifest.json").write_text(json.dumps({
        "numeric_features": ["f_x"],
        "categorical_features": ["strategy", "family", "timeframe", "symbol", "side"],
    }), encoding="utf-8")


def test_gate_fails_closed_when_champion_not_promoted(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "champion.json").write_text(json.dumps({
        "decision": "NO_ML_MODEL_PROMOTED", "champion": "logistic",
        "threshold": 0.5, "live_execution_authorized": False,
    }), encoding="utf-8")
    gate = MetaLabelGate(tmp_path)
    result = gate.evaluate({"f_x": 1.0, "strategy": "S", "family": "F", "timeframe": "4h", "symbol": "BTC/USDT", "side": 1})
    assert gate.enabled is False
    assert result.approved_for_paper is False
    assert result.live_execution_authorized is False


def test_gate_rejects_unsafe_live_metadata(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "champion.json").write_text(json.dumps({
        "decision": "FORWARD_PAPER_META_CANDIDATE", "champion": "logistic",
        "threshold": 0.5, "live_execution_authorized": True,
    }), encoding="utf-8")
    gate = MetaLabelGate(tmp_path)
    assert gate.enabled is False
    assert gate.reason == "UNSAFE_METADATA_LIVE_FLAG"


def test_promoted_gate_scores_one_event_but_never_authorizes_live(tmp_path):
    _write_manifest(tmp_path)
    models = tmp_path / "models"; models.mkdir()
    x = pd.DataFrame({"f_x": [-2.0, -1.0, 1.0, 2.0], "strategy": ["S"] * 4, "family": ["F"] * 4, "timeframe": ["4h"] * 4, "symbol": ["BTC/USDT"] * 4, "side": [1] * 4})
    y = np.array([0, 0, 1, 1])
    # This small fixture intentionally uses only f_x; categorical columns are
    # accepted by the runtime event but the saved pipeline selects the numeric input.
    model = Pipeline([("model", LogisticRegression(random_state=314))])
    model.fit(x[["f_x"]], y)
    # Wrapper so the fixture pipeline accepts the exact manifest columns.
    class ManifestFixtureModel:
        def __init__(self, inner): self.inner = inner
        def predict_proba(self, frame): return self.inner.predict_proba(frame[["f_x"]])
    joblib.dump(ManifestFixtureModel(model), models / "classifier_logistic.joblib")
    (tmp_path / "champion.json").write_text(json.dumps({
        "decision": "FORWARD_PAPER_META_CANDIDATE", "champion": "logistic",
        "threshold": 0.5, "live_execution_authorized": False,
    }), encoding="utf-8")
    gate = MetaLabelGate(tmp_path)
    result = gate.evaluate({"f_x": 2.0, "strategy": "S", "family": "F", "timeframe": "4h", "symbol": "BTC/USDT", "side": 1})
    assert gate.enabled is True
    assert result.score is not None
    assert result.approved_for_paper is True
    assert result.live_execution_authorized is False
