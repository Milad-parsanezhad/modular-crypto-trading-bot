from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaGateDecision:
    status: str
    approved_for_paper: bool
    score: float | None
    threshold: float | None
    champion: str | None
    reason: str
    live_execution_authorized: bool = False


class MetaLabelGate:
    """Load a v0.21 champion and make PAPER-only abstention decisions.

    The loader is deliberately fail-closed. A model artifact is not sufficient:
    champion.json must explicitly say FORWARD_PAPER_META_CANDIDATE and must not
    authorize live execution. This class can never return live authorization.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.champion: str | None = None
        self.threshold: float | None = None
        self.enabled = False
        self.reason = "NOT_LOADED"
        self._load()

    def _load(self) -> None:
        champion_path = self.artifact_dir / "champion.json"
        manifest_path = self.artifact_dir / "dataset_manifest.json"
        if not champion_path.exists() or not manifest_path.exists():
            self.reason = "MISSING_CHAMPION_OR_MANIFEST"
            return
        try:
            champion = json.loads(champion_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.reason = f"INVALID_METADATA:{type(exc).__name__}"
            return
        if bool(champion.get("live_execution_authorized", False)):
            self.reason = "UNSAFE_METADATA_LIVE_FLAG"
            return
        if champion.get("decision") != "FORWARD_PAPER_META_CANDIDATE":
            self.reason = f"NOT_PROMOTED:{champion.get('decision', 'UNKNOWN')}"
            return
        name = champion.get("champion")
        threshold = champion.get("threshold")
        if not name or threshold is None or not np.isfinite(float(threshold)):
            self.reason = "INVALID_CHAMPION_FIELDS"
            return
        model_path = self.artifact_dir / "models" / f"classifier_{name}.joblib"
        if not model_path.exists():
            self.reason = "MISSING_CHAMPION_MODEL"
            return
        try:
            self.model = joblib.load(model_path)
        except Exception as exc:
            self.reason = f"MODEL_LOAD_ERROR:{type(exc).__name__}"
            return
        self.champion = str(name)
        self.threshold = float(threshold)
        self.numeric_features = list(manifest.get("numeric_features", []))
        self.categorical_features = list(manifest.get("categorical_features", []))
        self.required_features = self.numeric_features + self.categorical_features
        if not self.required_features:
            self.model = None
            self.reason = "EMPTY_FEATURE_MANIFEST"
            return
        self.enabled = True
        self.reason = "READY_FORWARD_PAPER_ONLY"

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "champion": self.champion,
            "threshold": self.threshold,
            "reason": self.reason,
            "live_execution_authorized": False,
        }

    def evaluate(self, event: pd.DataFrame | dict) -> MetaGateDecision:
        if not self.enabled or self.model is None or self.threshold is None:
            return MetaGateDecision(
                status="ABSTAIN_FAIL_CLOSED", approved_for_paper=False,
                score=None, threshold=self.threshold, champion=self.champion,
                reason=self.reason, live_execution_authorized=False,
            )
        row = pd.DataFrame([event]) if isinstance(event, dict) else event.copy()
        if len(row) != 1:
            return MetaGateDecision("ABSTAIN_FAIL_CLOSED", False, None, self.threshold, self.champion, "ONE_EVENT_REQUIRED", False)
        missing = [c for c in self.required_features if c not in row.columns]
        if missing:
            return MetaGateDecision("ABSTAIN_FAIL_CLOSED", False, None, self.threshold, self.champion, f"MISSING_FEATURES:{','.join(missing[:10])}", False)
        try:
            x = row[self.required_features]
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(x)
                score = float(p[0, 1] if np.ndim(p) == 2 and p.shape[1] > 1 else np.asarray(p).ravel()[0])
            elif hasattr(self.model, "decision_function"):
                raw = float(np.asarray(self.model.decision_function(x)).ravel()[0])
                score = float(1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30))))
            else:
                score = float(np.asarray(self.model.predict(x)).ravel()[0])
        except Exception as exc:
            return MetaGateDecision("ABSTAIN_FAIL_CLOSED", False, None, self.threshold, self.champion, f"SCORING_ERROR:{type(exc).__name__}", False)
        approved = bool(np.isfinite(score) and score >= self.threshold)
        return MetaGateDecision(
            status="PAPER_ALLOW" if approved else "ABSTAIN_META_FILTER",
            approved_for_paper=approved, score=score, threshold=self.threshold,
            champion=self.champion,
            reason="SCORE_PASSED_FROZEN_THRESHOLD" if approved else "SCORE_BELOW_FROZEN_THRESHOLD",
            live_execution_authorized=False,
        )
