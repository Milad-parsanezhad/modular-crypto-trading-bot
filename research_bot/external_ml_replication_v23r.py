from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from research_bot.ml_framework_v23r import economic_metrics


@dataclass(frozen=True)
class ExternalReplicationContract:
    min_selected: int = 100
    min_profit_factor: float = 1.05
    require_positive_mean_net_return: bool = True
    min_symbols: int = 5
    live_execution_authorized: bool = False


@dataclass(frozen=True)
class FrozenChampion:
    model: object
    champion_family: str
    champion_seed: int
    threshold: float
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    feature_schema_sha256: str

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features


def _schema_sha(numeric: tuple[str, ...], categorical: tuple[str, ...]) -> str:
    payload = json.dumps({"numeric": list(numeric), "categorical": list(categorical)}, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def load_frozen_champion(internal_artifact_dir: str | Path) -> FrozenChampion:
    root = Path(internal_artifact_dir)
    manifest_path = root / "validation_champion_manifest.json"
    model_path = root / "validation_champion.joblib"
    if not manifest_path.exists() or not model_path.exists():
        raise FileNotFoundError("validation-frozen champion artifact is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("live_execution_authorized") is not False:
        raise RuntimeError("champion manifest is not explicitly fail-closed for LIVE")
    if manifest.get("fit_segment") != "development_only":
        raise RuntimeError("champion was not fitted on development_only")
    if manifest.get("selection_segment") != "validation_only":
        raise RuntimeError("champion was not selected on validation_only")
    if manifest.get("test_used_for_refit") is not False:
        raise RuntimeError("test contamination flag in champion manifest")
    threshold = float(manifest["frozen_threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("frozen threshold is not finite")
    numeric = tuple(manifest.get("numeric_features", []))
    categorical = tuple(manifest.get("categorical_features", []))
    if not numeric:
        raise RuntimeError("frozen feature schema has no numeric features")
    expected_schema_sha = str(manifest.get("feature_schema_sha256", ""))
    actual_schema_sha = _schema_sha(numeric, categorical)
    if not expected_schema_sha or actual_schema_sha != expected_schema_sha:
        raise RuntimeError(f"frozen feature schema hash mismatch expected={expected_schema_sha} actual={actual_schema_sha}")
    model = joblib.load(model_path)
    return FrozenChampion(
        model=model,
        champion_family=str(manifest["champion_family"]),
        champion_seed=int(manifest["champion_seed"]),
        threshold=threshold,
        numeric_features=numeric,
        categorical_features=categorical,
        feature_schema_sha256=expected_schema_sha,
    )


def assert_external_schema(frame: pd.DataFrame, champion: FrozenChampion) -> None:
    missing = [c for c in champion.features if c not in frame.columns]
    if missing:
        raise ValueError(f"external feature schema mismatch; missing={missing}")
    if "label_future_net_return" not in frame.columns:
        raise ValueError("external holdout is missing cost-aware return labels")
    if "symbol" not in frame.columns:
        raise ValueError("external holdout is missing symbol provenance")


def probability_score_frozen(champion: FrozenChampion, frame: pd.DataFrame) -> np.ndarray:
    """Score only. This module intentionally exposes no fit/refit API."""
    assert_external_schema(frame, champion)
    x = frame[list(champion.features)]
    model = champion.model
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(x), dtype=float)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    pred = np.asarray(model.predict(x), dtype=float).ravel()
    return np.clip(pred, 0.0, 1.0)


def evaluate_frozen_external(
    champion: FrozenChampion,
    frame: pd.DataFrame,
    *,
    contract: ExternalReplicationContract | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply exactly the frozen internal threshold to an external-venue holdout.

    No external threshold search, model selection, refit, calibration or feature
    selection is permitted here. A pass is replication evidence only; strategy
    risk integration and fresh forward PAPER remain separate downstream gates.
    """
    contract = contract or ExternalReplicationContract()
    assert_external_schema(frame, champion)
    score = probability_score_frozen(champion, frame)
    selected = score >= champion.threshold
    metrics = economic_metrics(frame["label_future_net_return"], selected)
    selected_symbols = int(frame.loc[selected, "symbol"].nunique()) if selected.any() else 0
    passed = bool(
        int(metrics["selected"]) >= contract.min_selected
        and selected_symbols >= contract.min_symbols
        and np.isfinite(float(metrics["profit_factor"]))
        and float(metrics["profit_factor"]) >= contract.min_profit_factor
        and (
            not contract.require_positive_mean_net_return
            or (np.isfinite(float(metrics["mean_net_return"])) and float(metrics["mean_net_return"]) > 0)
        )
    )
    pred = frame[[c for c in ["signal_time", "label_end_time", "symbol", "timeframe", "side", "label_positive_net", "label_future_net_return"] if c in frame.columns]].copy()
    pred["score"] = score
    pred["frozen_threshold"] = champion.threshold
    pred["selected"] = selected
    decision = {
        "decision": "EXTERNAL_ML_REPLICATION_PASS" if passed else "EXTERNAL_ML_REPLICATION_FAIL",
        "external_pass": passed,
        "champion_family": champion.champion_family,
        "champion_seed": champion.champion_seed,
        "frozen_threshold": champion.threshold,
        "threshold_tuned_on_external": False,
        "model_refit_on_external": False,
        "feature_schema_sha256": champion.feature_schema_sha256,
        "external_rows": int(len(frame)),
        "external_symbols": int(frame["symbol"].nunique()),
        "selected_symbols": selected_symbols,
        **metrics,
        "requires_strategy_level_risk_backtest": True,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    return pred, decision
