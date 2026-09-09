from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    git_commit: str
    dataset_fingerprint: str
    feature_family: str
    model_family: str
    hyperparameters: dict[str, Any]
    random_seed: int | None
    forecast_horizon: str
    decision_rule: str
    cost_model: dict[str, Any]
    train_window: str
    validation_window: str
    holdout_window: str | None
    return_path_artifact: str
    status: str

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hyperparameters"] = dict(sorted(self.hyperparameters.items()))
        payload["cost_model"] = dict(sorted(self.cost_model.items()))
        return payload

    def fingerprint(self) -> str:
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()


def validate_trial(record: TrialRecord) -> None:
    required = {
        "trial_id": record.trial_id,
        "git_commit": record.git_commit,
        "dataset_fingerprint": record.dataset_fingerprint,
        "feature_family": record.feature_family,
        "model_family": record.model_family,
        "forecast_horizon": record.forecast_horizon,
        "decision_rule": record.decision_rule,
        "train_window": record.train_window,
        "validation_window": record.validation_window,
        "return_path_artifact": record.return_path_artifact,
        "status": record.status,
    }
    missing = [k for k, v in required.items() if not str(v).strip()]
    if missing:
        raise ValueError(f"missing trial fields: {missing}")
    if record.status not in {
        "HYPOTHESIS",
        "DISCOVERY_CANDIDATE",
        "VALIDATED_OOS",
        "REJECTED",
        "EXTERNAL_HOLDOUT_NEGATIVE_RESULT",
        "SEARCH_AWARE_SURVIVOR",
        "FORWARD_REPLICATED",
    }:
        raise ValueError(f"unsupported trial status: {record.status}")
    if record.status in {"SEARCH_AWARE_SURVIVOR", "FORWARD_REPLICATED"} and not record.holdout_window:
        raise ValueError("advanced evidence labels require an explicit holdout/forward window")


def append_trial(path: str | Path, record: TrialRecord) -> str:
    validate_trial(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fp = record.fingerprint()

    existing_fingerprints: set[str] = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_fingerprints.add(str(row.get("fingerprint") or ""))
    if fp in existing_fingerprints:
        return fp

    row = {"fingerprint": fp, **record.canonical_payload()}
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return fp


def load_trials(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
