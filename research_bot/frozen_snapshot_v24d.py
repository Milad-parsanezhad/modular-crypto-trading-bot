from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from research_bot.ml_framework_v23r import probability_score
from research_bot.v24c_external_plan import (
    FROZEN_CANDIDATES_V24C,
    FrozenModelBundle,
)


@dataclass(frozen=True)
class FrozenSnapshotContract:
    """Immutable v0.24b model/data snapshot identity used by v0.24d.

    The snapshot is a trusted GitHub Actions artifact created by the user's own
    v0.24b workflow. v0.24d does not reconstruct, refit, reseed or retune the
    selected estimators. It verifies byte-level identities and loads the exact
    persisted champion objects under the persistence environment recorded by the
    source artifact.
    """

    source_run_id: int = 34578494059
    source_artifact_id: int = 10190676664
    source_artifact_name: str = "v24b-family-portfolio-34578494059"
    source_artifact_digest: str = "0384391db85922309e7b67e0e0481f2cbf8795cf069ee48f146e36ba857525db"

    model_file: str = "family_validation_frozen_champions.joblib"
    model_sha256: str = "ec4a81b7d708b0ffc7a80238668fa1bb34cccdac0408c51e4aff082075a5a22a"
    dataset_file: str = "strategy_event_dataset.csv"
    dataset_sha256: str = "fe1a48a8854b9550283db41cc43821ba635b7a63b07eb48df894dd06f337e338"
    leaderboard_file: str = "family_validation_leaderboard.csv"
    leaderboard_sha256: str = "dadcbbb36491f643388c70d40a7c27222c749fb9647ffb93d4c974fa85d320a8"
    decision_file: str = "decision.json"
    decision_sha256: str = "9daaec9bf3821a16303be5a06f4a706c11f08e03d997c09003c95ea8a952a1a5"

    sklearn_version: str = "1.9.1"
    numpy_version: str = "2.5.3"
    pandas_version: str = "3.0.5"
    joblib_version: str = "1.6.0"

    forward_paper_authorized: bool = False
    live_execution_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


EXPECTED_MODEL_CLASSES = {
    "sgd_logistic": "SGDClassifier",
    "logistic": "LogisticRegression",
}


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_runtime_compatible(contract: FrozenSnapshotContract | None = None) -> dict[str, str]:
    c = contract or FrozenSnapshotContract()
    actual = {
        "scikit-learn": package_version("scikit-learn"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "joblib": package_version("joblib"),
    }
    expected = {
        "scikit-learn": c.sklearn_version,
        "numpy": c.numpy_version,
        "pandas": c.pandas_version,
        "joblib": c.joblib_version,
    }
    mismatch = {k: {"actual": actual[k], "expected": expected[k]} for k in expected if actual[k] != expected[k]}
    if mismatch:
        raise RuntimeError(f"FROZEN_SNAPSHOT_RUNTIME_MISMATCH {json.dumps(mismatch, sort_keys=True)}")
    return actual


def verify_snapshot_files(snapshot_dir: str | Path, contract: FrozenSnapshotContract | None = None) -> dict[str, str]:
    c = contract or FrozenSnapshotContract()
    root = Path(snapshot_dir)
    expected = {
        c.model_file: c.model_sha256,
        c.dataset_file: c.dataset_sha256,
        c.leaderboard_file: c.leaderboard_sha256,
        c.decision_file: c.decision_sha256,
    }
    observed: dict[str, str] = {}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"FROZEN_SNAPSHOT_FILE_MISSING {name}")
        got = file_sha256(path)
        observed[name] = got
        if got != digest:
            raise RuntimeError(f"FROZEN_SNAPSHOT_SHA_MISMATCH {name}: got={got} expected={digest}")
    return observed


def _pipeline_columns(model) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not hasattr(model, "named_steps") or "prep" not in model.named_steps or "model" not in model.named_steps:
        raise RuntimeError("FROZEN_PIPELINE_STRUCTURE_MISMATCH")
    prep = model.named_steps["prep"]
    numeric: list[str] = []
    categorical: list[str] = []
    for name, _transformer, cols in getattr(prep, "transformers_", []):
        if name == "num":
            numeric = [str(x) for x in list(cols)]
        elif name == "cat":
            categorical = [str(x) for x in list(cols)]
    if not numeric:
        raise RuntimeError("FROZEN_PIPELINE_NUMERIC_SCHEMA_MISSING")
    return tuple(numeric), tuple(categorical)


def _schema_sha(numeric: tuple[str, ...], categorical: tuple[str, ...]) -> str:
    raw = json.dumps({"numeric": list(numeric), "categorical": list(categorical)}, sort_keys=True).encode("utf-8")
    return sha256(raw).hexdigest()


def load_exact_frozen_bundles(
    snapshot_dir: str | Path,
    contract: FrozenSnapshotContract | None = None,
) -> dict[str, FrozenModelBundle]:
    """Load the exact persisted v0.24b champions without refit or retuning."""
    c = contract or FrozenSnapshotContract()
    verify_snapshot_files(snapshot_dir, c)
    assert_runtime_compatible(c)
    champions = joblib.load(Path(snapshot_dir) / c.model_file)
    if not isinstance(champions, dict):
        raise RuntimeError("FROZEN_SNAPSHOT_OBJECT_NOT_DICT")

    bundles: dict[str, FrozenModelBundle] = {}
    for candidate in FROZEN_CANDIDATES_V24C:
        if candidate.strategy not in champions:
            raise RuntimeError(f"FROZEN_STRATEGY_MISSING {candidate.strategy}")
        model = champions[candidate.strategy]
        numeric, categorical = _pipeline_columns(model)
        schema = _schema_sha(numeric, categorical)
        if schema != candidate.feature_schema_sha256:
            raise RuntimeError(
                f"FROZEN_FEATURE_SCHEMA_MISMATCH {candidate.strategy}: got={schema} expected={candidate.feature_schema_sha256}"
            )
        model_class = model.named_steps["model"].__class__.__name__
        expected_class = EXPECTED_MODEL_CLASSES.get(candidate.model_name)
        if expected_class and model_class != expected_class:
            raise RuntimeError(
                f"FROZEN_MODEL_CLASS_MISMATCH {candidate.strategy}: got={model_class} expected={expected_class}"
            )
        bundles[candidate.strategy] = FrozenModelBundle(
            candidate=candidate,
            model=model,
            numeric_features=numeric,
            categorical_features=categorical,
        )
    return bundles


def replay_archived_validation(
    snapshot_dir: str | Path,
    bundles: dict[str, FrozenModelBundle],
    contract: FrozenSnapshotContract | None = None,
) -> pd.DataFrame:
    """Replay archived validation rows and require exact frozen selections.

    This audit uses the byte-verified v0.24b event dataset. It does not rebuild
    market data from a rolling API and does not alter any selected model.
    """
    c = contract or FrozenSnapshotContract()
    verify_snapshot_files(snapshot_dir, c)
    dataset = pd.read_csv(Path(snapshot_dir) / c.dataset_file)
    rows: list[dict] = []
    for candidate in FROZEN_CANDIDATES_V24C:
        bundle = bundles[candidate.strategy]
        val = dataset[(dataset["strategy"] == candidate.strategy) & (dataset["segment"] == "validation")].copy().reset_index(drop=True)
        if len(val) != candidate.expected_validation_events:
            raise RuntimeError(
                f"FROZEN_ARCHIVED_VALIDATION_COUNT_MISMATCH {candidate.strategy}: got={len(val)} expected={candidate.expected_validation_events}"
            )
        missing = [x for x in bundle.features if x not in val.columns]
        if missing:
            raise RuntimeError(f"FROZEN_ARCHIVED_FEATURE_MISSING {candidate.strategy}: {missing}")
        score = probability_score(bundle.model, val[list(bundle.features)])
        selected = np.asarray(score, dtype=float) >= float(candidate.threshold)
        if int(selected.sum()) != candidate.expected_validation_selected:
            raise RuntimeError(
                f"FROZEN_BINARY_REPLAY_MISMATCH {candidate.strategy}: got={int(selected.sum())} expected={candidate.expected_validation_selected}"
            )
        rows.append({
            "strategy": candidate.strategy,
            "status": "EXACT_FROZEN_BINARY_REPLAY_PASS",
            "validation_events": int(len(val)),
            "selected": int(selected.sum()),
            "threshold": float(candidate.threshold),
            "score_min": float(np.nanmin(score)),
            "score_max": float(np.nanmax(score)),
            "model_class": bundle.model.named_steps["model"].__class__.__name__,
            "feature_count": int(len(bundle.features)),
            "model_refit": False,
            "threshold_retuned": False,
        })
    return pd.DataFrame(rows)
