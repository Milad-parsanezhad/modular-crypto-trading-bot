from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import ExperimentManifest


def dataframe_fingerprint(df: pd.DataFrame, *, include_index: bool = False) -> str:
    """Stable SHA-256 fingerprint of a dataframe's content and schema."""

    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    payload = pd.util.hash_pandas_object(normalized, index=include_index).values.tobytes()
    schema = "|".join(f"{c}:{normalized[c].dtype}" for c in normalized.columns).encode("utf-8")
    return hashlib.sha256(schema + b"\n" + payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> Path:
    """Persist immutable research metadata next to experiment artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(manifest))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target
