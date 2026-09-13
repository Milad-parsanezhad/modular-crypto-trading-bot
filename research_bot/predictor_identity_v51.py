from __future__ import annotations

"""Identity/provenance contract for the v0.51 forward scoring bundle."""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


V51_PREDICTOR_BUNDLE_VERSION = "v51-v47-c1-fold5-production-bundle-v1"
V51_CANONICAL_V47_RUN = 34705324352
V51_CANONICAL_V47_HEAD = "802b0cde38549617589b6a4a313949361dd9425c"
V51_CANONICAL_V47_ARTIFACT = 10300833544
V51_CANONICAL_V47_DIGEST = "sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118"
V51_CANONICAL_V44_RUN = 34700944062
V51_CANONICAL_V44_PREPARED_ARTIFACT = 10299564398
V51_PRODUCTION_FOLD = 5
V51_PRODUCTION_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT")
V51_REPAIRED_PROSPECTIVE_START = "2026-09-13T12:00:00Z"


@dataclass(frozen=True)
class PredictorVerificationV51:
    symbol: str
    fold: int
    rows: int
    canonical_temperature: float
    max_abs_expected_r_error: float
    mean_abs_expected_r_error: float
    selected_model_match_fraction: float
    identity_match: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.identity_match
            and self.rows > 0
            and np.isfinite(self.canonical_temperature)
            and self.canonical_temperature > 0
            and np.isfinite(self.max_abs_expected_r_error)
            and self.max_abs_expected_r_error <= 1e-6
            and self.selected_model_match_fraction == 1.0
        )


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    h = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verification_manifest(rows: list[PredictorVerificationV51]) -> dict[str, Any]:
    result = {
        "bundle_version": V51_PREDICTOR_BUNDLE_VERSION,
        "canonical_v44_run": V51_CANONICAL_V44_RUN,
        "canonical_v44_prepared_artifact": V51_CANONICAL_V44_PREPARED_ARTIFACT,
        "canonical_v47_run": V51_CANONICAL_V47_RUN,
        "canonical_v47_head": V51_CANONICAL_V47_HEAD,
        "canonical_v47_artifact": V51_CANONICAL_V47_ARTIFACT,
        "canonical_v47_artifact_digest": V51_CANONICAL_V47_DIGEST,
        "production_fold": V51_PRODUCTION_FOLD,
        "production_symbols": list(V51_PRODUCTION_SYMBOLS),
        "repaired_prospective_start": V51_REPAIRED_PROSPECTIVE_START,
        "verifications": [asdict(row) | {"passed": row.passed} for row in rows],
    }
    result["all_verified"] = bool(len(rows) == len(V51_PRODUCTION_SYMBOLS) and all(row.passed for row in rows))
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def assert_verified_manifest(payload: dict[str, Any]) -> None:
    if payload.get("bundle_version") != V51_PREDICTOR_BUNDLE_VERSION:
        raise ValueError("unexpected v0.51 predictor bundle version")
    if int(payload.get("production_fold", -1)) != V51_PRODUCTION_FOLD:
        raise ValueError("unexpected v0.51 production fold")
    if tuple(payload.get("production_symbols", ())) != V51_PRODUCTION_SYMBOLS:
        raise ValueError("unexpected v0.51 production symbol universe")
    if payload.get("repaired_prospective_start") != V51_REPAIRED_PROSPECTIVE_START:
        raise ValueError("unexpected v0.51 prospective start")
    if payload.get("all_verified") is not True:
        raise ValueError("v0.51 predictor bundle is not fully verified")
