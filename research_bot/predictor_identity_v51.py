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
            self.symbol in V51_PRODUCTION_SYMBOLS
            and self.fold == V51_PRODUCTION_FOLD
            and self.identity_match is True
            and isinstance(self.rows, int)
            and not isinstance(self.rows, bool)
            and self.rows > 0
            and np.isfinite(self.canonical_temperature)
            and self.canonical_temperature > 0
            and np.isfinite(self.max_abs_expected_r_error)
            and np.isfinite(self.mean_abs_expected_r_error)
            and self.max_abs_expected_r_error <= 1e-6
            and self.mean_abs_expected_r_error <= 1e-6
            and np.isfinite(self.selected_model_match_fraction)
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
    result["all_verified"] = bool(
        len(rows) == len(V51_PRODUCTION_SYMBOLS)
        and tuple(row.symbol for row in rows) == V51_PRODUCTION_SYMBOLS
        and all(row.passed for row in rows)
    )
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def _strict_verification_row(raw: Any, expected_symbol: str) -> PredictorVerificationV51:
    if not isinstance(raw, dict):
        raise ValueError("v0.51 verification row must be an object")
    expected_keys = {
        "symbol",
        "fold",
        "rows",
        "canonical_temperature",
        "max_abs_expected_r_error",
        "mean_abs_expected_r_error",
        "selected_model_match_fraction",
        "identity_match",
        "passed",
    }
    if set(raw) != expected_keys:
        raise ValueError("unexpected v0.51 verification row schema")
    if raw.get("symbol") != expected_symbol:
        raise ValueError("unexpected or duplicate v0.51 verification symbol")
    if raw.get("identity_match") is not True:
        raise ValueError("v0.51 predictor identity mismatch")
    if raw.get("passed") is not True:
        raise ValueError("v0.51 verification row not passed")
    try:
        row = PredictorVerificationV51(
            symbol=str(raw["symbol"]),
            fold=int(raw["fold"]),
            rows=int(raw["rows"]),
            canonical_temperature=float(raw["canonical_temperature"]),
            max_abs_expected_r_error=float(raw["max_abs_expected_r_error"]),
            mean_abs_expected_r_error=float(raw["mean_abs_expected_r_error"]),
            selected_model_match_fraction=float(raw["selected_model_match_fraction"]),
            identity_match=raw["identity_match"],
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid v0.51 verification row values") from exc
    # Reject bool-as-int and lossy/coerced row counts.
    if isinstance(raw["rows"], bool) or row.rows != raw["rows"]:
        raise ValueError("invalid v0.51 verification row count")
    if row.fold != V51_PRODUCTION_FOLD:
        raise ValueError("unexpected v0.51 verification fold")
    if not row.passed:
        raise ValueError("v0.51 verification metrics fail closed")
    return row


def assert_verified_manifest(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("v0.51 predictor manifest must be an object")

    immutable = {
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
    }
    for key, expected in immutable.items():
        if payload.get(key) != expected:
            raise ValueError(f"unexpected v0.51 manifest field: {key}")

    claimed_hash = payload.get("manifest_sha256")
    if not isinstance(claimed_hash, str) or len(claimed_hash) != 64:
        raise ValueError("missing/invalid v0.51 manifest hash")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if canonical_json_sha256(unhashed) != claimed_hash:
        raise ValueError("v0.51 manifest hash mismatch")

    verifications = payload.get("verifications")
    if not isinstance(verifications, list) or len(verifications) != len(V51_PRODUCTION_SYMBOLS):
        raise ValueError("v0.51 verification rows incomplete")
    rows = [
        _strict_verification_row(raw, expected_symbol)
        for raw, expected_symbol in zip(verifications, V51_PRODUCTION_SYMBOLS)
    ]
    recomputed_all_verified = bool(all(row.passed for row in rows))
    if payload.get("all_verified") is not recomputed_all_verified or not recomputed_all_verified:
        raise ValueError("v0.51 predictor bundle is not fully verified")
