from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "evidence" / "ledger"
REQUIRED = {
    "schema_version", "project_version", "maturity_tier", "evidence_class",
    "evaluation_basis", "original_objective", "scope", "engineering_result",
    "scientific_result", "economic_result", "provenance", "safety",
    "limitations", "contribution_to_next_version", "classifications",
    "source_documents",
}
EVIDENCE_CLASSES = {
    "VERIFIED_CANONICAL", "VERIFIED_REPOSITORY_RECORD", "RETROSPECTIVE_RECONSTRUCTION"
}
CLASSIFICATIONS = {
    "ENGINEERING_PASS", "ENGINEERING_PARTIAL", "SCIENTIFIC_METHOD_PASS",
    "HYPOTHESIS_SUPPORTED", "HYPOTHESIS_REJECTED", "DIAGNOSTIC_SUCCESS",
    "INSUFFICIENT_EVIDENCE", "HISTORICAL_PROTOTYPE", "SUPERSEDED_WITH_VALUE",
}


def validate_record(record: dict, path: Path) -> None:
    missing = sorted(REQUIRED.difference(record))
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if record["schema_version"] != "1.0":
        raise ValueError(f"{path}: unsupported schema_version")
    if record["evidence_class"] not in EVIDENCE_CLASSES:
        raise ValueError(f"{path}: invalid evidence_class")
    classes = record["classifications"]
    if not isinstance(classes, list) or not classes or not set(classes).issubset(CLASSIFICATIONS):
        raise ValueError(f"{path}: invalid classifications")
    for name in ("engineering_result", "scientific_result", "economic_result"):
        section = record[name]
        if not isinstance(section, dict) or not {"status", "summary"}.issubset(section):
            raise ValueError(f"{path}: malformed {name}")
    safety = record["safety"]
    if safety.get("paper_execution") is not False or safety.get("live_execution") is not False:
        raise ValueError(f"{path}: execution safety must remain false")


def validate_ledger() -> list[Path]:
    paths = sorted(LEDGER.glob("v*.json"))
    if not paths:
        raise ValueError("evidence ledger is empty")
    versions: set[str] = set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(record, path)
        version = str(record["project_version"])
        if version in versions:
            raise ValueError(f"duplicate project_version: {version}")
        versions.add(version)
    return paths
