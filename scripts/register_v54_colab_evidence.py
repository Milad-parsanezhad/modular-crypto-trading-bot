#!/usr/bin/env python3
"""Register sanitized, append-only v0.54 Colab evidence.

This utility never enables trading and never uploads anything by itself. It
creates a deterministic evidence directory that can be reviewed and committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

MAX_FILE_BYTES = 100 * 1024 * 1024
SOURCE_BUNDLE_SHA256 = (
    "4b90656f9501d22381ef1e8d3e8c0ac3dbe3f295ef5fe15ba1ca7dca5ec4faee"
)
SOURCE_FINGERPRINT = (
    "929eceaff3ae4286183acadb55cd0964deaf674d325783afacfed70e63d67db5"
)
FORBIDDEN_TRUE_FLAGS = (
    "LIVE_EXECUTION",
    "PAPER_EXECUTION",
    "BOT_FORWARD_PAPER_ENABLED",
    "BOT_PAPER_EXECUTION_ENABLED",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|api[_-]?secret|secret[_-]?key|password|token)"
               r"\s*[:=]\s*['\"]?[^\s,'\"}]{8,}"),
    re.compile(r"(?i)authorization\s*:\s*(bearer|basic)\s+\S+"),
    re.compile(r"(?i)(postgres(?:ql)?|mysql|redis)://[^\s]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value

def assert_regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Evidence file is empty: {path}")
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Evidence file exceeds {MAX_FILE_BYTES} bytes: {path}")

def assert_no_secrets(path: Path) -> None:
    assert_regular_file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]
    if matches:
        raise ValueError(f"Potential secret detected in {path}; evidence rejected")

def is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

def assert_research_only(status: dict) -> None:
    serialized = json.dumps(status, sort_keys=True)
    for flag in FORBIDDEN_TRUE_FLAGS:
        direct = status.get(flag)
        lower = status.get(flag.lower())
        if is_true(direct) or is_true(lower):
            raise ValueError(f"Unsafe evidence: {flag} was enabled")
        if re.search(rf'"{re.escape(flag)}"\s*:\s*true', serialized, re.I):
            raise ValueError(f"Unsafe evidence: {flag} was enabled")
    if status.get("scientific_promotion_authorized") is True:
        raise ValueError("Unexpected scientific promotion authorization")

def normalize_time(value: object) -> str:
    if value:
        candidate = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            raise ValueError("Run timestamp must include a timezone")
        return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def evidence_files(
    run_status: Path,
    tests: Path | None,
    log: Path | None,
    artifacts: Iterable[Path],
) -> list[tuple[str, Path]]:
    items = [("run_status.json", run_status)]
    if tests:
        items.append(("tests.txt", tests))
    if log:
        items.append(("runner.log", log))
    for index, artifact in enumerate(artifacts, start=1):
        safe_name = Path(artifact.name).name
        items.append((f"artifacts/{index:02d}-{safe_name}", artifact))
    return items

def register(args: argparse.Namespace) -> Path:
    run_status = args.run_status.resolve()
    tests = args.tests.resolve() if args.tests else None
    log = args.log.resolve() if args.log else None
    artifacts = [item.resolve() for item in args.artifact]

    status = load_json(run_status)
    assert_research_only(status)

    inputs = evidence_files(run_status, tests, log, artifacts)
    for _, source in inputs:
        assert_no_secrets(source)

    timestamp = normalize_time(
        status.get("finished_at")
        or status.get("completed_at")
        or status.get("started_at")
    )
    identity_seed = (
        sha256_file(run_status)
        + SOURCE_FINGERPRINT
        + SOURCE_BUNDLE_SHA256
    ).encode("utf-8")
    suffix = hashlib.sha256(identity_seed).hexdigest()[:12]
    run_id = args.run_id or f"{timestamp}-{suffix}"

    output_root = args.output.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Evidence run already exists: {destination}")

    staging = output_root / f".{run_id}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir(parents=True)

    try:
        manifest_files = []
        for relative_name, source in inputs:
            target = staging / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest_files.append(
                {
                    "path": relative_name,
                    "sha256": sha256_file(target),
                    "size_bytes": target.stat().st_size,
                }
            )

        manifest = {
            "schema_version": "v54-colab-evidence-1",
            "run_id": run_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "classification": "ENGINEERING_EVIDENCE_ONLY",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "research_only": True,
            "paper_execution": False,
            "live_execution": False,
            "kraken_sealed": True,
            "scientific_promotion_authorized": False,
            "files": manifest_files,
        }
        manifest_path = staging / "artifact_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return destination

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    destination = register(parse_args(argv))
    print(destination)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
