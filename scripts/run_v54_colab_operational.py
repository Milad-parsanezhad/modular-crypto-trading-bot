#!/usr/bin/env python3
"""Google Colab launcher for the frozen v0.54 research workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

EXPECTED_BUNDLE_SHA256 = (
    "4b90656f9501d22381ef1e8d3e8c0ac3dbe3f295ef5fe15ba1ca7dca5ec4faee"
)
BUNDLE = Path("/content/V54_OPERATIONAL_SOURCE_BUNDLE.zip")
PROJECT = Path("/content/v54_bot")
RUNS = Path("/content/v54_operational_runs")
EXECUTION_FLAGS = (
    "LIVE_EXECUTION",
    "PAPER_EXECUTION",
    "BOT_FORWARD_PAPER_ENABLED",
    "BOT_PAPER_EXECUTION_ENABLED",
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def is_source_zip(path: Path) -> bool:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        names = {name.replace("\\", "/") for name in archive.namelist()}
        return (
            any(name.endswith("pyproject.toml") for name in names)
            and any("/research_bot/" in f"/{name}" for name in names)
            and any("/scripts/" in f"/{name}" for name in names)
            and archive.testzip() is None
        )

def acquire_bundle() -> Path:
    if BUNDLE.exists() and is_source_zip(BUNDLE):
        return BUNDLE
    if BUNDLE.exists():
        backup = BUNDLE.with_suffix(".invalid-upload")
        counter = 1
        while backup.exists():
            backup = BUNDLE.with_name(f"{BUNDLE.stem}.invalid-upload-{counter}")
            counter += 1
        BUNDLE.replace(backup)
        print("Preserved invalid previous upload at:", backup)

    from google.colab import files

    print("Upload only V54_OPERATIONAL_SOURCE_BUNDLE.zip")
    uploaded = files.upload()
    candidates = []
    for name, payload in uploaded.items():
        temporary = Path("/content") / f".candidate-{Path(name).name}"
        temporary.write_bytes(payload)
        if is_source_zip(temporary):
            candidates.append((name, temporary))
        else:
            temporary.unlink(missing_ok=True)

    if len(candidates) != 1:
        raise RuntimeError(
            "Upload exactly one valid source ZIP. JSON result files are not source bundles."
        )

    original_name, candidate = candidates[0]
    candidate.replace(BUNDLE)
    digest = sha256_file(BUNDLE)
    if digest != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(
            f"Source bundle hash mismatch: expected {EXPECTED_BUNDLE_SHA256}, got {digest}"
        )
    print("Accepted:", original_name)
    return BUNDLE

def safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            archive.extract(member, root)

def find_project_root(root: Path) -> Path:
    candidates = [
        item.parent
        for item in root.rglob("pyproject.toml")
        if (item.parent / "research_bot").is_dir()
        and (item.parent / "scripts").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one project root, found {len(candidates)}")
    return candidates[0]

def run(
    command: Sequence[str],
    cwd: Path,
    allowed: set[int] = {0},
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    print(completed.stdout, flush=True)
    if completed.returncode not in allowed:
        raise RuntimeError(
            f"Command returned {completed.returncode}: {' '.join(command)}"
        )
    return completed

def configure_safety() -> None:
    for flag in EXECUTION_FLAGS:
        os.environ[flag] = "false"
    os.environ["TRADING_EXECUTION"] = "false"
    os.environ["INSTITUTIONAL_RESEARCH"] = "true"
    os.environ["EXECUTION_MODE"] = "RESEARCH_ONLY"

def assert_safety() -> None:
    truthy = {"1", "true", "yes", "on", "enabled"}
    active = [
        flag for flag in EXECUTION_FLAGS
        if os.environ.get(flag, "").strip().lower() in truthy
    ]
    if active:
        raise RuntimeError("Execution flags active: " + ", ".join(active))

def prepare_project(bundle: Path) -> Path:
    extraction = Path(tempfile.mkdtemp(prefix="v54-source-", dir="/content"))
    safe_extract(bundle, extraction)
    source = find_project_root(extraction)
    if PROJECT.exists():
        archived = Path(tempfile.mkdtemp(prefix="v54-previous-", dir="/content"))
        PROJECT.replace(archived / PROJECT.name)
    shutil.copytree(source, PROJECT)
    return PROJECT

def latest_status() -> Path:
    candidates = sorted(
        RUNS.rglob("run_status.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Workflow produced no run_status.json")
    return candidates[0]

def main() -> int:
    configure_safety()
    assert_safety()
    bundle = acquire_bundle()
    print("Source bundle SHA256:", sha256_file(bundle))
    project = prepare_project(bundle)

    run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-e", ".[dev]"],
        project,
        timeout=900,
    )
    run(
        [sys.executable, "-m", "compileall", "-q", "research_bot", "scripts", "tests"],
        project,
        timeout=300,
    )
    run([sys.executable, "-m", "pytest", "-q"], project, timeout=1800)

    RUNS.mkdir(parents=True, exist_ok=True)
    workflow = run(
        [
            sys.executable,
            "-m",
            "scripts.run_v54_workflow",
            "--output-root",
            str(RUNS),
        ],
        project,
        allowed={0, 2},
        timeout=7200,
    )
    assert_safety()

    status_path = latest_status()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = {
        "status": status.get("status"),
        "status_path": str(status_path),
        "archive": status.get("archive"),
        "workflow_return_code": workflow.returncode,
        "research_only": True,
        "paper_execution": False,
        "live_execution": False,
        "scientific_promotion_authorized": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    archive = status.get("archive")
    if archive and Path(archive).is_file():
        from google.colab import files
        files.download(archive)
    # Return success to Colab after a safely recorded infrastructure block.\n    # The scientific status remains explicit in run_status.json.\n    return 0\n\nif __name__ == "__main__":
    raise SystemExit(main())
