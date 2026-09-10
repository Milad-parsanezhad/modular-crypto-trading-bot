from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from typing import Sequence


@dataclass
class CommandResult:
    stage: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def fingerprint(self) -> str:
        tail = (self.stderr or self.stdout or "").strip().splitlines()
        return f"{self.stage}|{self.returncode}|{tail[-1] if tail else ''}"[:500]


@dataclass
class BootstrapResult:
    ok: bool
    status: str
    branch: str
    destination: str
    method: str | None
    attempts: int
    elapsed_seconds: float
    message: str


class LoopBreaker:
    """Stop retry loops and force a method change quickly.

    A Git command is time-boxed, repeated identical failures are counted, and the
    caller changes recovery method rather than repeating the same failing clone.
    """

    def __init__(self, command_timeout_seconds: float = 60.0, max_identical_failures: int = 2):
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if max_identical_failures < 1:
            raise ValueError("max_identical_failures must be >= 1")
        self.command_timeout_seconds = float(command_timeout_seconds)
        self.max_identical_failures = int(max_identical_failures)
        self._failure_counts: dict[str, int] = {}

    def record_failure(self, result: CommandResult) -> bool:
        fp = result.fingerprint
        self._failure_counts[fp] = self._failure_counts.get(fp, 0) + 1
        return self._failure_counts[fp] >= self.max_identical_failures


def run_timed(stage: str, command: Sequence[str], *, timeout_seconds: float = 60.0, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
    start = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        p = subprocess.run(
            list(command), cwd=str(cwd) if cwd is not None else None,
            env=merged_env, text=True, capture_output=True,
            timeout=timeout_seconds, check=False,
        )
        return CommandResult(stage, list(command), p.returncode, p.stdout, p.stderr, time.monotonic() - start, False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(stage, list(command), 124, stdout, stderr, time.monotonic() - start, True)


def _append_audit(path: Path | None, event: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _clean_worktree(dest: Path, timeout: float) -> tuple[bool, str]:
    r = run_timed("status", ["git", "status", "--porcelain"], timeout_seconds=timeout, cwd=dest)
    if not r.ok:
        return False, (r.stderr or r.stdout).strip()
    if r.stdout.strip():
        return False, "LOCAL_WORKTREE_DIRTY"
    return True, "CLEAN"


def _remote_branch_exists(repo_url: str, branch: str, timeout: float) -> CommandResult:
    return run_timed(
        "ls_remote_branch",
        ["git", "ls-remote", "--exit-code", "--heads", repo_url, f"refs/heads/{branch}"],
        timeout_seconds=timeout,
    )


def _verify_head(dest: Path, expected_commit: str | None, timeout: float) -> tuple[bool, str]:
    r = run_timed("verify_head", ["git", "rev-parse", "HEAD"], timeout_seconds=timeout, cwd=dest)
    if not r.ok:
        return False, (r.stderr or r.stdout).strip()
    actual = r.stdout.strip().lower()
    if expected_commit and actual != expected_commit.strip().lower():
        return False, f"HEAD_MISMATCH expected={expected_commit.strip()} actual={actual}"
    return True, actual


def bootstrap_research_branch(
    repo_url: str,
    branch: str,
    destination: str | Path,
    *,
    command_timeout_seconds: float = 60.0,
    max_attempts: int = 2,
    audit_log: str | Path | None = None,
    expected_commit: str | None = None,
) -> BootstrapResult:
    """Obtain an existing research branch without infinite clone/fetch loops.

    Recovery order:
      1. Preflight remote branch existence.
      2. Reuse an existing clean checkout when present.
      3. Try a direct shallow branch clone exactly once.
      4. If it fails, destroy only the partial directory created by this call and
         immediately switch method to git-init + exact branch-ref fetch + checkout.
      5. Optionally verify HEAD against an immutable expected commit.

    A missing branch is never silently created, dirty local work is never reset,
    and every Git operation is time-boxed (60 seconds by default).
    """
    started = time.monotonic()
    dest = Path(destination).expanduser().resolve()
    audit = Path(audit_log).expanduser().resolve() if audit_log else None
    guard = LoopBreaker(command_timeout_seconds, max_identical_failures=max_attempts)
    attempts = 0

    if not repo_url or not branch:
        return BootstrapResult(False, "INVALID_INPUT", branch, str(dest), None, 0, 0.0, "repo_url and branch are required")

    remote = _remote_branch_exists(repo_url, branch, command_timeout_seconds)
    _append_audit(audit, {"event": "remote_preflight", **asdict(remote)})
    if not remote.ok:
        reason = "REMOTE_CHECK_TIMEOUT" if remote.timed_out else "BRANCH_NOT_FOUND_OR_REMOTE_UNREACHABLE"
        return BootstrapResult(False, reason, branch, str(dest), None, 1, time.monotonic() - started, (remote.stderr or remote.stdout).strip())

    if dest.exists() and (dest / ".git").exists():
        clean, message = _clean_worktree(dest, command_timeout_seconds)
        if not clean:
            return BootstrapResult(False, "EXISTING_CHECKOUT_NOT_SAFE", branch, str(dest), "reuse", 1, time.monotonic() - started, message)
        methods = [
            ("reuse_fetch", ["git", "fetch", "--no-tags", "--depth=1", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}"]),
            ("reuse_checkout", ["git", "checkout", "-B", branch, f"origin/{branch}"]),
        ]
        for stage, cmd in methods:
            attempts += 1
            r = run_timed(stage, cmd, timeout_seconds=command_timeout_seconds, cwd=dest)
            _append_audit(audit, {"event": stage, **asdict(r)})
            if not r.ok:
                return BootstrapResult(False, "REUSE_FAILED", branch, str(dest), "reuse", attempts, time.monotonic() - started, (r.stderr or r.stdout).strip())
        verified, head = _verify_head(dest, expected_commit, command_timeout_seconds)
        _append_audit(audit, {"event": "head_verification", "ok": verified, "message": head})
        if not verified:
            return BootstrapResult(False, "HEAD_MISMATCH", branch, str(dest), "reuse", attempts, time.monotonic() - started, head)
        return BootstrapResult(True, "READY", branch, str(dest), "reuse", attempts, time.monotonic() - started, f"existing clean checkout reused; head={head}")

    if dest.exists() and any(dest.iterdir()):
        return BootstrapResult(False, "DESTINATION_NOT_EMPTY", branch, str(dest), None, 0, time.monotonic() - started, "destination exists and is not a Git checkout")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not any(dest.iterdir()):
        dest.rmdir()

    # Method A: direct branch clone exactly once.
    attempts += 1
    direct = run_timed(
        "direct_clone",
        ["git", "clone", "--filter=blob:none", "--single-branch", "--branch", branch, "--depth=1", repo_url, str(dest)],
        timeout_seconds=command_timeout_seconds,
    )
    _append_audit(audit, {"event": "direct_clone", **asdict(direct)})
    if direct.ok:
        verified, head = _verify_head(dest, expected_commit, command_timeout_seconds)
        _append_audit(audit, {"event": "head_verification", "ok": verified, "message": head})
        if not verified:
            return BootstrapResult(False, "HEAD_MISMATCH", branch, str(dest), "direct_clone", attempts, time.monotonic() - started, head)
        return BootstrapResult(True, "READY", branch, str(dest), "direct_clone", attempts, time.monotonic() - started, f"direct shallow branch clone succeeded; head={head}")

    guard.record_failure(direct)

    # The destination did not exist before Method A, so anything present now was
    # created by the failed clone and may contain stale .git/worktree state.
    # Remove the entire partial checkout before changing method.
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=False)

    # Method B: change strategy immediately instead of retrying clone.
    dest.mkdir(parents=True, exist_ok=False)
    for stage, cmd in [
        ("fallback_init", ["git", "init"]),
        ("fallback_origin", ["git", "remote", "add", "origin", repo_url]),
        ("fallback_fetch", ["git", "fetch", "--no-tags", "--depth=1", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}"]),
        ("fallback_checkout", ["git", "checkout", "-B", branch, f"origin/{branch}"]),
    ]:
        attempts += 1
        r = run_timed(stage, cmd, timeout_seconds=command_timeout_seconds, cwd=dest)
        _append_audit(audit, {"event": stage, **asdict(r)})
        if not r.ok:
            loop_detected = guard.record_failure(r)
            status = "LOOP_BREAKER_OPEN" if loop_detected or r.timed_out else "FALLBACK_FAILED"
            return BootstrapResult(False, status, branch, str(dest), "init_fetch_fallback", attempts, time.monotonic() - started, (r.stderr or r.stdout).strip())

    verified, head = _verify_head(dest, expected_commit, command_timeout_seconds)
    _append_audit(audit, {"event": "head_verification", "ok": verified, "message": head})
    if not verified:
        return BootstrapResult(False, "HEAD_MISMATCH", branch, str(dest), "init_fetch_fallback", attempts, time.monotonic() - started, head)
    return BootstrapResult(True, "READY", branch, str(dest), "init_fetch_fallback", attempts, time.monotonic() - started, f"clone failed once; exact-ref fallback succeeded; head={head}")
