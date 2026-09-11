from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

from research_bot.repo_recovery import bootstrap_research_branch, run_timed


def run(cmd, cwd=None, env=None):
    subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def make_remote(tmp_path: Path) -> tuple[Path, str, str]:
    src = tmp_path / "src"
    bare = tmp_path / "remote.git"
    src.mkdir()
    run(["git", "init"], src)
    run(["git", "config", "user.email", "test@example.com"], src)
    run(["git", "config", "user.name", "test"], src)
    (src / "README.md").write_text("main\n", encoding="utf-8")
    run(["git", "add", "."], src)
    run(["git", "commit", "-m", "main"], src)
    run(["git", "branch", "-M", "main"], src)
    run(["git", "checkout", "-b", "research/test"], src)
    (src / "research.txt").write_text("research\n", encoding="utf-8")
    run(["git", "add", "."], src)
    run(["git", "commit", "-m", "research"], src)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    run(["git", "clone", "--bare", str(src), str(bare)], tmp_path)
    return bare, "research/test", head


def test_direct_bootstrap_and_idempotent_reuse(tmp_path):
    remote, branch, head = make_remote(tmp_path)
    dest = tmp_path / "checkout"
    first = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10, expected_commit=head)
    assert first.ok is True
    assert first.status == "READY"
    assert (dest / "research.txt").exists()

    second = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10, expected_commit=head)
    assert second.ok is True
    assert second.method == "reuse"
    current = subprocess.check_output(["git", "branch", "--show-current"], cwd=dest, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    assert current == branch
    assert actual == head


def test_dirty_checkout_is_never_reset(tmp_path):
    remote, branch, _ = make_remote(tmp_path)
    dest = tmp_path / "checkout"
    assert bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10).ok
    p = dest / "research.txt"
    p.write_text("local unsaved work\n", encoding="utf-8")
    result = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10)
    assert result.ok is False
    assert result.status == "EXISTING_CHECKOUT_NOT_SAFE"
    assert p.read_text(encoding="utf-8") == "local unsaved work\n"


def test_missing_branch_fails_without_retry_loop(tmp_path):
    remote, _, _ = make_remote(tmp_path)
    started = time.monotonic()
    result = bootstrap_research_branch(str(remote), "research/does-not-exist", tmp_path / "missing", command_timeout_seconds=5)
    elapsed = time.monotonic() - started
    assert result.ok is False
    assert result.status == "BRANCH_NOT_FOUND_OR_REMOTE_UNREACHABLE"
    assert elapsed < 5


def test_expected_head_mismatch_fails_closed(tmp_path):
    remote, branch, _ = make_remote(tmp_path)
    dest = tmp_path / "checkout"
    result = bootstrap_research_branch(
        str(remote), branch, dest,
        command_timeout_seconds=10,
        expected_commit="0" * 40,
    )
    assert result.ok is False
    assert result.status == "HEAD_MISMATCH"
    assert "HEAD_MISMATCH" in result.message


def test_nonempty_nonrepo_destination_is_never_deleted(tmp_path):
    remote, branch, _ = make_remote(tmp_path)
    dest = tmp_path / "important"
    dest.mkdir()
    sentinel = dest / "do-not-delete.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    result = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10)
    assert result.ok is False
    assert result.status == "DESTINATION_NOT_EMPTY"
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_command_timeout_opens_instead_of_hanging():
    result = run_timed(
        "sleep_test",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=0.15,
    )
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.elapsed_seconds < 1.5


def test_direct_clone_failure_switches_method_and_removes_stale_partial_state(tmp_path, monkeypatch):
    remote, branch, head = make_remote(tmp_path)
    real_git = shutil.which("git")
    assert real_git
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"clone\" ]; then\n"
        "  dest=\"${@: -1}\"\n"
        "  mkdir -p \"${dest}\"\n"
        "  printf 'stale\\n' > \"${dest}/STALE_PARTIAL_FILE\"\n"
        "  echo 'forced direct clone failure' >&2\n"
        "  exit 97\n"
        "fi\n"
        f"exec {real_git} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper_dir}{os.pathsep}{os.environ['PATH']}")

    dest = tmp_path / "recovered"
    audit = tmp_path / "audit.jsonl"
    result = bootstrap_research_branch(
        str(remote), branch, dest,
        command_timeout_seconds=10,
        expected_commit=head,
        audit_log=audit,
    )
    assert result.ok is True
    assert result.method == "init_fetch_fallback"
    assert not (dest / "STALE_PARTIAL_FILE").exists()
    assert (dest / "research.txt").read_text(encoding="utf-8") == "research\n"
    actual = subprocess.check_output([real_git, "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    current = subprocess.check_output([real_git, "branch", "--show-current"], cwd=dest, text=True).strip()
    assert actual == head
    assert current == branch
    audit_text = audit.read_text(encoding="utf-8")
    assert '"event": "direct_clone"' in audit_text
    assert '"event": "fallback_fetch"' in audit_text
