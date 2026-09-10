from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

from research_bot.repo_recovery import bootstrap_research_branch, run_timed


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def make_remote(tmp_path: Path) -> tuple[Path, str]:
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
    run(["git", "clone", "--bare", str(src), str(bare)], tmp_path)
    return bare, "research/test"


def test_direct_bootstrap_and_idempotent_reuse(tmp_path):
    remote, branch = make_remote(tmp_path)
    dest = tmp_path / "checkout"
    first = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10)
    assert first.ok is True
    assert first.status == "READY"
    assert (dest / "research.txt").exists()

    second = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10)
    assert second.ok is True
    assert second.method == "reuse"
    current = subprocess.check_output(["git", "branch", "--show-current"], cwd=dest, text=True).strip()
    assert current == branch


def test_dirty_checkout_is_never_reset(tmp_path):
    remote, branch = make_remote(tmp_path)
    dest = tmp_path / "checkout"
    assert bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10).ok
    p = dest / "research.txt"
    p.write_text("local unsaved work\n", encoding="utf-8")
    result = bootstrap_research_branch(str(remote), branch, dest, command_timeout_seconds=10)
    assert result.ok is False
    assert result.status == "EXISTING_CHECKOUT_NOT_SAFE"
    assert p.read_text(encoding="utf-8") == "local unsaved work\n"


def test_missing_branch_fails_without_retry_loop(tmp_path):
    remote, _ = make_remote(tmp_path)
    started = time.monotonic()
    result = bootstrap_research_branch(str(remote), "research/does-not-exist", tmp_path / "missing", command_timeout_seconds=5)
    elapsed = time.monotonic() - started
    assert result.ok is False
    assert result.status == "BRANCH_NOT_FOUND_OR_REMOTE_UNREACHABLE"
    assert elapsed < 5


def test_command_timeout_opens_instead_of_hanging():
    result = run_timed(
        "sleep_test",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=0.15,
    )
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.elapsed_seconds < 1.5
