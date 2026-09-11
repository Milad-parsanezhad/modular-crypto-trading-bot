from __future__ import annotations

import argparse
import json

from research_bot.repo_recovery import bootstrap_research_branch


def main() -> int:
    ap = argparse.ArgumentParser(description="Timeboxed, fail-safe research branch bootstrap")
    ap.add_argument("repo_url")
    ap.add_argument("branch")
    ap.add_argument("destination")
    ap.add_argument("--timeout-seconds", type=float, default=60.0)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--audit-log", default=None)
    ap.add_argument("--expected-commit", default=None, help="Optional immutable SHA that checkout HEAD must equal")
    args = ap.parse_args()

    result = bootstrap_research_branch(
        args.repo_url,
        args.branch,
        args.destination,
        command_timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        audit_log=args.audit_log,
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False, default=str))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
