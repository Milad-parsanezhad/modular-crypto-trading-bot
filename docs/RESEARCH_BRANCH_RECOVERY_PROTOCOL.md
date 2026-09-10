# Research Branch Recovery Protocol

This protocol exists because repeated branch-create/clone/fetch attempts can waste time and hide the real failure mode.

## Root cause observed

A recurring failure mode was not a genuine need to create a new branch. The research ref already existed, so repeated create-ref calls returned GitHub HTTP 422 (`Reference already exists`). Repeating the same operation cannot resolve that state.

The correct interpretation is:

- `Reference already exists` => stop branch creation immediately;
- fetch/read the existing branch ref;
- reuse or update that branch only after checking its current head and worktree state.

## One-minute anti-loop rule

For repository acquisition and branch preparation:

1. Every network Git operation is time-boxed to 60 seconds by default.
2. The same failed method is not retried indefinitely.
3. After a direct clone fails once, the process changes method to `git init + exact-ref shallow fetch + checkout`.
4. Two identical failure fingerprints open the loop breaker and return control to diagnosis.
5. Interactive credential prompts are disabled with `GIT_TERMINAL_PROMPT=0` so a hidden prompt cannot create an apparent hang.
6. A missing branch is reported as missing; it is never silently created from the wrong base.
7. A dirty existing checkout is never hard-reset or overwritten by the recovery tool.
8. A failed clone's entire partial destination is removed before fallback because stale `.git` state or worktree files can corrupt recovery. This cleanup is permitted only because that destination was created by the failed clone in the same call; pre-existing non-repository user directories are never deleted.
9. An optional immutable `expected_commit` can be supplied. A checkout is `READY` only when `git rev-parse HEAD` exactly equals that SHA.

## Preferred decision tree

### Remote branch already exists

Reuse it. Do not call `create branch` again.

### Local checkout exists and is clean

Fetch the exact remote ref at depth 1 and check it out. If an expected commit SHA was supplied, verify HEAD before returning success.

### No local checkout

Attempt one shallow single-branch clone. If it fails or times out, immediately change method rather than repeating clone.

### Fallback method

Delete only the partial checkout created by the failed clone, initialize an empty repository, add origin, shallow-fetch exactly `refs/heads/<branch>`, check out that ref, then verify the immutable expected SHA when provided.

### Remote branch missing or remote unreachable

Stop with a diagnostic result. Branch creation is a separate explicit action and must identify the intended base commit/ref.

## Machine-readable audit

`scripts/research_branch_bootstrap.py` writes optional JSONL audit events. Each event records stage, command, return code, elapsed seconds and timeout state. Credentials are never written by the recovery module. The CLI also accepts `--expected-commit`.

Example:

```bash
python scripts/research_branch_bootstrap.py \
  https://github.com/OWNER/REPO.git \
  research/v23-ml-rebuild-audit2 \
  /tmp/research-checkout \
  --timeout-seconds 60 \
  --max-attempts 2 \
  --expected-commit <IMMUTABLE_SHA> \
  --audit-log /tmp/research-checkout-audit.jsonl
```

## CI behavior

`.github/workflows/v23r-repo-recovery-guard.yml` now implements all of the following:

- checks out the immutable PR-head/push SHA rather than GitHub's synthetic pull-request merge ref;
- gives the primary checkout one minute and immediately changes method if it fails or resolves the wrong SHA;
- wipes the complete ephemeral partial workspace before fallback so stale worktree state cannot leak into recovery;
- fetches the exact research branch ref and proves it resolves to the event SHA;
- records checkout provenance as an artifact;
- runs the local recovery unit suite;
- performs a forced-failure integration test by intercepting exactly the direct `git clone`, making it fail, and proving that the independent `git init + fetch + checkout` fallback recovers the correct SHA and file content;
- uses current Node-24-generation GitHub actions and disables the multi-gigabyte shared pip cache for this lightweight guard.

A normal successful primary checkout does not exercise fallback, so the forced-failure integration test is mandatory evidence that the recovery path itself works.

## General research automation rule

The same operational principle applies beyond Git:

- identify a repeated failure signature;
- stop identical retries quickly;
- preserve intermediate evidence/logs;
- change method or reduce the problem to a smaller diagnostic test;
- bind every expensive research run to an immutable source SHA;
- only resume training after dependency, upstream-evidence and dataset invariants pass.

This is the default failure policy for subsequent ML, deep-learning and research workflows in this repository.
