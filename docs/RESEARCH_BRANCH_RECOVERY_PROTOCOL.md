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

## Preferred decision tree

### Remote branch already exists

Reuse it. Do not call `create branch` again.

### Local checkout exists and is clean

Fetch the exact remote ref at depth 1 and check out/reset only the branch ref, not arbitrary user files.

### No local checkout

Attempt one shallow single-branch clone. If it fails or times out, immediately change method rather than repeating clone.

### Fallback method

Initialize an empty repository, add the origin, shallow-fetch exactly `refs/heads/<branch>`, then check out that fetched ref.

### Remote branch missing or remote unreachable

Stop with a diagnostic result. Branch creation is a separate explicit action and must identify the intended base commit/ref.

## Machine-readable audit

`scripts/research_branch_bootstrap.py` can write JSONL audit events. Each event records stage, command, return code, elapsed seconds and whether a timeout occurred. Credentials are never written by the recovery module.

Example:

```bash
python scripts/research_branch_bootstrap.py \
  https://github.com/OWNER/REPO.git \
  research/v23-ml-rebuild-audit2 \
  /tmp/research-checkout \
  --timeout-seconds 60 \
  --max-attempts 2 \
  --audit-log /tmp/research-checkout-audit.jsonl
```

## CI behavior

`.github/workflows/v23r-repo-recovery-guard.yml` gives the normal `actions/checkout` step a one-minute timeout and `continue-on-error`. If checkout fails, the workflow switches to an exact-ref fetch fallback instead of rerunning the same checkout step. A local synthetic Git test suite verifies direct bootstrap, idempotent reuse, missing-branch failure, dirty-worktree preservation and timeout behavior.

## General research automation rule

The same operational principle should apply beyond Git:

- identify a repeated failure signature;
- stop identical retries quickly;
- preserve intermediate evidence/logs;
- change method or reduce the problem to a smaller diagnostic test;
- only resume the expensive research/training job after the blocking invariant passes.

This is the default failure policy for subsequent ML, deep-learning and research workflows in this repository.
