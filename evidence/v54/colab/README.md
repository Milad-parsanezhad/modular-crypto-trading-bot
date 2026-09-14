# v0.54 Colab Operational Evidence

This directory is the append-only evidence registry for Colab executions of the
v0.54 feature-family audit.

## Safety and scientific status

- Authorized mode: `RESEARCH_ONLY`
- `LIVE_EXECUTION=false`
- `PAPER_EXECUTION=false`
- `BOT_FORWARD_PAPER_ENABLED=false`
- `BOT_PAPER_EXECUTION_ENABLED=false`
- Kraken remains sealed.
- A successful engineering run does not authorize scientific promotion or live
  execution.

## Canonical run inputs

- Branch: `research/v54-feature-audit-oos`
- Operational evidence branch: `evidence/v54-colab-operational`
- Source fingerprint:
  `929eceaff3ae4286183acadb55cd0964deaf674d325783afacfed70e63d67db5`
- Source bundle SHA-256:
  `4b90656f9501d22381ef1e8d3e8c0ac3dbe3f295ef5fe15ba1ca7dca5ec4faee`
- Colab runner SHA-256:
  `eab1d2e0e45b9fc40d56e41698f390f4de503179aaefdf0e63cb386ef36878c1`

The source ZIP is a transport artifact. Git remains the authoritative source;
binary run archives are attached to releases/workflow artifacts when available,
while their cryptographic digests and manifests are committed here.

## Required evidence per run

Each run gets a unique directory under `runs/<run-id>/` containing:

1. `run_manifest.json` — immutable input and environment metadata.
2. `run_status.json` — terminal workflow status.
3. `tests.txt` — exact test command and complete terminal summary.
4. `runner.log` — sanitized execution log.
5. `artifact_manifest.json` — SHA-256 and size of every retained artifact.

Never commit exchange credentials, authorization headers, cookies, tokens,
private URLs, wallet identifiers, or database connection strings.

## Status vocabulary

- `UPLOAD_VERIFIED`
- `DEPENDENCIES_INSTALLED`
- `TESTS_PASSED`
- `BLOCKED_AT_COINEX_CONNECTIVITY`
- `WORKFLOW_COMPLETE`
- `RUN_INCOMPLETE`
- `RUN_REJECTED`

Infrastructure blockage is not a scientific rejection. Partial output is
recorded as partial evidence and must not be promoted to a completed result.

## Current Colab attempt

The user-supplied Colab transcript on 2026-09-14 proves:

- the correct source bundle was uploaded;
- its SHA-256 matched the canonical transport digest;
- editable package installation started and dependencies were resolved;
- no terminal test or workflow result was yet supplied.

It is therefore recorded as `RUN_INCOMPLETE`, not as a successful experiment.

## Registering a new result

Use:

```bash
python scripts/register_v54_colab_evidence.py \
  --run-status /path/to/run_status.json \
  --tests /path/to/tests.txt \
  --log /path/to/runner.log \
  --output evidence/v54/colab/runs
```

The registration utility rejects secrets, computes hashes, preserves the source
files, and refuses to overwrite an existing run.
