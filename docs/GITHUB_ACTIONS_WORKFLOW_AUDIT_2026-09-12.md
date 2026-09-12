# GitHub Actions Workflow Audit — 2026-09-12

Status: engineering/governance audit. This document does not alter any frozen scientific result.

## Current authoritative research workflow

v0.43 uses a three-stage DAG:

1. `prepare-v43`
   - installs the research package;
   - verifies frozen lineage and invariants;
   - downloads public development data only;
   - performs outcome-independent data-quality screening;
   - builds causal features and settled event labels;
   - freezes the five fold index;
   - uploads a prepared checkpoint.

2. `fold-v43` matrix (`1..5`)
   - each fold downloads exactly the prepared checkpoint from the same workflow run;
   - trains one asset-specific model across development venues for that asset;
   - performs timestamp-cluster moving-block perturbations;
   - calibrates and predicts OOS;
   - uploads independent fold evidence.

3. `finalize-v43`
   - waits for all five fold jobs;
   - concatenates OOS predictions;
   - applies non-overlap and the independent-by-venue Financial Governor;
   - computes fold, venue, breadth, block-CI, cluster-CI, stress, forecast-skill and perturbation gates;
   - writes one canonical decision and a 90-day evidence artifact.

This DAG completed successfully for authoritative v0.43 run `34689018365`.

## Audit findings

### 1. PR checkout provenance ambiguity — FIX FOR FUTURE WORKFLOWS

`actions/checkout@v4` without an explicit `ref` on `pull_request` checked out PR merge ref `26d310ec...`, not scientific head `c13d113...`.

For v0.43, the merge-ref audit showed that the only extra file was `docs/V42_RESULTS_2026-09-12.md`; no scientific code differed, so the frozen result is unaffected.

**Required from v0.44 onward:**

```yaml
- uses: actions/checkout@v4
  with:
    ref: ${{ github.event.pull_request.head.sha || github.sha }}
```

Every canonical artifact must also contain an `execution_provenance.json` recording:
- workflow run id;
- checked-out SHA from `git rev-parse HEAD`;
- event head SHA;
- Python version;
- package version;
- core dependency versions;
- preregistration file digest;
- prepared-data/event-table digest.

### 2. Legacy v17 ran on every PR — FIXED IN MAINTENANCE BRANCH

Historical `.github/workflows/v17-strategy-lab.yml` had an unscoped `pull_request:` trigger. It therefore consumed a runner on every modern research commit.

Maintenance branch now limits it to its v0.17 code/tests/workflow and adds concurrency cancellation.

### 3. Historical Kraken workflow remained manually dispatchable — HARD-DISABLED

Historical v0.34 Kraken workflow targeted an obsolete v0.33 candidate and exposed `workflow_dispatch`. This creates a holdout-consumption risk inconsistent with current governance.

Maintenance branch turns it into an archived, skipped guard. Any future Kraken evaluation must be a **new preregistered one-shot workflow** tied to a current development winner.

### 4. Historical production/paper smoke conflicted with current policy — HARD-DISABLED

Historical `v1-production-smoke.yml` expected `paper_execution_enabled=True` and invoked `/paper/run-once`. Current research governance is `PAPER=false`, `LIVE=false`.

Maintenance branch turns this workflow into an archived, skipped guard. Paper execution requires a future explicit promotion experiment.

### 5. Evidence retention is incomplete — FIX REQUIRED FOR v0.44

The canonical final v0.43 artifact is retained for 90 days, but prepared/fold artifacts are retained for only 7 days. The final artifact contains aggregated predictions/trades/diagnostics but not every preparation manifest.

**Required from v0.44 onward:** final canonical artifact must copy and retain:
- availability manifest;
- data-quality manifest;
- series/data manifest;
- eligible-universe manifest;
- fold index;
- prep status;
- hashes of the prepared event table;
- fold diagnostics;
- model/perturbation diagnostics;
- final OOS predictions/trades;
- decision JSON;
- execution provenance JSON.

The frozen v0.43 prepared event table SHA-256 is:
`e9bf3875e8dbae762e11f7abc80fb70af3a8d857660826260a966d1d3c1bf076`.

### 6. Runtime warnings / performance debt

Observed warnings during v0.43 preparation:
- pandas resample warning: `origin` is ignored for non-Tick-like `1d/7d` rules;
- DataFrame fragmentation warnings during causal normalization / venue indicator insertion;
- sklearn deprecation warning for explicit logistic `penalty` parameter in vectorized equivalence tests.

These did not change the frozen v0.43 decision, but they should be resolved before v0.44 where doing so does not alter the frozen downstream hypothesis.

For the 4h thesis path, daily completed-context behavior is calendar-day aligned and remains causal. Weekly alignment in the historical v0.20 code should not be silently changed inside v0.44 because that would confound the sampling ablation; any semantic weekly realignment requires a separate experiment/version.

## Required v0.44 workflow contract

### Trigger
- PR only to the v0.43 frozen research base (or an explicitly named v0.44 base branch).
- `workflow_dispatch` allowed for rerunning the exact frozen snapshot only.
- path filtering for v0.44 scientific files.
- `concurrency.cancel-in-progress: true`.

### Snapshot integrity
- explicit head-SHA checkout;
- `permissions: contents: read` unless a job proves it needs more;
- no secrets/exchange credentials;
- Kraken client construction prohibited by test and runtime guard;
- no PAPER/LIVE endpoint invocation.

### DAG

`research-ci-v44` → independent invariant/static tests.

`prepare-v44` → download once, quality-screen, freeze raw 4h panel and event-sampling variants, write hashes.

`fold-v44 [1..5]` → parallel purged walk-forward, same frozen model family and risk/economic gates.

`finalize-v44` → aggregate, compare the preregistered sampling variants, apply multiplicity-aware decision rule, copy **all** manifests into one 90-day canonical artifact.

### Scientific isolation

v0.44 may change **sampling/event clock only**. It must not simultaneously add Transformer/RL/on-chain/LOB features, relax thresholds, prune assets based on outcomes, or touch Kraken.

## Active-path principle

The repository contains many historical workflows because the thesis is an auditable research lineage. Historical workflows should remain visible, but they must not compete for runners or expose obsolete live/holdout actions. The active research path should be small and explicit: current CI + current characterization + frozen evidence. Everything else is archival unless a new preregistration explicitly reactivates it.
