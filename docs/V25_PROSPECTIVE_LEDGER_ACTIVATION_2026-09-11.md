# v0.25 Prospective Evidence Ledger Activation — 2026-09-11

## Purpose

This document records the activation of the first **default-branch, prospectively scheduled, cryptographically linked evidence ledger** for the v0.25 risk-aware portfolio-ranking challenger.

The scientific purpose is not to optimize performance during observation. It is to prevent the research process from quietly turning future evidence into another tuning set.

## Research motivation

The v0.25 ranker was selected using development/validation evidence only. Previously consumed OKX/KuCoin results are retained as spent diagnostics and cannot promote the model.

The prospective stage therefore uses a sequential/prequential research logic: predictions and portfolio decisions are frozen before newly arriving outcomes are eligible for evaluation. This is consistent with data-stream evaluation literature, where temporal order and test-before-adaptation are central, and with financial-validation work emphasizing model-selection bias, non-stationarity and truly out-of-sample evidence.

Operationally, GitHub documents that scheduled Actions run only from the default branch and may be delayed during high-load periods, especially around minute 0. The scheduler is therefore installed on `main` and runs at minute 23 rather than at the top of the hour.

## Frozen identities

- Frozen collector commit: `fac456ccc0eb445ce7f2d8554840f37da9142ac7`
- Frozen v0.24b event-filter source run: `34578494059`
- Frozen v0.25 ranker source run: `34590214474`
- Frozen ranker champion: `hgb_expected_r`
- Operational prospective boundary: `2026-09-11T16:00:00Z`
- Venues: `OKX`, `KuCoin`
- Timeframe: `4h`
- Scheduler cadence: `23 0,4,8,12,16,20 * * *` UTC

The scheduled workflow checks out the exact frozen collector SHA instead of the moving `main` branch tip.

## Default-branch scheduler

Workflow:

`.github/workflows/v25-prospective-evidence-ledger.yml`

The workflow pins the major GitHub Actions used for checkout, Python setup and artifact upload to exact action commit SHAs. It also restores the frozen model/runtime dependency environment before reading the persisted ranker.

The research-branch workflow is engineering-only and no longer owns the production-like schedule. This prevents duplicate collectors if the research branch is later merged.

## Initial ledger validation

### Bootstrap run

- Workflow run: `34602117989`
- Artifact: `v25-prospective-evidence-ledger-34602117989`
- Artifact ID: `10264906094`
- Artifact digest: `sha256:8d87f86c6cb691dcdc685eb66244ddd840cb35f5b6f7eb586b03bc9f8538e2c9`
- Result: `SUCCESS`
- Scientific state: `WAITING_FOR_FUTURE_BOUNDARY`
- Economics exposed: **no**
- Sample spent: **no**

### Manifest-hardening run

The first run exposed one engineering provenance weakness: the checksum manifest was created before the final `pip freeze` and source-run files were written. No economic information was involved. The scheduler was immediately hardened so the final manifest is generated after all provenance files exist.

Canonical hardening run:

- Workflow run: `34602394915`
- Artifact: `v25-prospective-evidence-ledger-34602394915`
- Artifact ID: `10264374727`
- Artifact digest: `sha256:0ed38eddb141325c06adae072260462440ae45f5deac3e1e4da6a5c1025d31e4`
- Result: `SUCCESS`
- Scientific state: `WAITING_FOR_FUTURE_BOUNDARY`
- Economics exposed: **no**
- Sample spent: **no**

The canonical artifact contains a complete SHA-256 manifest including `decision.json`, frozen collector identity, source-run identities, environment lock, previous-run lineage and chain metadata.

## Pre-boundary trigger-governance hardening

A specialist review identified a second methodology risk before the prospective clock began: a workflow that remains callable by `push` or `workflow_dispatch` after the future boundary could permit an investigator-chosen first look. Even with metric blinding before maturity, allowing the economic first-read job to be manually timed is an avoidable optional-stopping / governance weakness.

Two controls were therefore added **before** the frozen future boundary:

1. **Trigger governance:** after `2026-09-11T16:00:00Z`, any non-`schedule` trigger is rejected before prospective evidence is read.
2. **Scheduler continuity audit:** each accepted run records the previous accepted-run timestamp. A post-boundary scheduled run aborts before reading new evidence if the accepted-chain gap exceeds the frozen `5.5 h` tolerance. This converts GitHub's documented schedule-delay/drop behavior into an explicit fail-closed scientific condition rather than silently backfilling an unknown observation gap.

The controls were validated by a final pre-boundary dry run:

- Workflow run: `34611629491`
- Artifact: `v25-prospective-evidence-ledger-34611629491`
- Artifact ID: `10267919110`
- Artifact digest: `sha256:2dbf88c90c853cb3ada64e365dcb0861fbdae3dbaf2c9541a28f0beb49d4e121`
- Trigger event: `push`
- Trigger governance: `allowed = true` because the run occurred before the boundary
- Previous accepted run: `34602394915`
- Observed chain gap: `1.5927777778 h`
- Continuity status: `OK`
- Scientific state: `WAITING_FOR_FUTURE_BOUNDARY`
- Economics exposed: **no**
- Sample spent: **no**
- Chain format: `v0.25-prospective-evidence-chain-v2`

The artifact hashes `trigger_governance.json` and `scheduler_continuity.json` into its chain link, in addition to the previous artifact digest, previous decision SHA-256 and frozen model/collector identities.

After that dry run succeeded, commit `ebbe06fc633babd74551d313d16db38d4aa5d114` removed both `push` and `workflow_dispatch` from the production-like ledger. The ledger is now **schedule-only**. Any future engineering change to the workflow must be treated as a protocol change and documented rather than used to manually interrogate the future sample.

## Cryptographic lineage

The ledger artifacts explicitly link to the previous accepted artifact using:

- previous run ID;
- previous artifact digest;
- previous decision SHA-256;
- previous chain-link SHA-256;
- current trigger-governance SHA-256;
- current scheduler-continuity SHA-256;
- current decision SHA-256;
- current frozen-snapshot identity SHA-256.

This creates an auditable hash-linked evidence sequence. A later scheduled artifact records the identity of the previously accepted artifact rather than silently replacing it.

## Blinding and first-look rule

Before maturity, scheduled runs may expose only coverage/provenance information needed to know whether the sample is large enough. They may **not** expose the comparative economic read used for scientific promotion.

The first mature read is permitted only after the frozen maturity conditions are satisfied, including:

- at least `168` elapsed hours;
- at least `200` future events per venue;
- minimum usable symbol coverage per venue.

The first mature PASS or FAIL is terminal for that prospective sample. The same future window cannot be expanded and re-read repeatedly to rescue a failed result.

## Portfolio gate after maturity

The frozen economic gate requires, per venue, sufficient ranked accepted trades plus:

- positive portfolio return;
- ranker return greater than frozen-score baseline;
- profit factor threshold;
- MTM drawdown within the 5% research ceiling;
- rolling CVaR within the frozen limit;
- no hard-DD kill;
- survival at the 36 bps transaction-cost stress level;
- positive paired moving-block bootstrap lower bound for ranker-vs-baseline portfolio return uplift.

Only a survivor advances to the CPCV/PBO/DSR search-aware audit. This stage does **not** authorize PAPER replacement or LIVE execution.

## Current scientific status

`BLINDED_NOT_STARTED / WAITING_FOR_FUTURE_BOUNDARY`

`forward_paper_authorized = false`

`paper_replacement_authorized = false`

`live_execution_authorized = false`

Because the future boundary is 16:00 UTC and a 4h candle must be completed before use, the scheduled run at 16:23 UTC is still expected to remain pre-boundary for eligible completed-bar evidence. The first scheduled run capable of observing the completed 16:00–20:00 UTC candle is the 20:23 UTC run, subject to normal GitHub Actions scheduler delay and the continuity guard.

## Defense value

This stage provides direct evidence for the following thesis-defense questions:

1. How was optional stopping / repeated future peeking controlled?
2. How was model identity frozen before future evaluation?
3. How were exchange-history restatements handled?
4. How can the examiner verify that a future artifact was not silently replaced?
5. How were scheduler delay/drop risks converted into an explicit fail-closed rule?
6. Why does a green CI run prove protocol integrity but not alpha?
7. Why is validation improvement insufficient without fresh portfolio-level evidence?

## External references

- GitHub Actions scheduled events: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- GitHub Actions workflow syntax: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- Brzezinski & Stefanowski, *Prequential AUC: properties of the area under the ROC curve for data streams with concept drift*, Knowledge and Information Systems (2017).
- Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*, Journal of Computational Finance (2015).
