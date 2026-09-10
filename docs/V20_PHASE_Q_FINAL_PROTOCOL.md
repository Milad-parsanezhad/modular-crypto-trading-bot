# v0.20 Phase-Q Final Prospective Microstructure Quality Protocol

Date: 2026-09-10

Status: **PRE-REGISTERED BEFORE THE COUNTABLE PILOT WINDOW**

Protocol version: `v0.20-phase-q-1`

## Objective

The sole objective of Phase Q is to determine whether the prospective public REST microstructure measurement layer is sufficiently complete, synchronized and reproducible to justify freezing a 4-hour feature specification. Phase Q does **not** test profitability, select a trading rule or authorize model fitting.

## Frozen time boundary

- Pilot start: `2026-09-10T13:00:00Z`
- Earliest maturity: `2026-09-17T13:00:00Z`
- Measurement target cadence: 30 minutes
- Forecast/trading horizon: 4 hours
- Minimum elapsed time: 168 hours
- Minimum nominal opportunities: 336

Data created before the pilot start cannot be backfilled into maturity.

## Countable evidence contract

A snapshot counts toward Phase Q only when all of the following are true:

1. `version == v0.19` and research status is the prospective multi-venue microstructure collection status;
2. final payload SHA-256 is valid;
3. `phase_q_protocol.protocol_version == v0.20-phase-q-1`;
4. `generated_at >= 2026-09-10T13:00:00Z`;
5. GitHub event is `schedule`;
6. Git ref is exactly `refs/heads/main`;
7. the no-signal, no-PAPER-replacement and no-LIVE authorization invariants remain false;
8. the fixed-window microstructure contract remains unchanged.

Manual `workflow_dispatch`, pull-request smoke runs, branch runs, duplicate hashes, duplicate snapshots inside the same 30-minute measurement slot, tampered payloads and backfilled rows cannot increase maturity or coverage.

## Frozen core Gate Q

For **each** of BTC/USDT and ETH/USDT:

- authorized coverage ratio >= 0.80;
- p95 cross-venue clock skew <= 60 seconds;
- p95 accepted-trade staleness <= 30 seconds.

Globally:

- elapsed time >= 168 hours;
- expected 30-minute opportunities >= 336.

The gate is conjunctive: every required condition must pass.

## Extended diagnostics

The following are recorded as operational diagnostics and warnings without silently changing the frozen core gate:

- countable vs excluded snapshot count;
- duplicate-slot ratio;
- longest run of missing measurement slots between observed snapshots;
- snapshots containing provider failures;
- provider-failure event count;
- per-venue observation/acceptance counts;
- unique snapshot hashes;
- SHA-256 of the ordered evidence ledger.

Diagnostic warnings are retained for thesis interpretation even when the core gate passes.

## Immutable evidence architecture

The 30-minute collector uploads one immutable JSON artifact per GitHub Actions run. A separate v0.20 monitor harvests the retained `v19-forward-microstructure-*` artifacts through the GitHub Actions artifact API, reconstructs the evidence directory, filters countable evidence and produces:

- `artifact_harvest_manifest.json`;
- `phase_q_quality_gate.json`;
- `PHASE_Q_QUALITY_REPORT.md`.

The monitor runs every six hours. Its cadence is reporting-only and does not create extra market observations.

GitHub Actions artifacts are intentionally used as immutable run outputs rather than rewriting a rolling research file. Artifact IDs, source SHA values, timestamps and digests remain independently auditable.

## Scheduler interpretation

GitHub scheduled workflows may execute later than their nominal cron time. The project therefore treats the **actual stored observation/generation timestamps** as authoritative. No synthetic timestamp correction is applied. Long gaps are reported explicitly rather than filled.

## Passing Gate Q does not authorize model fitting

`PHASE_Q_PASSED_FEATURE_FREEZE_ALLOWED` authorizes exactly one next action:

`FREEZE_4H_FEATURE_SPEC`

It does **not** authorize:

- predictive model fitting;
- alpha claims;
- PAPER strategy replacement;
- testnet promotion;
- LIVE execution.

A separate Phase-P protocol must first freeze the 4-hour feature construction, target, trial registry, models, costs and inferential comparisons before any predictive evaluation begins.

## Phase-P minimums already reserved

The planned predictive stage remains:

`PRICE_ONLY` vs `PRICE_PLUS_MICROSTRUCTURE`

with at least 250 independent quality-filtered 4-hour decision timestamps, point-in-time feature construction, explicit transaction costs, dependence-aware inference, FDR across planned comparisons and search-aware correction if multiple adaptive candidates are tried.

No feature definition may be selected by looking at the future target during Phase Q.

## Stop / redesign rules

Phase Q should be redesigned rather than force-passed when persistent structural provider limitations prevent acceptable coverage or synchronization. A failed pilot is a valid thesis result. If the REST measurement design proves structurally inadequate, a separately registered WebSocket/L2 source may be opened as a new evidence stage; it must not be silently mixed with the current REST pilot.

## Scientific boundary

Phase Q answers: **is the new data stream mature and trustworthy enough to define the next experiment?**

It does not answer: **is the strategy profitable?**
