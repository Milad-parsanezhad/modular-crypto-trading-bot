# Project Status — 2026-09-11

## Purpose

Defense-oriented status map for the MSc research program. Engineering success, scientific promotion, negative evidence and prospective validation are deliberately separated.

## Claim boundary

- `LIVE_EXECUTION = false`
- `paper_replacement_authorized = false`
- no guaranteed-profit claim is authorized
- a green GitHub Actions run proves protocol/engineering execution, not alpha
- previously observed terminal/external data cannot be relabeled as fresh evidence after redesign
- post-boundary prospective evidence is governed by a frozen schedule-only first-look protocol

## Evidence ladder

| Stage | Status | Scientific interpretation |
|---|---|---|
| v0.17 Ichimoku strategy lab | TESTED | S6 locally promising; research/forward-observation only |
| v0.18 cost/regime replication | TESTED / NOT PROMOTED | improvements existed but uncertainty did not justify promotion |
| v0.19 multi-timeframe tournament | TESTED / NOT PROMOTED | lower timeframes were strongly friction-sensitive |
| v0.20 42-candidate tournament | TESTED / EXTERNAL FAIL | internal H4 candidates passed; external winner breached frozen 5% DD ceiling |
| v0.23r ML rebuild | TESTED | leakage/provenance/split contracts hardened |
| v0.24 pooled meta-labeling | REJECTED | `NO_META_MODEL_PROMOTED` |
| v0.24b family meta + portfolio | REJECTED | `NO_FAMILY_META_PROMOTION`; filtered portfolio underperformed and hit DD kill |
| v0.24c external route | BLOCKED | reconstructed model identity differed by one validation selection; economic read stopped |
| v0.24c temporal models | CHALLENGER | TCN/LSTM validation champions; both exceeded 5% validation DD ceiling |
| v0.24d exact frozen replay | TESTED / REPRODUCIBILITY PASS | persisted binaries reproduced exact archived validation selections |
| v0.24d OKX+KuCoin triangulation | REJECTED | event-level edge survived, portfolio PF/DD/bootstrap gates did not |
| v0.25 portfolio ranker | CHALLENGER | `hgb_expected_r` won validation, but spent external transfer diagnostic did not confirm generalization |
| v0.25 prospective ledger | ACTIVE / BLINDED | schedule-only, hash-linked future evidence collection frozen before boundary; no economic first look yet |

## v0.24d — decisive architectural result

Workflow `34585325516`; artifact `10193425439`; digest `sha256:e75423d67476172c78b24c03b5a253257fa0a76b94996fc0557e4d2b85fc8acb`.

Exact replay:

- `H4_S6_BREAKOUT`: `118/236` archived validation selections reproduced exactly;
- `H4_D1_OB_BOS_RISK`: `47/185` reproduced exactly.

### OKX

- 1,868 events / 638 frozen-selected;
- event-level 24 bps PF `1.1817`, mean `+0.1220R`;
- 36 bps PF `1.1306`; 60 bps PF `1.0370`;
- MTM portfolio: 165 accepted, return `-0.861%`, PF `0.9621`;
- max intrabar-stress DD `-5.335%`;
- paired block-uplift lower bound below zero;
- hard MTM kill triggered.

### KuCoin

- 1,869 events / 641 frozen-selected;
- event-level 24 bps PF `1.1975`, mean `+0.1327R`;
- 36 bps PF `1.1447`; 60 bps PF `1.0481`;
- MTM portfolio: 156 accepted, return `+0.315%`, PF `1.0147`;
- max intrabar-stress DD `-5.703%`;
- paired block-uplift lower bound below zero;
- hard MTM kill triggered.

Decision: **`EXTERNAL_REPLICATION_FAILED`**.

Scientific discovery:

> **Event eligibility / alpha filtering and portfolio admission / arbitration are different learning problems.**

The event filter improved broad event-level economics, yet scarce-risk portfolio arbitration did not preserve that advantage.

## v0.25 — risk-aware portfolio ranking

The redesign deliberately kept the event filter frozen and compared a compact allocator/ranking set rather than opening a large new search surface:

- frozen event score baseline;
- frozen score-margin baseline;
- Ridge expected-R;
- HistGradientBoosting expected-R;
- lower-quartile HGB uncertainty-aware priority;
- pairwise logistic ranking challenger.

Workflow `34590214474` — **SUCCESS**.  
Artifact `v25-risk-aware-ranking-34590214474`; ID `10195326409`; digest `sha256:3380bc537b551fd7f62ddbd37f250c9eaacb52d5e1dabd6799b39eac78d4a6bc`.

Validation-only champion: **`hgb_expected_r`**.

| Metric | HGB expected-R | Frozen-score baseline |
|---|---:|---:|
| accepted | 69 | 55 |
| total return | +3.922% | +1.874% |
| PF | 1.3427 | 1.1948 |
| mean R | +0.2295R | +0.1426R |
| max realized DD | -4.672% | -5.201% |

The HGB allocator avoided the frozen 5% validation kill while the baseline crossed it. This is **selection-sample evidence only**.

### Spent external transfer diagnostic

After freezing the ranker, the already-consumed v0.24d OKX/KuCoin data were inspected only as a post-hoc transfer diagnostic:

- OKX: baseline `+4.175%` vs HGB `+3.496%`; HGB DD about `-5.378%`;
- KuCoin: baseline `+7.567%` vs HGB `+5.020%`; HGB DD about `-5.386%`.

No rescue tuning was performed.

Scientific interpretation:

> **Validation improvement did not automatically generalize to the previously observed external regimes.**

Decision: `RANKER_FROZEN_FOR_FUTURE_TIME_RESEARCH`; label `CHALLENGER_NOT_PROMOTED`.

## v0.25 — prospective future-evidence contract

Frozen collector SHA: `fac456ccc0eb445ce7f2d8554840f37da9142ac7`.

Operational future boundary: **`2026-09-11T16:00:00Z`**.

Venues: OKX + KuCoin. Timeframe: 4h. Only certainly completed candles are eligible. Signals must be born at or after the future boundary and their exit bar must be completed before entering the future sample.

Maturity is outcome-independent and requires at least:

- `168` elapsed hours;
- `200` future events per venue;
- at least `5` usable symbols per venue.

Until maturity, economic results are blinded.

The first mature read is terminal for that sample. PASS or FAIL marks the sample spent; same-sample rescue tuning is forbidden.

Frozen economic gate per venue includes:

- at least `50` ranked accepted trades;
- PF >= `1.05`;
- positive realized return;
- return greater than frozen-score baseline;
- absolute MTM drawdown <= `5%`;
- rolling CVaR <= `2%`;
- no hard-DD kill;
- 36 bps stress PF >= `1.0` and mean R > 0;
- paired moving-block-bootstrap uplift lower bound > 0.

Even a pass produces only `FUTURE_TIME_CANDIDATE_PRE_CPCV_PBO_DSR`, not PAPER replacement.

## Prospective Evidence Ledger — operational hardening

Workflow: `.github/workflows/v25-prospective-evidence-ledger.yml` on default branch `main`.

Schedule: `23 0,4,8,12,16,20 * * *` UTC.

The workflow:

- executes the exact frozen collector SHA rather than moving `main`;
- restores the frozen dependency environment;
- downloads exact v0.24b filters and frozen v0.25 ranker;
- preserves first-observed OHLCV bars append-only and records exchange restatements;
- links every accepted artifact to the previous run/artifact/decision hashes;
- writes a complete SHA-256 artifact manifest;
- retains prospective evidence artifacts for 90 days;
- keeps PAPER/LIVE flags false.

### Final pre-boundary governance dry run

Run `34611629491` — **SUCCESS**.  
Artifact `v25-prospective-evidence-ledger-34611629491`; ID `10267919110`; digest `sha256:2dbf88c90c853cb3ada64e365dcb0861fbdae3dbaf2c9541a28f0beb49d4e121`.

State: `WAITING_FOR_FUTURE_BOUNDARY / BLINDED_NOT_STARTED`. Economics were not exposed and the sample was not spent.

A specialist audit closed two additional methodological failure modes **before** the boundary:

1. **Investigator-timed first look:** after the dry run, the production-like ledger was sealed to schedule-only. `push` and `workflow_dispatch` are no longer normal evidence triggers.
2. **Scheduler discontinuity/backfill ambiguity:** a post-boundary accepted-chain gap greater than the frozen `5.5 h` tolerance aborts before new evidence is read.

Trigger-governance and scheduler-continuity JSON records are themselves hashed into prospective chain-link v2.

## Current status / next scientific gate

Current state:

`WAITING_FOR_FUTURE_BOUNDARY / BLINDED_NOT_STARTED`

The correct next action is **not** more tuning. The ranker, future clock, gate and collector are frozen. The system must now accumulate prospective evidence without human intervention.

When maturity is reached, exactly one economic read is permitted. If it fails, v0.25 is rejected on fresh future evidence and any redesign starts a new clock. If it passes, the candidate advances to CPCV/PBO/DSR/search-aware review and then, only if that survives, to prospective PAPER observation.

## Defense interpretation

The research contribution is not a claim that increasingly complex models always increase returns. The defensible contribution is a falsifiable and reproducible framework that:

1. distinguishes signal quality from portfolio economics;
2. detects when encouraging validation does not transfer;
3. preserves negative results;
4. freezes model/data/runtime identity;
5. prevents manual optional-peeking in the future gate;
6. makes future evidence auditable through an immutable hash-linked chain;
7. refuses LIVE authorization while scientific evidence remains incomplete.
