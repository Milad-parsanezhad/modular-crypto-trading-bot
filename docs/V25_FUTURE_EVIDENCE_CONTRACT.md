# v0.25 Blinded Prospective Evidence Contract

## Scientific role

This document freezes the prospective-evidence rules for the v0.25 portfolio-admission ranker. The v0.25 ranker itself was selected on development/validation evidence before this collector contract was finalized. The prospective collector therefore starts on a **later, deliberately conservative operational boundary** so that data ingestion, blinding, first-look semantics and promotion gates are frozen before any eligible outcome can enter the test.

Operational prospective start: **2026-09-11T16:00:00Z**.

No signal born before this timestamp can count as v0.25 prospective promotion evidence. A future event is eligible only when its signal is on/after the boundary and its entire exit bar is completed.

`LIVE_EXECUTION = false`

## Frozen identities

The prospective evaluator must use, without refit or rescue tuning:

- the exact persisted v0.24b family event filters;
- the exact frozen v0.25 ranker artifact from run `34590214474`;
- ranker champion `hgb_expected_r`;
- unchanged event-filter thresholds;
- unchanged ranking feature schema;
- unchanged portfolio-risk semantics;
- UTC-aware, completed 4h candles only;
- explicit source/run/artifact/environment provenance.

Changing the model, threshold, feature set, admissible universe, risk contract or scientific gate after eligible outcomes are read spends the sample and requires a new version and new future clock.

## Venues and universe

Frozen venues:

- OKX public spot OHLCV;
- KuCoin public spot OHLCV.

Frozen symbol universe is the pre-registered external v0.24c universe inherited by `V25FutureEvidenceContract`. Availability and data-quality checks may exclude a symbol; realized performance may not.

## Data-integrity rules

The scheduled collector uses a first-observed, append-only evidence chain.

1. Existing archived OHLCV rows are never silently overwritten by later exchange restatements.
2. Fresh timestamps may be appended after the last archived timestamp.
3. If an exchange changes an overlapping historical OHLCV row, the difference is counted as a restatement and recorded in provenance while the first-observed row remains the scientific record.
4. Every archived symbol frame receives a deterministic SHA-256 fingerprint.
5. Missing-data fraction, first/last timestamp, source and usable/not-usable status are recorded per venue and symbol.
6. The in-progress current candle is excluded. The latest admissible bar is the latest certainly completed 4h candle.
7. CCXT pagination advances on full candle boundaries rather than `last_timestamp + 1 ms` to reduce exchange-specific `since` alignment ambiguity.

The CCXT manual explicitly notes that unified timestamps are UTC milliseconds, returned OHLCV histories may contain gaps, exchange-specific history limits exist, and the latest current candle may be incomplete until the next candle starts. It also recommends continuously storing OHLCV when historical API depth is limited. Those constraints are therefore treated as data-contract requirements rather than ignored implementation details.

## Blind-until-mature rule

Economic statistics are **not read repeatedly** while the sample grows.

Before maturity, scheduled runs may expose only:

- elapsed prospective time;
- usable-symbol counts;
- future event counts;
- gap/restatement provenance;
- source/artifact hashes;
- engineering/fail-closed status.

They must not expose or use for redesign:

- portfolio return;
- profit factor;
- drawdown;
- accepted-trade outcome statistics;
- baseline-versus-ranker economic uplift;
- cost-stress result;
- bootstrap economic confidence intervals.

This makes the first mature economic evaluation a pre-registered first look rather than a sequence of optional peeks.

## Frozen maturity gate

The first economic read is permitted only when all of the following hold:

- at least **168 elapsed hours** from the operational boundary;
- at least **5 usable symbols per venue**;
- at least **200 eligible future events per venue**.

These conditions are intentionally outcome-independent.

If they are not met, state is `INSUFFICIENT_FUTURE_SAMPLE` and economics remain blinded.

## First mature read: portfolio evaluation

At the first mature read, the exact same eligible event set is evaluated under two frozen admission priorities:

1. frozen event-model score baseline;
2. frozen v0.25 `hgb_expected_r` ranker.

Both use the same MTM portfolio engine and the same frozen event-eligibility mask. The ranker only changes admission priority when simultaneous eligible events compete for scarce risk.

The evaluation includes:

- mark-to-market equity;
- intrabar stress drawdown;
- rolling historical CVaR;
- correlation-aware risk scaling;
- open-risk caps;
- hard 5% MTM drawdown kill;
- accepted-trade count;
- portfolio-level 24 / 36 / 60 bps cost stress;
- paired moving-block bootstrap of the time-aligned baseline-versus-ranker close-MTM return differential.

## Frozen economic gates

For each venue, the ranker must satisfy all of the following at the first mature read:

- ranked accepted trades >= **50**;
- profit factor >= **1.05**;
- total realized return > 0;
- ranked total realized return > baseline total realized return;
- absolute maximum intrabar MTM drawdown <= **5%**;
- maximum rolling CVaR loss <= **2%**;
- hard MTM kill not triggered;
- at **36 bps round-trip stress**, ranker PF >= **1.00**;
- at 36 bps, accepted mean R > 0;
- paired portfolio-return uplift bootstrap CI lower bound > 0.

A scientific PASS requires **both OKX and KuCoin** to pass their venue gates.

## Sample-spending / terminal first-look rule

The first mature PASS or FAIL is terminal for this prospective sample.

- PASS: `FUTURE_ALLOCATOR_GATE_PASS_PRE_SEARCH_AUDIT`
- FAIL: `FUTURE_ALLOCATOR_GATE_FAIL`

After either terminal state, later scheduled jobs are prohibited from expanding the same test window and re-reading economics to obtain a more favorable answer. Any redesign begins a new research version and new prospective clock.

## Promotion boundary

Even a prospective PASS does **not** authorize PAPER replacement or LIVE execution. It only creates:

`FUTURE_TIME_CANDIDATE_PRE_CPCV_PBO_DSR`

The surviving candidate must still pass search-aware review (including CPCV/PBO/Deflated-Sharpe-style diagnostics where applicable) and then candidate-specific Forward PAPER observation.

A FAIL becomes:

`REJECTED_ON_FRESH_FUTURE_EVIDENCE`

Negative evidence is retained in the thesis/defense record rather than tuned away.

## GitHub Actions rule

GitHub documents that scheduled workflows run from the **default branch**, and that schedules near the start of an hour can be delayed or dropped under high load. Therefore the production scheduler belongs on `main`, but it must check out an **exact frozen collector commit SHA** from the research branch. The schedule is offset to minute 23 after each four-hour boundary rather than minute 0.

This separates two identities:

- scheduler/orchestrator identity on `main`;
- immutable scientific collector identity at a pinned research SHA.

The scheduler must never silently follow later research-branch commits.

## Claim boundary

A green CI run demonstrates reproducible execution of the frozen protocol. It is not evidence of alpha by itself. No v0.25 result authorizes real-money execution.

`forward_paper_authorized = false`  
`paper_replacement_authorized = false`  
`live_execution_authorized = false`
