# Research Changelog — Evidence, Decisions and Next Experiments

This changelog is intentionally scientific rather than promotional. Every stage records what changed, what was tested, what failed, and why the next experiment was designed.

## 2026-09-11 — v0.24d exact frozen replay + dual-venue external triangulation

**Status:** TESTED / REJECTED FOR PROMOTION  
**Decision:** `EXTERNAL_REPLICATION_FAILED`

- exact persisted v0.24b model replay passed under the original dependency environment;
- archived selections reproduced exactly: S6 `118/236`, D1-OB `47/185`;
- external universe was pre-registered and disjoint from the original 12-symbol internal panel;
- OKX and KuCoin both showed positive event-level filtered economics after 24/36/60 bps cost stress;
- portfolio-level PF, MTM drawdown and paired-bootstrap lower-bound gates failed on both venues;
- LIVE/PAPER promotion remained false.

**Discovery:** event-level filtering and portfolio admission are separate learning problems. When simultaneous selected events compete for a limited risk budget, deterministic alphabetical tie-breaking can discard economically stronger candidates.

**Next experiment:** v0.25 risk-aware learning-to-rank / portfolio arbitration, followed by genuinely future-time validation. The spent v0.24d sample may be used only for diagnosis/development, never relabeled as fresh test evidence.

## 2026-09-11 — v0.24c fresh-evidence ladder + temporal challengers

**Status:** TESTED / CHALLENGERS ONLY

- LSTM, GRU, TCN, CNN-LSTM and Transformer were run in a development/validation-only tournament;
- TCN became the validation challenger for `H4_S6_BREAKOUT`;
- LSTM became the validation challenger for `H4_D1_OB_BOS_RISK`;
- both exceeded the 5% validation drawdown ceiling and were not promoted;
- external validation initially stopped before outcome reading because reconstruction of the frozen D1-OB model differed by one selected event.

**Discovery:** model name + seed + threshold is not a sufficient scientific identity for persisted estimators. Binary artifact + environment + dataset fingerprint must be frozen.

## 2026-09-11 — v0.24b family-specific meta-labeling + overlap-aware portfolio risk

**Status:** TESTED / REJECTED  
**Decision:** `NO_FAMILY_META_PROMOTION`

- five strategy-family models were sample-eligible;
- family-level local improvements appeared for S6 and D1-OB;
- the base overlap-aware shadow portfolio outperformed the family-filtered portfolio;
- filtered portfolio hit the research drawdown kill.

**Discovery:** event classification improvement does not guarantee portfolio improvement.

## 2026-09-11 — v0.24 strategy-aware meta-labeling

**Status:** TESTED / REJECTED  
**Decision:** `NO_META_MODEL_PROMOTED`

- pooled ML improved mean-R, PF and win rate modestly on the internal test;
- total return fell because coverage dropped;
- bootstrap uplift crossed zero, PBO was high, DSR probability was weak and 36 bps stress was negative;
- S6 showed the strongest local filtering improvement but was not promoted post-hoc.

**Discovery:** prediction quality must be measured as incremental economic value after costs and risk constraints.

## 2026-09-10 to 2026-09-11 — v0.23r ML rebuild and audit hardening

**Status:** ENGINEERING / METHODOLOGY VERIFIED

- unique-timestamp panel split, purge/embargo and label-end-time controls;
- strict feature allow/deny lists;
- external replication gating and immutable provenance;
- repository recovery circuit breaker and forced-failure clone fallback;
- causal upstream vision/multimodal/temporal integrity tests.

**Discovery:** reproducibility and repository recovery must themselves be tested, not assumed.

## Earlier evidence

Earlier stages include the causal Ichimoku strategy lab, multi-timeframe strategy tournament, cost/regime replication, derivatives/order-flow holdout, forward microstructure collection and thesis evidence synthesis. See `RESEARCH_TRACEABILITY_MATRIX.md` and `PROJECT_STATUS_2026-09-11.md` for the complete chain.

## Update policy

Every major experiment should append:

- frozen hypothesis and date;
- source commit / workflow / artifact identity;
- data scope and freshness label;
- validation/test/external/forward distinction;
- scientific decision;
- negative evidence and discovered failure mode;
- next experiment and why it is allowed to use its proposed data.

This file is part of the MSc defense evidence package and must not be rewritten to hide failed experiments.