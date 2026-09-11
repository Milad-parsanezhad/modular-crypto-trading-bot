# v0.24d — Frozen Snapshot External Triangulation Protocol

**Status:** PRE-REGISTERED / NOT YET PROMOTED  
**Primary objective:** convert the v0.24c reproducibility blocker into a stricter, byte-identical frozen-model validation experiment without reusing the already observed v0.24b terminal sample for selection.

## 1. Research question

Can the exact serialized v0.24b family meta-models, with their original feature schema and frozen thresholds, produce incremental economic value on a pre-registered cross-sectional external universe under realistic costs and mark-to-market portfolio risk **without any model refit, reseeding, threshold retuning or outcome-based venue/symbol selection**?

This experiment is deliberately narrower than “find the best model.” The model-search phase is over for these hypotheses. v0.24d tests whether the already frozen artifact is reproducible and externally transportable.

## 2. Why this stage exists

v0.24c found a one-event mismatch when attempting to reconstruct `H4_D1_OB_BOS_RISK` from model family + seed + threshold:

`got=46 expected=47`

The external economic read was therefore blocked before any external outcome was used. The correct response is not to relax the count, adjust the threshold, or pick a runner-up. The correct response is to consume the exact persisted model object and archived dataset.

Scikit-learn explicitly warns that loading pickle/joblib estimators across different library versions is unsupported/inadvisable, and recommends preserving the training-data snapshot, source code, dependency versions and validation score with the model. v0.24d adopts that persistence contract directly.

## 3. Prior research / comparable systems reviewed before implementation

### 3.1 Backtest-overfitting control

Arian, Norouzi Mobarekeh & Seco (2024), *Knowledge-Based Systems*, DOI `10.1016/j.knosys.2024.112477`, compare out-of-sample testing methods under non-stationarity, autocorrelation and regime shifts and report stronger overfitting control from Combinatorial Purged Cross-Validation than standard alternatives, including lower PBO and stronger DSR behavior.

**Implication for our design:** a successful external replication still does not authorize PAPER by itself; search-aware CPCV/PBO/DSR remains downstream.

### 3.2 Model persistence

Official scikit-learn model-persistence documentation states that cross-version pickle/joblib loading is unsupported/inadvisable and recommends retaining immutable training data, source code, dependency versions and CV evidence.

Reference: `https://scikit-learn.org/stable/model_persistence.html`

**Implication for our design:** v0.24d pins the original persisted dependency versions and verifies byte-level SHA-256 identities before `joblib.load`.

### 3.3 Artifact immutability / cross-run reuse

GitHub Actions documents artifact download by workflow run and supports cross-run artifact retrieval with an authenticated token, repository and run ID. Modern artifact versions expose immutable artifact identities/digests.

References:
- `https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts`
- `https://github.blog/news-insights/product-news/get-started-with-v4-of-github-actions-artifacts/`

**Implication for our design:** CI downloads the exact v0.24b artifact by frozen run/artifact identity instead of regenerating model files.

### 3.4 Comparable quant-platform failure modes

Microsoft Qlib documents both market non-stationarity/model decay and concrete dependency-induced behavioral breakage (for example pandas `groupby` default changes). FinRL issue history also contains timezone/index-drift data-ingestion failures. These are useful engineering analogues: in financial ML, a small data-time or dependency mismatch can alter model inputs even when the high-level strategy name is unchanged.

References:
- `https://github.com/microsoft/qlib`
- `https://github.com/AI4Finance-Foundation/FinRL/issues`

**Implication for our design:** timestamp normalization, data provenance, immutable dependency locks and exact replay are first-class gates rather than post-hoc debugging notes.

## 4. Bugs / methodological traps explicitly prevented

1. **Reconstructing a winner instead of loading the winner.** Model name + random seed + threshold is not treated as model identity.
2. **Cross-version joblib drift.** The original scikit-learn/numpy/pandas/joblib versions are pinned before deserialization.
3. **Rolling-API reconstruction drift.** Exact replay uses the archived v0.24b strategy-event dataset, not a later rolling API reconstruction.
4. **Venue cherry-picking.** External venues and symbols are pre-registered before outcome inspection; both admissible venues are reported, not only the better one.
5. **Outcome-based symbol deletion.** Symbols can be excluded only by predefined availability/data-quality rules; returns cannot determine inclusion.
6. **Same-test rescue tuning.** No v0.24b test/shadow outcome can change model, seed, features or thresholds.
7. **Event-level pseudo-portfolio compounding.** Final economics are evaluated with overlap-aware MTM portfolio simulation, not only independent event multiplication.
8. **Ignoring correlation / tail risk.** New entries remain subject to correlation scaling, CVaR budget, portfolio/strategy/directional risk caps and a 5% hard MTM kill.
9. **Cost fragility.** 24/36/60 bps round-trip stress is mandatory.
10. **Deep-model determinism overclaim.** Temporal challengers remain a separate track; v0.24d does not mix them into the tabular frozen model after observing their validation results.

## 5. Frozen source artifact

Source experiment:

- workflow run: `34578494059`
- artifact ID: `10190676664`
- artifact name: `v24b-family-portfolio-34578494059`
- artifact digest: `sha256:0384391db85922309e7b67e0e0481f2cbf8795cf069ee48f146e36ba857525db`

Critical file identities:

| File | SHA-256 |
|---|---|
| `family_validation_frozen_champions.joblib` | `ec4a81b7d708b0ffc7a80238668fa1bb34cccdac0408c51e4aff082075a5a22a` |
| `strategy_event_dataset.csv` | `fe1a48a8854b9550283db41cc43821ba635b7a63b07eb48df894dd06f337e338` |
| `family_validation_leaderboard.csv` | `dadcbbb36491f643388c70d40a7c27222c749fb9647ffb93d4c974fa85d320a8` |
| `decision.json` | `9daaec9bf3821a16303be5a06f4a706c11f08e03d997c09003c95ea8a952a1a5` |

Persistence environment:

- scikit-learn `1.9.1`
- numpy `2.5.3`
- pandas `3.0.5`
- joblib `1.6.0`
- scipy `1.18.1`
- threadpoolctl `3.6.0`
- cloudpickle `3.1.2`

Any mismatch => `BLOCKED`, before an external economic outcome is read.

## 6. Frozen candidates

No model search is allowed.

### H4_S6_BREAKOUT

- persisted model family: `SGDClassifier` (`sgd_logistic`)
- seed recorded in source experiment: `314`
- threshold: `0.2992513650430247`
- archived validation rows: `236`
- exact expected selected validation rows: `118`

### H4_D1_OB_BOS_RISK

- persisted model family: `LogisticRegression`
- seed recorded in source experiment: `314`
- threshold: `0.5481968244713689`
- archived validation rows: `185`
- exact expected selected validation rows: `47`

The replay gate must recover exactly 118 and 47 selections respectively.

## 7. External universe and venues

Frozen external symbol universe, deliberately disjoint from the v0.24b internal 12-symbol panel:

`ATOM, ETC, FIL, NEAR, UNI, APT, ARB, OP, SUI, INJ, AAVE, TON` versus USDT.

Pre-registered venues:

1. `OKX` public spot OHLCV through CCXT
2. `KuCoin` public spot OHLCV through CCXT

Both are evaluated when predefined data-quality eligibility is satisfied. A venue is not dropped for poor performance.

This experiment is **cross-venue/cross-sectional OOD**, but the calendar overlaps the original market regime and these exchanges have appeared in earlier project research. Therefore even a dual-venue positive result remains `EXTERNAL_REPLICATION_EVIDENCE_PRE_FUTURE_TIME`, not final strategy-level untouched proof.

## 8. Data-quality eligibility

Per venue/symbol:

- minimum 1,200 valid 4h bars;
- UTC timestamps, ascending and unique;
- no outcome-dependent filtering;
- estimated missing-bar fraction <= 2%;
- only spot markets are admissible;
- all fetch/provenance failures persisted.

At least 5 eligible symbols are required for a venue-level economic read.

## 9. Scoring contract

- exact persisted model pipeline;
- exact persisted feature columns extracted from the fitted `ColumnTransformer`;
- frozen threshold only;
- no model refit;
- no recalibration;
- no threshold search;
- no feature addition/removal after external outcome inspection.

## 10. Portfolio-risk contract

Inherited MTM risk contract:

- base risk/trade: `0.25%`;
- max open portfolio risk: `1.00%`;
- max strategy open risk: `0.50%`;
- max directional open risk: `0.75%`;
- max concurrent positions: `5`;
- hard MTM DD kill: `5%`;
- CVaR alpha: `95%`;
- max rolling CVaR loss fraction: `2%`;
- pairwise correlation threshold for full size: `0.80`;
- correlated risk multiplier: `0.50`;
- conservative liquidation reserve: 24 bps.

## 11. Venue-level economic replication gate

A venue is `ECONOMIC_PASS` only if all are true:

- accepted MTM-filtered trades >= 100;
- PF >= 1.05;
- mean accepted R > 0;
- absolute max intrabar-stress DD <= 5%;
- paired moving-block uplift CI lower bound > 0;
- positive-symbol fraction >= 60%;
- 36 bps filtered PF >= 1.00;
- 36 bps filtered mean R > 0.

A venue-level pass is evidence, **not PAPER authorization**.

## 12. Cross-venue decision

- both eligible venues PASS → `DUAL_VENUE_EXTERNAL_REPLICATION_PRE_FUTURE_TIME`;
- at least one eligible venue FAIL → `EXTERNAL_REPLICATION_FAILED` and route to Plan D research redesign;
- fewer than two venues have adequate data/transport → `DATA_INSUFFICIENT_OR_BLOCKED` and route to Plan B future-time accumulation.

No decision can authorize LIVE.

## 13. Required downstream evidence even after dual-venue PASS

1. genuinely future-time Plan B accumulation after the freeze timestamp;
2. CPCV/search-aware retraining audit on the internal model-selection family;
3. PBO and Deflated-Sharpe evidence;
4. fresh-time MTM portfolio risk;
5. prospective PAPER observation;
6. thesis-defense evidence manifest.

## 14. Thesis/defense outputs

The workflow must persist:

- frozen snapshot integrity manifest;
- exact replay table;
- dependency lock / `pip freeze`;
- per-venue data provenance;
- per-venue candidate events and predictions;
- per-venue 24/36/60 bps stress table;
- MTM ledgers and equity curves;
- venue evidence summaries;
- machine-readable `decision.json`;
- source commit SHA and artifact digest.

These artifacts support a defensible Chapter 4 narrative even if the result is negative: the experiment demonstrates exact model identity, controlled external transport and explicit falsification conditions.
