# v0.23r Clean ML Framework + Audit Restart

Frozen restart baseline: commit `2a5b62c84354b9cbffd35c0bdd6591d4219bf216` (v0.22d multimodal-fusion stage).

This branch intentionally rebuilds the machine-learning block and audit layer again from the v0.22d evidence boundary rather than inheriting the later v0.23 implementation wholesale. The purpose is methodological verification, not cosmetic refactoring.

## Why the restart exists

The previous ML work produced useful evidence, but a financial-ML thesis must survive implementation-level scrutiny. The restart explicitly guards against:

1. duplicate feature columns after joins;
2. future/outcome columns entering X through permissive dataframe selection;
3. a row-level panel split placing different assets from the same timestamp into different segments;
4. overlapping label horizons crossing development/validation/test boundaries;
5. using test data to select the model family or abstention threshold;
6. treating symbol/strategy identity as a primary predictive signal without an ablation;
7. interpreting unsupervised clusters as alpha without a separate economic test;
8. confusing unit-exposure prediction economics with strategy RiskEngine position sizing;
9. relaxing cost/risk gates after observing a favorable test result;
10. allowing any historical ML result to authorize LIVE execution;
11. testing GitHub's synthetic PR merge commit while attributing the evidence to a research-branch commit;
12. silently continuing after an upstream localization/multimodal regression;
13. hiding dependency drift behind a shared multi-gigabyte pip cache;
14. calling a three-symbol smoke test a broad cross-asset validation.

## Primary ML contract

- Chronological split: 60% development / 20% validation / 20% test.
- Panel split is performed by **unique timestamps**, not rows.
- An embargo is inserted around development→validation and validation→test boundaries.
- If labels have a future observation window, rows whose `label_end_time` crosses the next segment are purged.
- Default trading friction: 10 bps fee + 2 bps slippage each way = 24 bps round trip.
- `f_*` is the numeric input namespace, but prefix alone is not trusted: outcome-like names are separately denied.
- Default categorical context: timeframe + side.
- Symbol/strategy/family identity is excluded from the primary arm and may only appear in an explicit ablation.
- Development fits parameters.
- Validation chooses threshold and ranks models.
- Test is read-only after freeze.
- Historical test success can create only an ML research candidate; it cannot authorize PAPER replacement or LIVE execution.

## Supervised framework

The clean first-pass tabular registry contains regularized linear and nonlinear baselines:

- Logistic Regression
- Ridge Classifier
- SGD Logistic
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting

These are intentionally smaller than the earlier exhaustive model zoo. The purpose of this restart is to validate the data/split/economic plumbing first. XGBoost/LightGBM/CatBoost and larger hyperparameter searches are added only after the audit layer is green, so compute does not amplify a flawed dataset.

## Unsupervised framework

Unsupervised learning is explicitly separated from direct trade promotion:

- Isolation Forest: anomaly-state discovery;
- Gaussian Mixture (3 states);
- Gaussian Mixture (5 states).

Outputs are regime/anomaly diagnostics. A cluster label has no trading authority until an OOS conditional-performance experiment demonstrates incremental value.

## Deep-learning and vision boundary

LSTM/GRU/CNN-LSTM/TCN/Transformer remain a separate temporal track. They consume causal sequences and require their own multi-seed development/validation/test discipline. The tabular test set must not be recycled to tune the deep track after inspection.

Vision/localization evidence from v0.22c and multimodal evidence from v0.22d are preserved. The active CI DAG now re-runs localization, scientific-liquidity/Wyckoff, multimodal, deep-temporal and earlier vision regression tests before the real-data ML panel job can begin. RL remains downstream and is not connected by this workflow.

## Audit suite

The audit layer checks:

- unique column names;
- timestamp parseability and ordering;
- split labels and strict temporal separation;
- no timestamp group shared across segments;
- strict feature whitelist;
- outcome-name deny-list;
- missingness and infinite values;
- constant-feature burden;
- label validity and balance;
- prefix invariance;
- adversarial future mutation invariance;
- frozen validation threshold/model-selection provenance for test predictions.

Every audit can be persisted as CSV + JSON, with SHA-256 fingerprints.

## Active research CI DAG

The active `.github/workflows/v23r-ml-rebuild-audit.yml` is now a gated DAG rather than two unrelated jobs:

`dependency-integrity`
→ parallel `deterministic-audit-tests` + `upstream-evidence-integrity`
→ `real-data-panel-audit`
→ `final-evidence-gate`

Every job checks out the immutable pull-request head SHA (or push SHA), immediately verifies `git rev-parse HEAD`, and refuses to attribute results to a synthetic PR merge commit. The exact event SHA is also written into the final evidence artifact.

### Dependency integrity

Core and deep/vision jobs are isolated. Lightweight jobs do not restore the previous ~3 GB shared pip cache. Dependencies are installed without pip cache and validated with `pip check`; the real-data artifact records `pip freeze --all` for reproducibility.

### Current GitHub Actions generation

The active workflows use the current Node-24-generation action families verified during this rebuild:

- `actions/checkout@v7`
- `actions/setup-python@v7`
- `actions/upload-artifact@v7`
- `actions/download-artifact@v8`

This removes the Node-20 deprecation warning emitted by the older active workflow definitions.

## Broad real-data panel audit

The former three-symbol BTC/ETH/SOL smoke stage has been replaced in the active DAG by a twelve-symbol CoinEx 4h research panel:

BTC, ETH, SOL, XRP, DOGE, ADA, LINK, LTC, BCH, TRX, AVAX and DOT versus USDT.

The runner requests 2,600 closed 4h bars per market, creates cost-aware open[t+1]→open[t+2] labels, applies timestamp-grouped purging, runs the full dataset audit, trains the supervised registry across frozen seeds and fits unsupervised diagnostics on development only.

The final CI evidence gate requires at least eight symbols with >=500 usable labeled rows and at least 5,000 total labeled panel rows. This is a **broad research smoke/panel gate**, not a substitute for later external-venue or untouched-period replication.

## Artifact contract

The real-data runner saves:

- `ml_research_contract.json`
- `labeled_dataset.csv`
- `provenance.json`
- `ml_dataset_audit.csv`
- `ml_dataset_audit.json`
- `supervised_per_seed_leaderboard.csv`
- `unsupervised_diagnostics.csv`
- `decision.json`
- `ci_provenance.json`
- `pip-freeze.txt`

The terminal `final-evidence-gate` downloads the exact artifact created by its own workflow run and fails if required files are missing/empty, source SHA provenance does not match the event SHA, panel breadth is insufficient, the dataset audit did not pass, or any LIVE/FORWARD authorization flag is unexpectedly true.

## Repository-recovery integration

The separate recovery guard now has a one-minute checkout circuit breaker, exact-SHA verification, complete cleanup of only ephemeral partial checkout state, and a **forced direct-clone failure integration test**. The test proves that the fallback path actually executes `git init + exact-ref fetch + checkout`, recovers the correct commit and preserves expected file content. A normal successful checkout alone is no longer accepted as evidence that fallback works.

## Scientific interpretation rule

A successful CI DAG proves that source provenance, dependencies, upstream representation tests, ML dataset audits, broad-panel execution and artifact integrity all execute under the frozen contract. It does **not** prove alpha. A favorable historical model result remains a research candidate until it survives strategy-level cost/risk integration, multiple-testing-aware robustness, external replication and fresh forward PAPER evidence.
