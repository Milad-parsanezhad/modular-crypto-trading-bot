# v0.23r Clean ML Framework + Audit Restart

Frozen restart baseline: commit `2a5b62c84354b9cbffd35c0bdd6591d4219bf216` (v0.22d multimodal-fusion stage).

This branch intentionally rebuilds the machine-learning block and audit layer again from the v0.22d evidence boundary rather than inheriting the later v0.23 implementation wholesale. The purpose is methodological verification, not cosmetic refactoring.

## Why the restart exists

The previous ML work produced useful evidence, but a financial-ML thesis must survive implementation-level scrutiny. The restart explicitly guards against:

1. duplicate feature columns after joins (including the previously observed duplicate-ATR class of bug);
2. future/outcome columns entering X through permissive dataframe selection;
3. a row-level panel split placing different assets from the same timestamp into different segments;
4. overlapping label horizons crossing development/validation/test boundaries;
5. using test data to select the model family or abstention threshold;
6. treating symbol/strategy identity as a primary predictive signal without an ablation;
7. interpreting unsupervised clusters as alpha without a separate economic test;
8. confusing unit-exposure prediction economics with the strategy RiskEngine's position sizing;
9. relaxing cost/risk gates after observing a favorable test result;
10. allowing any historical ML result to authorize LIVE execution.

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
- Historical test success can create only `ML_RESEARCH_CANDIDATE`; it cannot authorize PAPER replacement or LIVE execution.

## Supervised framework

The clean first-pass tabular registry contains regularized linear and nonlinear baselines:

- Logistic Regression
- Ridge Classifier
- SGD Logistic
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting

These are intentionally smaller than the earlier exhaustive model zoo. The purpose of this restart is to validate the data/split/economic plumbing first. XGBoost/LightGBM/CatBoost and larger hyperparameter searches are added only after the audit layer is green, so that compute does not amplify a flawed dataset.

## Unsupervised framework

Unsupervised learning is explicitly separated from direct trade promotion:

- Isolation Forest: anomaly-state discovery;
- Gaussian Mixture (3 states);
- Gaussian Mixture (5 states).

Outputs are regime/anomaly diagnostics. A cluster label has no trading authority until an OOS conditional-performance experiment demonstrates incremental value.

## Deep-learning boundary

LSTM/GRU/CNN-LSTM/TCN/Transformer remain a separate temporal track. They should consume causal sequences and be selected across multiple seeds with development/validation/test discipline. The tabular test set must not be recycled to tune the deep track after inspection.

Vision/localization evidence from v0.22c and multimodal evidence from v0.22d are preserved, but this restart does **not** connect visual embeddings to reinforcement learning. RL remains downstream of stable localization, multimodal and temporal representations plus strategy-level economic validation.

## Audit suite

The new audit layer checks:

- unique column names;
- timestamp parseability and ordering;
- split labels and strict temporal separation;
- no timestamp group shared across segments;
- strict feature whitelist;
- outcome-name deny-list;
- missingness and infinite values;
- constant-feature burden;
- label validity and balance;
- prefix invariance (adding future rows cannot alter historical features);
- adversarial future mutation invariance;
- frozen validation threshold/model-selection provenance for test predictions.

Every audit can be persisted as CSV + JSON, with SHA-256 of the audit report and SHA-256 fingerprint of the dataframe.

## Real-data smoke experiment

`scripts/run_v23r_ml_rebuild_audit.py` fetches closed CoinEx 4h spot bars for BTC/USDT, ETH/USDT and SOL/USDT, constructs a deliberately compact causal market-feature panel, creates open[t+1]→open[t+2] cost-aware labels, applies timestamp-grouped purging, runs the full audit, trains the supervised registry across three frozen seeds, and fits unsupervised diagnostic models on development only.

The real-data runner saves:

- `ml_research_contract.json`
- `labeled_dataset.csv`
- `provenance.json`
- `ml_dataset_audit.csv`
- `ml_dataset_audit.json`
- `supervised_per_seed_leaderboard.csv`
- `unsupervised_diagnostics.csv`
- `decision.json`

## Scientific interpretation rule

A successful CI run proves that the framework and audit execute under the frozen contract. It does **not** prove alpha. A favorable historical model result is only a research candidate until it survives strategy-level cost/risk integration, multiple-testing-aware robustness, external replication and fresh forward PAPER evidence.
