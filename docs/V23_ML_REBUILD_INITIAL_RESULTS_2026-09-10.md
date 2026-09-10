# v0.23 Clean ML Rebuild — Initial Results

Date: 2026-09-10

Branch: `research/v23-ml-rebuild-clean`

Baseline cloned from: `research/v22-ict-vision-lab` at commit `2a5b62c84354b9cbffd35c0bdd6591d4219bf216`.

GitHub Actions run: `34521556267`

Artifact: `v23-clean-ml-rebuild`, artifact ID `10169950949`, SHA-256 `1ee2a81ee947486707cf7c04b3132e8619b4c4059ff9c2fde90179793ba56004`.

Status: **engineering integrity PASS; historical ML promotion FAIL; LIVE remains disabled.**

## 1. Clean rebuild integrity

All three CI jobs completed successfully.

### Core ML integrity

`21 passed`.

Coverage includes:

- duplicate-column rejection;
- future/outcome feature rejection;
- chronological split-overlap rejection;
- validation threshold independence from test mutation;
- development-only unsupervised fit invariance to validation/test mutation;
- concurrent signal-time risk cap;
- no double charging of transaction costs on post-cost R labels;
- model-zoo smoke fit;
- fail-closed promotion logic;
- v0.21 meta-label regression tests;
- v0.20 strategy-regression tests.

### Deep / vision integrity

`19 passed`.

Coverage includes:

- 24 bps round-trip target contract;
- future-mutation invariance of past temporal windows;
- chronological sequence split;
- turnover-cost accounting;
- forward tensor-shape tests for LSTM, GRU, CNN-LSTM, TCN and Transformer;
- LSTM optimization smoke test;
- v0.22c ICT-localization causality tests;
- v0.22d multimodal causality tests.

## 2. Fresh 4h labeled dataset

Source: CoinEx public spot OHLCV, closed 4h bars only.

Markets:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- ADA/USDT
- LINK/USDT
- AVAX/USDT
- DOGE/USDT

The runner requested 6,000 bars per market and generated **5,562 labeled 4h strategy events** from seven frozen v0.20 4h strategy families.

Chronological sample counts:

- development: 3,374
- validation: 1,116
- untouched test: 1,072

Dataset SHA-256:

`f90649a9ef2251fa7d8d75b3b218dcb56ca8e00cc69344b96a7ea1145051cbea`

Audit found **zero exact duplicate events** and no duplicate feature columns. The primary profitable-label rate was 31.55%; mean post-cost R across the entire audit dataset was -0.0214R. These whole-dataset statistics are descriptive and were not used to select the champion.

## 3. Strategy-event coverage

| Strategy | Attempts | Markets | Descriptive profitable rate | Descriptive mean post-cost R |
|---|---:|---:|---:|---:|
| H4_S6_BREAKOUT | 1,295 | 8 | 33.20% | +0.0758R |
| H4_KUMO_TRIANGLE | 1 | 1 | 0.00% | -1.0829R |
| H4_OB_BOS_RETEST | 1,562 | 8 | 30.67% | -0.0520R |
| H4_SUPPLY_DEMAND | 253 | 8 | 37.94% | +0.1109R |
| H4_CORRELATION_DIVERGENCE | 576 | 8 | 27.60% | -0.2319R |
| H4_D1_S6_VOL_RISK | 825 | 8 | 32.61% | +0.0157R |
| H4_D1_OB_BOS_RISK | 1,050 | 8 | 30.67% | -0.0405R |

These values mix chronological segments and therefore are **not champion-selection evidence**. They are kept only as dataset geometry/audit information.

## 4. Supervised classifier tournament

Eight requested classifier families trained successfully:

- Dummy prior
- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Unsupervised context features were fit on development only using KMeans, Gaussian Mixture and IsolationForest, then frozen before validation/test transformation.

The validation-only economic objective selected **Random Forest** with frozen threshold `0.4985528137`.

### Random Forest validation

- selected events: 224
- validation mean R: +0.7006R
- validation profit factor: 2.3888
- validation total portfolio return under research risk aggregation: +28.84%
- validation maximum drawdown: -5.71%
- validation ROC AUC: 0.5375

The high validation economic score did **not** generalize.

### Frozen Random Forest test

- selected events: 223
- test mean R: **-0.2862R**
- test profit factor: **0.6355**
- test total portfolio return: **-13.48%**
- test maximum drawdown: **-15.81%**
- test ROC AUC: **0.4858**
- dependence-aware bootstrap mean timestamp return: -0.1231%
- 95% moving-block bootstrap interval: **[-0.2574%, +0.0663%]**

Official decision:

`NO_MODEL_PROMOTED`

This is a scientifically useful failure: it detects validation instability/regime sensitivity instead of converting a strong validation backtest into a false profitability claim.

## 5. Other classifier test behavior

No classifier produced evidence sufficient for promotion. Test ROC AUCs were near random for this event-level representation, and the validation-selected subsets generally became economically negative in test. CatBoost was the least poor among some economic diagnostics but still had negative mean R and PF below 1.0, so it is not a candidate.

The Dummy prior itself moved from positive validation portfolio economics to strongly negative test economics, which is direct evidence that the underlying mix of strategy-event outcomes changed materially across chronological segments. This makes regime/change-point analysis a priority before further tuning.

## 6. R-multiple regressors

Eight requested regressors also completed successfully:

- Dummy mean
- Ridge
- Random Forest Regressor
- Extra Trees Regressor
- Histogram Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

The strongest validation regressor by mean selected R was Random Forest Regressor (+0.6027R), but on test its selected mean fell to -0.1991R with PF 0.7528. CatBoost Regressor was closest to flat on test (`-0.00022R`, PF `0.9997`) but this is not positive evidence and does not justify promotion.

## 7. Error/warning audit

No fatal ML/data-contract error occurred in the clean run. All framework and historical jobs succeeded.

Non-fatal warnings observed:

1. pandas reports that `origin` has no effect for non-Tick resampling rules in the inherited v0.20 completed-HTF helper. This is a cleanup item, not a failed causal test.
2. the inherited v0.22d robust scaler emits an all-NaN slice warning in an explicit missing-data test; the test still confirms finite output. Frozen v0.22d evidence should not be silently rewritten to hide this warning.
3. PyTorch reports a Transformer nested-tensor optimization warning with `norm_first=True`; this affects optimization path/performance, not output correctness.
4. GitHub Actions reports Node-20 deprecation for current action versions; this is infrastructure maintenance, not model logic.

The prior duplicate ATR-column defect that motivated the clean rebuild did **not** recur: duplicate columns are now fatal by contract and the real dataset passed that assertion.

## 8. Scientific interpretation and next experiment

This run does **not** support simply adding more estimators or relaxing the risk gate. The core problem exposed is temporal non-stationarity: a model selected on validation did not survive the untouched test.

The next experiment must therefore be a new version rather than tuning against this now-observed test. It should preregister:

- walk-forward/CPCV development-validation selection;
- change-point and regime diagnostics;
- candidate-specific or hierarchical strategy models instead of forcing one global mapping across heterogeneous strategy families;
- probability calibration on training/validation only;
- feature-family ablation including supervised-only versus development-fit unsupervised context;
- prior-evidence 4h candidate subset defined without using this test result;
- external-venue/future-period holdout for final adjudication;
- multiple seeds for stochastic estimators;
- cost and risk stress after a predictive candidate exists.

Deep temporal, vision and multimodal state representations remain challengers. RL remains disconnected until a state representation shows robust incremental out-of-sample economic value.

No result in v0.23 authorizes PAPER replacement or real-money LIVE execution.