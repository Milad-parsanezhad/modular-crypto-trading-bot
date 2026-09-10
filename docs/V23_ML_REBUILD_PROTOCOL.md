# v0.23 Clean ML Rebuild Protocol

Frozen start point: commit `2a5b62c84354b9cbffd35c0bdd6591d4219bf216` from `research/v22-ict-vision-lab`.

Status: research / PAPER only. LIVE execution remains prohibited.

## Why this branch exists

The ML stack is being rebuilt from a clean v0.22d baseline rather than incrementally patching the previous v0.23 branch. The goal is to make every data transformation, split, model-selection decision, economic metric and promotion gate auditable.

The previous v0.23 attempt discovered at least one concrete data-contract defect: a duplicate ATR-derived feature column could enter the generated ML dataset. That class of error is now prevented by a hard duplicate-column assertion before any learner is fit.

The v0.22d multimodal artifact is retained as prior evidence, not as a new tuning target. Its frozen result was `MULTIMODAL_REPRESENTATION_CANDIDATE` with validation AUC gain +0.03584 for supported fusion, test AUC 0.63879 and test balanced accuracy 0.60995. The scientifically supported liquidity-only increment was negative, and course-specific Wyckoff proxies also reduced AUC. Those facts are carried forward as evidence boundaries; the same test sample must not be repeatedly tuned against.

## Core scientific contract

1. **Point-in-time features only.** Every feature supplied to a learner must be computable at or before `signal_time`.
2. **Outcome isolation.** Entry/exit prices, stop/target prices, realized PnL/R, exit reason, post-trade equity and all `label_*` columns are targets or diagnostics only.
3. **Unique columns.** Duplicate feature names are a fatal error.
4. **Chronological split.** Development fits parameters; validation selects model/threshold; test is read once after freeze.
5. **Development-only preprocessing.** Scaling, imputation, clustering, anomaly models, PCA-like transforms and representation fitting must never use validation/test distributions.
6. **Economic objective.** Accuracy/AUC are diagnostics. Selection requires post-cost economic value, adequate sample size and risk compliance.
7. **Concurrent-signal accounting.** Simultaneous strategy events are aggregated by decision timestamp with a portfolio-risk cap rather than naively compounded as sequential independent bets.
8. **Multiple-seed stability.** Stochastic learners must be evaluated across multiple seeds before promotion.
9. **No test-driven repair.** Once test metrics are observed, architecture/hyperparameter changes create a new experiment/version and require a fresh holdout.
10. **Fail closed.** Missing provenance, invalid split, duplicate columns, leakage, insufficient events or failed risk gates produce `NO_MODEL_PROMOTED`.

## ML layers in v0.23

### Supervised tabular models

The rebuilt tournament includes linear, margin, nearest-neighbor, tree/ensemble, boosting and neural MLP families. Optional XGBoost/LightGBM/CatBoost are used when installed.

Primary classification target: post-cost profitable strategy event (`label_profitable_net`).

Secondary regression target: post-cost realized R (`label_r_multiple`).

### Unsupervised layer

KMeans, Gaussian-mixture state assignment and IsolationForest anomaly scores are fitted on development features only. They are treated as regime/context features, never as proof of alpha.

### Deep temporal layer

LSTM, GRU, TCN and Transformer sequence models are rebuilt behind a separate causal sequence contract. Windows end at `t`; targets begin after `t`; all normalization is fit from development/past information only. Deep models remain challengers until they beat simple baselines economically and across seeds.

### Vision / multimodal layer

v0.22c localization passed its frozen representation gate. v0.22d multimodal fusion also passed its frozen representation gate, but this does not authorize trading. In v0.23 the visual embedding can be used only as a frozen or newly preregistered feature family, with an explicit ablation against numeric-only models.

### Reinforcement learning

RL remains disconnected from trading state in this rebuild. It can be enabled only after the supervised/temporal/multimodal state representation passes its own out-of-sample and risk gates. RL will first be evaluated for execution/allocation under transaction costs rather than being allowed to manufacture directional alpha from noise.

## Risk and capital-management contract

Default research risk per selected event: 0.25% of equity.

Maximum aggregate risk opened at one decision timestamp: 1.0% of equity.

Maximum research drawdown gate: 5%.

Research execution includes existing post-cost R labels; costs must not be subtracted twice. Separate bar-level experiments must explicitly model maker/taker fees, slippage, spread and funding when applicable.

A test candidate must have at least 100 selected events, positive mean post-cost R, profit factor >= 1.05, maximum drawdown <= 5%, and a non-negative dependence-aware bootstrap lower bound before it may be called a `FORWARD_PAPER_CANDIDATE`.

## Required test suite

The branch must automatically test:

- duplicate-column rejection;
- banned outcome/leakage rejection;
- chronological split ordering;
- threshold independence from test data;
- development-only unsupervised fitting;
- concurrent portfolio-risk cap;
- cost/R accounting contract;
- deterministic seeds where expected;
- model-zoo smoke fitting;
- fail-closed promotion logic;
- deep sequence future-mutation invariance;
- LSTM/GRU/TCN/Transformer tensor-shape smoke tests.

No result in this branch authorizes real-money execution.