# v0.21 Labeled Strategy-Event Meta-ML Protocol

Frozen: 2026-09-10

Status: research / PAPER only. LIVE execution is prohibited by this protocol.

## Objective

v0.21 does not ask a learner to rediscover trading from raw prices. It uses the v0.20 source-derived strategy engine as an event generator, attaches a leakage-safe market snapshot at the exact closed-bar signal time, labels the later realized bracket outcome, and trains a meta-model to answer a narrower question:

> Given that a frozen strategy has emitted a candidate trade now, should the system execute it or abstain?

This is the appropriate role for the first machine-learning layer because the strategy rules, costs, risk controls and labels remain auditable.

## Event labels

Every candidate attempt receives six explicit targets:

- `label_profitable_net`: 1 if post-cost R-multiple is positive, else 0.
- `label_target_hit`: 1 only if the frozen bracket target is reached first.
- `label_ge_1r`: 1 for realized R >= 1.
- `label_ge_2r`: 1 for realized R >= 2.
- `label_outcome_3class`: -1 for R <= 0, 0 for 0 < R < 1, +1 for R >= 1.
- `label_r_multiple`: continuous realized post-cost R.

Outcomes are targets only. Entry/exit/stop/target prices, realized return, R-multiple, exit reason, risk result and post-trade equity are explicitly banned from the feature matrix.

## Point-in-time feature contract

Every model input with prefix `f_` is created from information available at or before `signal_time`. The feature layer includes normalized volatility, short/medium returns, EMA distances/slopes, causal Ichimoku distances, swing distances, retracement state, volume context, BOS/sweep/FVG/supply-demand flags, session flags, peer-break flags, cyclical time features and completed higher-timeframe context.

Higher-timeframe values are merged only after the source HTF candle has closed. The same causal-availability rule used by v0.20 is reused here.

## Split discipline

The inherited frozen chronological segmentation is retained:

- development: model fitting only;
- validation: probability/score threshold selection and champion ranking only;
- test: inspected only after the validation champion and threshold are frozen.

Test performance never feeds back into hyperparameters or champion selection.

## Model zoo

The default scikit-learn tournament covers major practical model families rather than pretending that every algorithm ever published is appropriate for this task.

Classifiers:

- Dummy prior baseline
- Logistic Regression
- Ridge Classifier
- SGD Logistic
- Passive-Aggressive linear classifier
- Linear SVM
- RBF SVM
- Gaussian Naive Bayes
- k-Nearest Neighbors
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting
- AdaBoost
- Multi-Layer Perceptron

When the `ml` extra is installed, the same tournament additionally attempts:

- XGBoost
- LightGBM
- CatBoost

Continuous R-multiple regressors:

- Dummy mean baseline
- OLS Linear Regression
- Ridge
- Elastic Net
- Huber robust regression
- Kernel Ridge
- SVR
- kNN regressor
- Decision Tree regressor
- Random Forest regressor
- Extra Trees regressor
- Gradient Boosting regressor
- Histogram Gradient Boosting regressor
- MLP regressor

Deep sequence models, transformers and reinforcement-learning agents are deliberately not mixed into this event-level tournament. They require sequence tensors, separate validation protocols, multi-seed evaluation and a different target definition; treating them as interchangeable tabular classifiers would be scientifically invalid. They remain the next architecture layer after v0.21 establishes a reliable labeled-event baseline.

## Economic model selection

Classification accuracy alone is not the objective. For each classifier, the validation score distribution is used to freeze an abstention threshold. Selected validation events are assessed by post-cost mean R, profit factor, win rate, coverage, compound account return and maximum drawdown. The validation economic score ranks models.

The frozen validation champion is then evaluated on the untouched test segment. A test pass requires at least 100 selected events, positive mean R, PF >= 1.05 and maximum drawdown <= 5% under the fixed 0.25% research risk scale.

A pass yields only `FORWARD_PAPER_META_CANDIDATE`. It does not replace an existing PAPER strategy automatically and never authorizes LIVE capital.

## Saved evidence

The end-to-end runner saves:

- `labeled_ml_dataset.csv`
- `strategy_label_coverage.csv`
- `data_provenance.json`
- `dataset_manifest.json` including SHA-256 and exact safe feature list
- `classifier_leaderboard.csv`
- `regressor_leaderboard.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- serialized fitted pipelines under `models/`
- `champion.json`
- `decision.json`

This makes each ML conclusion traceable back to a source strategy event and its point-in-time feature snapshot.
