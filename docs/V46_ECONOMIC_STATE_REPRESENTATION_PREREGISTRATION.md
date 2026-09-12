# v0.46 — Economic-State Representation Redesign Preregistration

Date frozen: 2026-09-12  
Status: **PREREGISTERED BEFORE v0.46 EMPIRICAL EXECUTION**

## Motivation

The canonical v0.45 diagnostic localized the v0.44 failure to three interacting issues:

- weak broad TARGET/STOP separability under the frozen representation;
- large reliability/calibration error relative to the training-only empirical baseline;
- a large TIME mass that is economically heterogeneous and materially positive on average.

Canonical v0.45 evidence:

- workflow run: `34703156785`
- scientific head: `f04dc6196e9284051ef6a5c7be92f74f5739f6a2`
- artifact: `10301201258`
- artifact digest: `sha256:40f59f420248bce82b7be8ad563e5244ef5651c9ccb397325b60a4d0d0048f15`
- result: `V45_DIAGNOSTIC_COMPLETE / NO_PROMOTION`

The next admissible question is therefore whether a more economically aligned terminal-state representation can improve out-of-sample probability skill and economic selection **without adding model capacity or new data sources**.

## Scientific hypothesis

The frozen competing-risk labels `TARGET / STOP / TIME` may compress economically distinct timeout paths into one terminal class. A challenger representation that distinguishes positive from non-positive timeout outcomes may expose useful structure that a simple probabilistic learner can exploit more reliably.

This is a representation hypothesis, not a claim that positive timeout events should be selected after the fact.

## Frozen arms

Exactly two arms are allowed:

1. `R0_FROZEN_THREE_STATE_BASELINE`
   - terminal labels remain `TARGET / STOP / TIME`;
   - simple multinomial logistic regression is fit on the exact frozen v0.41 feature set;
   - this is a representation/learner-matched control, not the historical HistGB competing-risk model.

2. `R1_FOUR_STATE_TIMEOUT_SIGN`
   - `TARGET` remains `TARGET`;
   - `STOP` remains `STOP`;
   - `TIME` with realized post-cost `net_r > 0` becomes `TIME_POSITIVE`;
   - `TIME` with realized post-cost `net_r <= 0` becomes `TIME_NONPOSITIVE`.

The zero threshold is fixed because it is the economic sign boundary. It is not tuned.

## Frozen learner

Both arms use the same deliberately simple model:

- scikit-learn multinomial `LogisticRegression`;
- `solver="lbfgs"`;
- `C=1.0`;
- `max_iter=2000`;
- no class-weight tuning;
- no feature selection;
- no hyperparameter search;
- one model per asset trained jointly across CoinEx / OKX / KuCoin;
- the exact frozen `V41_FEATURES` plus frozen venue indicators used in v0.44;
- finite missing values are handled exactly as already prepared in v0.44.

The purpose is to test representation before model capacity.

## Data and fold lock

The experiment reuses the exact prepared v0.44 event table and exact five common purged chronological fold boundaries from canonical run `34700944062`.

This evidence is already consumed development evidence. Therefore:

- v0.46 may compare frozen arms on development data;
- a positive result can only nominate a frozen development candidate;
- it cannot authorize Kraken, PAPER or LIVE by itself;
- no same-sample retuning is allowed after results.

Kraken remains sealed and must not be instantiated or fetched.

## Training / calibration / test discipline

For every asset and every frozen fold:

- fit uses only events whose labels are fully resolved before the calibration boundary;
- calibration uses only the frozen calibration interval;
- test uses only the frozen OOS interval;
- no timestamp may cross the frozen fold boundaries;
- assets with insufficient support fail closed.

### Probability calibration

No additional flexible calibrator is introduced. The logistic softmax probabilities are evaluated directly. This deliberately tests whether the representation itself improves reliability.

## Frozen forecast metrics

For each arm, fold and asset:

- multiclass Brier score;
- log loss where all required classes are supported;
- classwise one-vs-rest Brier score;
- classwise reliability / resolution decomposition using ten fixed equal-width probability bins;
- macro one-vs-rest AUC where mathematically defined.

The primary representation comparison is:

- median multiclass Brier score across supported asset-fold units;
- median reliability error across supported units;
- positive-fold fraction for the economic score described below.

## Frozen economic reconstruction

For each training set only, estimate the mean realized `net_r` for each terminal state. Test-event expected R is reconstructed as:

`E[R|x] = sum_k P(state=k|x) * mean_train_R(state=k)`

No test outcome enters these state values.

For R0 the states are `TARGET / STOP / TIME`.
For R1 the states are `TARGET / STOP / TIME_POSITIVE / TIME_NONPOSITIVE`.

### Admission rule

A test event is model-admissible only when:

- reconstructed expected R is strictly positive;
- its prediction is finite;
- the asset/fold has all required training-state support.

No probability threshold is tuned.

The existing Financial Governor is then applied unchanged to admissible events for economic characterization.

## Frozen comparison gates

`R1_FOUR_STATE_TIMEOUT_SIGN` earns **development-candidate** status only if all of the following hold:

1. it has lower median multiclass Brier score than R0;
2. it has lower median reliability error than R0;
3. at least 3/5 folds have positive post-cost expectancy after model admission and Financial Governor;
4. aggregate post-governor expectancy is positive;
5. aggregate post-governor profit factor is at least 1.05;
6. maximum account drawdown is no worse than 5%;
7. the 36 bps stress profit factor is at least 1.00;
8. no frozen provenance, support, fold or holdout guard fails.

If any gate fails, the decision is `V46_REPRESENTATION_REJECT_OR_INSUFFICIENT_EVIDENCE`.

If every gate passes, the decision is `V46_DEVELOPMENT_CANDIDATE` only. Kraken remains sealed until a separate prospective external-holdout preregistration is created.

## Explicitly forbidden

Under v0.46, do not:

- alter stop, target or 30-bar horizon;
- tune the TIME-positive threshold away from zero;
- prune assets, regimes, event families or features from observed outcomes;
- alter transaction costs or Financial Governor thresholds;
- add Transformer / Decision Transformer / Mamba / PatchTST;
- add PPO / DQN / RL;
- add on-chain / order-book / sentiment features;
- open Kraken;
- enable PAPER or LIVE.

## Required outputs

The canonical v0.46 artifact must contain:

- exact source/provenance record;
- arm/fold/asset support manifest;
- class prevalence tables;
- forecast metrics;
- reliability-bin tables;
- expected-R state-value tables computed from training only;
- OOS predictions;
- post-governor trades;
- fold and aggregate economic summaries;
- final decision JSON;
- cryptographic hashes for canonical outputs.

## Frozen state

`V46_PREREG_FROZEN / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`
