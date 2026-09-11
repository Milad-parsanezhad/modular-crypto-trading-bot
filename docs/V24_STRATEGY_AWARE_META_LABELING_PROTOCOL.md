# v0.24 Strategy-Aware Meta-Labeling & Incremental Economic Value Protocol

Status: **research only**. Forward PAPER replacement and LIVE execution are explicitly unauthorized.

## Research question

Does a machine-learning abstention layer create **incremental out-of-sample economic value** when placed on top of already-defined trading strategies, rather than attempting to predict market direction from every bar?

The null hypothesis is that the ML layer does not improve the frozen base strategies after fees/slippage and risk normalization.

## Frozen initial strategy set

The first v0.24 experiment is restricted to 4h candidates so that timeframe, friction and execution semantics stay comparable:

- `H4_S6_BREAKOUT`
- `H4_KUMO_TRIANGLE`
- `H4_OB_BOS_RETEST`
- `H4_SUPPLY_DEMAND`
- `H4_CORRELATION_DIVERGENCE`
- `H4_D1_S6_VOL_RISK`
- `H4_D1_OB_BOS_RISK`

No strategy may be added after the untouched test has been inspected for this experiment.

## Event definition and labels

The inherited bracket simulator is treated as a triple-barrier implementation:

- signal is known at the close of bar `t`;
- entry is the open of `t+1`;
- horizontal lower/upper barriers are stop and target;
- the vertical barrier is `max_hold_bars`;
- a stop/target collision inside the same bar is resolved stop-first;
- frozen friction is 10 bps fee + 2 bps slippage **each way**, or 24 bps round trip.

The principal meta-label is `label_meta_execute = 1` only when the frozen strategy event has positive post-cost R-multiple. Triple-barrier exit class and vertical-barrier timeout are retained as labels/audit fields, not model features.

## CUSUM event feature

A causal CUSUM detector is added as a market-event/context feature. Its threshold at time `t` uses a volatility estimate shifted one bar; the close-to-close return at `t` is available at the signal close. Therefore future rows cannot alter historical CUSUM state.

CUSUM is not assumed to be alpha by itself. It is tested only as context for strategy-event quality.

## Point-in-time feature contract

Only `f_*` numeric features plus the declared non-identity context (`timeframe`, `side`) may enter X. Post-trade fields such as entry/exit result, R-multiple, realized return, exit reason, future target and equity are forbidden by the inherited v0.23r leakage guards.

Strategy name and symbol are not direct model features in the primary arm. Instead, frozen setup parameters such as reward/risk, ATR stop multiple and maximum holding period are encoded as numeric context. This reduces simple strategy/symbol memorization.

## Split discipline

All strategy events from all symbols are pooled and split by **unique signal timestamps**:

- 60% development: model fitting only;
- 20% validation: model-family/seed/threshold selection only;
- 20% untouched test: inspected once after freeze.

A two-timestamp embargo is applied and any development/validation event whose `label_end_time` crosses the next segment boundary is purged.

## Model search

The model family set is inherited from the clean v0.23r tabular registry to preserve comparability:

- Logistic Regression
- Ridge Classifier
- SGD Logistic
- Random Forest
- Extra Trees
- Gradient Boosting
- Histogram Gradient Boosting

Each stochastic family is evaluated across the frozen seeds `314, 2718, 1618`. Deep temporal models, localization/vision and multimodal fusion remain separate upstream research tracks and are not allowed to contaminate this tabular selection loop.

## Selection objective

The validation objective is not raw classification accuracy. For every score threshold, v0.24 compares:

`Base strategy events` vs `Base strategy events + ML abstention filter`.

The threshold objective rewards improvement in mean R and profit factor and penalizes incremental drawdown. A threshold must retain at least 200 validation events.

The model family is selected by mean validation objective across seeds; the seed is then selected by validation objective. The test set does not choose the family, seed or threshold.

## Untouched-test evidence

After the champion is frozen, v0.24 evaluates:

- base vs filtered expectancy R;
- base vs filtered profit factor;
- total compounded return under fixed 0.25% research risk per event;
- maximum drawdown;
- symbol-level uplift breadth;
- strategy-level uplift breadth;
- paired moving-block bootstrap CI of incremental event return;
- 24 / 36 / 60 bps round-trip cost stress;
- Deflated-Sharpe-style multiple-testing diagnostic;
- validation-only CSCV/PBO ranking-instability diagnostic.

The validation PBO implementation is deliberately described as a diagnostic rather than full CPCV because the candidate survives to a separate full-CPCV retraining stage only if the untouched-test gate is passed.

## Internal candidate gate

`INTERNAL_META_RESEARCH_CANDIDATE` requires all of the following:

- >=100 selected untouched-test events;
- positive filtered expectancy;
- filtered PF >=1.05;
- filtered total return greater than the unfiltered base;
- filtered maximum drawdown <=5%;
- paired block-bootstrap uplift CI lower bound >0;
- >=60% of symbols improve;
- >=50% of frozen strategies improve;
- Deflated Sharpe probability >=0.95;
- validation PBO <=0.50;
- positive expectancy and PF >=1.0 under 36 bps round-trip stress.

A pass **does not authorize Forward PAPER or LIVE**. It only unlocks the next mandatory stages:

1. full CPCV/retraining analysis;
2. frozen external-venue replication without refit/retuning;
3. strategy/portfolio-level risk and overlap simulation;
4. fresh Forward PAPER observation.

## Failure semantics

If any scientific gate fails, the decision is `NO_META_MODEL_PROMOTED`. The gate is never weakened after the result is observed.
