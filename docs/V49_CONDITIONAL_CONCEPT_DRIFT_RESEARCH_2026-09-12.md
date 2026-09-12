# v0.49 Research Note — Conditional Concept Drift / Time-Varying Feature→Outcome Relationships

Date: 2026-09-12
Status: RESEARCH COMPLETE / PREREGISTRATION NEXT

## Why v0.49 exists

Canonical v0.48 rejected a simple broad marginal-drift explanation. Continuous V41 feature distributions, R1 state, common outcome, coarse regime and event-family marginals did not show a preregistered broad excess-shift signal. Side and venue mixture moved in some folds, but those shifts did not establish the required harmful linkage to forecast quality.

Therefore the unresolved mechanism is narrower: the marginal distribution of X may remain broadly similar while the conditional mapping P(Y|X), probability calibration function, or residual process changes over time.

This is concept/conditional drift rather than ordinary covariate drift.

## External research reviewed

1. Zhao, Liu & Prakash, ICLR 2026, *Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift*. The paper explicitly separates temporal shift from concept drift and argues that changing input→output relationships require treatment distinct from ordinary temporal-distribution changes.
   - https://proceedings.iclr.cc/paper_files/paper/2026/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html

2. Gao et al., AISTATS 2025, *Causal Discovery-Driven Change Point Detection in Time Series*. It motivates change-point detection on conditional distributions rather than only on joint/marginal distributions, using conditional divergence after accounting for other variables.
   - https://proceedings.mlr.press/v258/gao25g.html

3. Franck et al., 2025, *Monitoring the calibration of probability forecasts with an application to concept drift detection*. It proposes monitoring probability calibration over time using cumulative-sum style monitoring, supporting residual/calibration-process diagnostics without changing the underlying predictor.
   - https://arxiv.org/abs/2510.25573

4. Huang, Ma & Michailidis, UAI 2026, *Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift*. It emphasizes that calibration reliability can deteriorate under temporal dependence and concept shift even with a fixed forecasting backbone.
   - https://proceedings.mlr.press/v337/huang26b.html

5. Chen et al., IJCAI 2025, *Learning to Extrapolate and Adjust: Two-Stage Meta-Learning for Concept Drift in Online Time Series Forecasting*. It distinguishes longer-lived macro drift from abrupt micro drift, reinforcing the need to diagnose temporal relationship changes before choosing an adaptive learner.
   - https://www.ijcai.org/proceedings/2025/542

## Research conclusion

The next experiment must remain diagnostic-only. It must not introduce a larger model, adaptive trading policy, asset/venue/side filtering, new threshold, or external Kraken evidence.

The correct next question is:

> Does the already-frozen v0.47 C1 predictor exhibit conditional calibration drift, feature→residual relationship drift, or a time-local change in forecast-loss process that is materially larger from CAL_LATE→TEST than the system's own CAL_EARLY→CAL_LATE baseline?

## Proposed diagnostic families

### A. Conditional calibration-map drift

For each canonical fold×symbol and each common state TARGET / STOP / TIME:

- reproduce the frozen v0.46 R1 multinomial learner;
- reproduce the frozen v0.47 C1 temperature calibrator using the full calibration window only;
- obtain calibrated probabilities on CAL_EARLY, CAL_LATE and TEST;
- fit a fixed regularized binary calibration map per state:

`logit(P(Y_state=1)) = intercept + slope * logit(p_state)`

- define calibration-map distance as Euclidean distance in `(intercept, slope)`;
- internal distance = CAL_EARLY→CAL_LATE;
- forward distance = CAL_LATE→TEST;
- excess conditional calibration drift = forward - internal;
- aggregate the three state-level excess distances by the median.

This diagnoses P(Y|model probability) instability rather than P(X) drift.

### B. Feature→residual interaction drift

For each V41 continuous/semantic feature and each common state, compute the Spearman relationship between the feature and signed forecast residual:

`residual_state = I(Y=state) - p_state`

For each feature×state pair:

- internal relationship change = |rho(CAL_EARLY) - rho(CAL_LATE)|;
- forward relationship change = |rho(CAL_LATE) - rho(TEST)|;
- excess = forward - internal.

The fold×symbol interaction statistic is the median excess across finite feature×state pairs.

This is a direct diagnostic of time-varying feature→outcome relationships after conditioning on the model forecast.

### C. Forecast-loss process change

Use event-level common-space multiclass Brier loss. For an ordered evaluation partition B relative to reference A:

- standardize B losses using mean/std estimated only from A;
- compute `max(abs(cumsum(z_t))) / sqrt(n_B)`.

Internal statistic = CAL_LATE monitored against CAL_EARLY.
Forward statistic = TEST monitored against CAL_LATE.
Excess CUSUM = forward - internal.

This detects persistent or abrupt changes in the forecast-error process without altering predictions.

## Statistical aggregation

The frozen universe remains the exact 120 C1 fold×symbol units from canonical v0.47.

For each of the three diagnostic families:

- calculate unit-level excess statistics;
- within each fold, perform a one-sided exact sign test for excess > 0;
- Benjamini-Hochberg adjust the five fold p-values within that family;
- use `q <= 0.10` and median excess > 0 for a positive fold;
- require at least 3/5 positive folds for family-level support.

This mirrors the v0.48 philosophy and avoids selecting a favorable asset or fold after inspection.

## Linkage diagnostics

To establish practical relevance rather than drift-for-drift's-sake:

- link unit conditional-calibration excess to canonical C1 common Brier and macro OVR AUC;
- link unit conditional-calibration excess to unit-level post-governor expectancy where >=5 executed trades exist;
- use fold-cluster bootstrap Spearman 90% intervals;
- harmful directions are: higher drift → higher Brier, lower AUC, lower expectancy.

These are diagnostic associations only and cannot create a trading filter.

## Decision interpretation before empirical execution

A scientifically meaningful conditional-instability result requires broad fold support, not one dramatic period.

Potential routes to preregister:

- CONDITIONAL_INSTABILITY_SUPPORTED: calibration-map drift in >=3/5 folds, at least one secondary family in >=3/5, plus at least one harmful linkage CI excluding zero.
- CONDITIONAL_SHIFT_PRESENT_LINK_INCONCLUSIVE: primary + secondary support but linkage remains inconclusive.
- RESIDUAL_PROCESS_SHIFT_ONLY: primary calibration-map support absent, but both interaction and loss-process families are broadly positive.
- CONDITIONAL_INSTABILITY_NOT_SUPPORTED: no family reaches broad support.
- INCONCLUSIVE: all other combinations.

## What v0.49 must not do

- no candidate promotion;
- no Kraken read;
- PAPER=false and LIVE=false;
- no asset, venue, side, regime, event-family, month or fold pruning;
- no change to mother strategy, features, R1 label, C1 calibrator, costs or Financial Governor;
- no Transformer / Decision Transformer / PatchTST / CMamba;
- no PPO / DQN / RL;
- no on-chain, news, sentiment or order-book features;
- no post-result bin, threshold, change-point or significance tuning.

## Research state

`V49_RESEARCH_COMPLETE / CONDITIONAL_DRIFT_DIAGNOSTIC_JUSTIFIED / NO_EMPIRICAL_RESULT_YET / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`
