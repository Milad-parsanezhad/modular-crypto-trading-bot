# v0.47 Research Note — Probability Calibration After v0.46

Date: 2026-09-12
Status: **RESEARCH COMPLETE / NO EMPIRICAL EXECUTION YET**

## Trigger from v0.46

Authoritative v0.46 result:

- run: `34704267477`
- head: `e3fa7d73e8ef2b7e3bcd5a0f61b61147b055fbd1`
- decision: `V46_REPRESENTATION_REJECT_OR_INSUFFICIENT_EVIDENCE`
- R1 aggregate expectancy: `+0.01341R`
- R1 PF: `1.0234`
- R1 stress PF: `0.9659`
- R1 positive-fold fraction: `0.40`
- R1 worst drawdown: `-5.37%`
- R1 common Brier: `0.61654` versus R0 `0.61215`
- R1 mean reliability error: `0.02211` versus R0 `0.02161`

The timeout-sign representation improved aggregate point economics but did not improve probability calibration or temporal robustness. This isolates calibration as a scientifically plausible next bottleneck.

## Literature reviewed

### 1. Berta et al. (AISTATS 2026) — Structured Matrix Scaling for Multi-Class Calibration

Source: https://proceedings.mlr.press/v300/berta26a.html

Key implication: multiclass post-hoc calibration can improve probability quality, but increasingly expressive calibrators add parameters and can overfit when the calibration sample is limited. The paper motivates starting with low-complexity calibration and escalating only when justified.

### 2. Alberge et al. (2026) — On the calibration of survival models with competing risks

Source: https://arxiv.org/abs/2602.00194

Key implication: competing-risk probability calibration is a distinct problem; calibration should be assessed with proper probability criteria and corrected without conflating it with discrimination. Recalibration can improve probability behavior while preserving ranking/discrimination.

### 3. Kull et al. — Dirichlet calibration

Source: https://arxiv.org/abs/1910.12656

Key implication: Dirichlet calibration is a native multiclass recalibration method and can improve Brier score and log loss across classifier families. It is more expressive than temperature scaling, therefore it should be treated as the second, not first, complexity level in this project.

### 4. Austin & Putter (2026) — competing-risk discrimination and calibration

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12947980/

Key implication: discrimination and calibration should be evaluated separately; time-dependent Brier and explicit calibration metrics are both necessary. Good discrimination does not imply good calibration.

### 5. Calibration survey

Source: https://arxiv.org/abs/2112.10327

Key implication: calibration is necessary when probabilities feed downstream cost-sensitive decisions. Proper scoring rules and independent calibration data are required to avoid optimistic estimates.

## Research conclusion

The next experiment should **not** change Mother Strategy, labels, features, learner family, costs, Financial Governor, asset universe, folds or thresholds. It should isolate a single question:

> Can post-hoc probability recalibration of the frozen v0.46 R1 four-state model reduce OOS common-space Brier/reliability error and improve downstream economic robustness without using test labels?

## Bounded method set selected before execution

Three fixed arms will be evaluated in this order:

1. `C0_IDENTITY_R1` — uncalibrated R1 probabilities; control.
2. `C1_TEMPERATURE_R1` — one scalar temperature per asset-fold, fit only on the existing calibration window by minimizing multiclass negative log likelihood.
3. `C2_DIRICHLET_R1` — L2-regularized multiclass logistic calibration on log-clipped R1 probabilities, fit only on the existing calibration window with fixed `C=1.0`.

No other calibrator may be added after seeing v0.47 results.

## Why this ordering

Temperature scaling has one fitted scalar and is the lowest-variance intervention. Dirichlet calibration is more expressive and is included only as a second fixed complexity step. This follows the bias/variance concern highlighted by modern multiclass-calibration literature.

## Leakage firewall

For each frozen fold and asset:

- base R1 learner is trained on FIT only;
- calibrator is fit on CALIBRATION only;
- OOS metrics and economics are computed on TEST only;
- TEST labels are never used to fit or choose the calibrator;
- no cross-fold pooling of test outcomes is allowed for fitting.

## Planned primary forecast space

Primary comparison remains the common frozen three-state space:

`TARGET / STOP / TIME`

For all calibrated four-state probabilities:

`P(TIME) = P(TIME_POSITIVE) + P(TIME_NONPOSITIVE)`

Native four-state scores remain diagnostic only.

## Scientific stopping rule

The first/simplest calibrated arm in the frozen order `C1 -> C2` that passes **all** frozen forecast and economic gates may become a development candidate. If neither passes, v0.47 is rejected. No result-dependent blending or manual arm selection is allowed.

## Governance

- Kraken: SEALED
- PAPER: false
- LIVE: false
- post-result threshold tuning: forbidden
- post-result asset pruning: forbidden
- post-result regime/event-family pruning: forbidden
- Transformer / Decision Transformer / Mamba / PPO / DQN escalation: forbidden in v0.47
- alternative-data additions: forbidden

This research note precedes v0.47 preregistration and empirical implementation.
