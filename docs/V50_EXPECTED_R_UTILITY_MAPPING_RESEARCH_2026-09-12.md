# v0.50 — Expected-R / Economic Utility Mapping — Research Note

Date: 2026-09-12
Status: RESEARCH COMPLETE / PRE-EMPIRICAL

## Why this stage exists

v0.47 improved probability calibration but failed economic robustness. v0.48 did not support broad marginal distribution drift as the main explanation. v0.49 did not support broad conditional concept drift as the main explanation. The remaining unresolved layer is the mapping from calibrated state probabilities into economic utility (`expected_r_v47`) and then into capital admission.

The current C1 decision path is:

`C1 calibrated state probabilities -> FIT-estimated state net-R means -> expected_r_v47 -> expected_r > 0 admission -> non-overlap -> Financial Governor -> executed trades`

The v0.47 runner confirms that state utility means are estimated only from FIT; expected-R is the probability-weighted sum of those FIT means; model admission is `expected_r_v47 > 0`; then non-overlap and the frozen Financial Governor are applied.

## Research basis

Recent decision-theoretic work reinforces that probability quality is not identical to downstream decision quality:

1. Perez-Lebel et al., AISTATS 2025, “Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration” — studies regret induced by approximate/calibrated posteriors under downstream costs.
   https://proceedings.mlr.press/v258/perez-lebel25a.html
2. Qiao & Zhao, COLT 2025, “Truthfulness of Decision-Theoretic Calibration Measures” — connects calibration to downstream decision regret.
   https://proceedings.mlr.press/v291/qiao25a.html
3. Flores et al., AISTATS 2026, “A Consequentialist Critique of Binary Classification Evaluation” — argues evaluation should reflect downstream decision consequences rather than discrimination alone.
   https://proceedings.mlr.press/v300/flores26a.html
4. Hegazy, Jordan & Dieuleveut, AISTATS 2026, “Scalable Utility-Aware Multiclass Calibration” — evaluates calibration relative to user utility.
   https://proceedings.mlr.press/v300/hegazy26a.html
5. Gibbs & Tibshirani, COLT 2026, “Sample-Efficient Omniprediction for Proper Losses” — studies predictors in terms of downstream losses/decisions.
   https://proceedings.mlr.press/v336/gibbs26a.html

These results support a diagnostic decomposition before trying another model.

## Frozen scientific question

Does v0.47 fail economically because:

A. FIT-estimated state economic means do not transport to CAL/TEST;
B. predicted expected-R fails to rank realized net-R;
C. the `expected_r > 0` admission rule degrades realized expectancy;
D. non-overlap degrades the candidate set;
E. the Financial Governor degrades the post-non-overlap set;
or F. none of these individual layers explains the instability broadly enough?

## What v0.50 must NOT do

v0.50 is failure attribution only. It must not:
- change the state definition;
- refit or enlarge the predictive learner;
- add features;
- change temperature calibration;
- tune an expected-R threshold;
- prune assets, venues, sides, regimes, families, months or folds;
- change costs, non-overlap rules or Financial Governor rules;
- inspect Kraken;
- enable PAPER or LIVE execution.

## Expected implication

If a specific utility/admission layer is consistently harmful under frozen diagnostics, only then can a later separately preregistered stage propose a repair. A v0.50 finding is not itself a trading candidate or a promotion event.