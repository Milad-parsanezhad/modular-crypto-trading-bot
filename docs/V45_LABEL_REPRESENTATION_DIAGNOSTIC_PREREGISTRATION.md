# v0.45 Label / Representation Diagnostic — Preregistration

Date frozen: 2026-09-12  
Status: **DIAGNOSTIC-ONLY / NON-PROMOTIONAL / ROUTE C**

## Activation basis

This experiment is activated by the prospectively frozen `docs/V45_POST_V44_DECISION_POLICY.md` after the canonical v0.44 finalizer returned:

`V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE`

Canonical source evidence:

- v0.44 workflow run: `34700944062`
- scientific head: `5efb843d385d194c549a69dc4b2fd38d56ab906c`
- canonical final artifact: `10300676250`
- canonical artifact SHA-256: `f3834a6dda2758648f5e5ad3dc176ad89699d559f0d8076482ef8cf673002ff2`
- prepared-data artifact: `10299564398`
- prepared-data artifact SHA-256: `a229a36f98cc1c84de1ae303f3e747be44138780937e5a327e3e6876eb737384`

Route selection is recorded exactly once in `docs/V45_ROUTE_SELECTION_2026-09-12.json`.

## Scientific question

Why do the frozen v0.43/v0.44 cause-specific TARGET and STOP probability models remain worse than a training-only empirical hazard baseline out of sample?

This stage does **not** ask whether a new trading strategy is profitable. It diagnoses whether the current failure is dominated by:

1. weak target/stop separability or excessive censoring;
2. probability calibration error versus insufficient resolution;
3. weak information content in the frozen causal feature representation;
4. adverse interaction between the fixed 30-bar horizon and the current TARGET/STOP/TIME label process.

## Data firewall and consumed-evidence status

v0.45 diagnostic reuses only already-consumed v0.44 development evidence.

Allowed inputs:

- exact `v44-prepared-data` artifact from run `34700944062`;
- exact canonical `v44-information-driven-sampling-characterization` artifact from the same run.

No exchange API is called by v0.45. Kraken must not be instantiated, queried or downloaded. No new market period is consumed.

Because v0.44 outcomes have already been observed, all v0.45 outputs are **diagnostic/hypothesis-generating only**. They cannot promote a model, threshold, asset, event family, regime, feature subset or execution mode.

## Frozen analysis population

### Calibration / forecast diagnostics

Use the canonical v0.44 OOS prediction table exactly as finalized. Analyze each available sampling variant separately and preserve the frozen five-fold identity.

S2 may have no modeled rows; absence remains an explicit diagnostic result and is never repaired by changing its sampling threshold.

### Feature-information diagnostics

Use only S0 (`CLOCK_MOTHER_BASELINE`) rows falling inside the five common frozen v0.44 **test windows**. This isolates representation quality from the S1/S2 sampling filters.

No fit/calibration rows are mixed into the test diagnostic population for the primary feature-information table.

Frozen feature list: exactly `V41_FEATURES` from `research_bot/event_competing_risk_v41.py`. No new feature may be added in v0.45.

## Diagnostic A — label separability and censoring

For S0 common-test events report, without filtering by outcome:

- TARGET / STOP / TIME counts and fractions;
- distributions by fold, venue, side, regime and corrected event family;
- event duration in 4h bars;
- cause-specific at-risk table for ages 1..30;
- TARGET hazard, STOP hazard and terminal TIME frequency by event age;
- TIME-event net-R mean/median and duration distribution.

Minimum reporting rule: no stratum is interpreted when `n < 50`; the row may be emitted but must be marked `LOW_SUPPORT`.

The trading target remains 3R and the maximum horizon remains 30 bars. v0.45 does not relabel observations under alternative barriers or horizons.

## Diagnostic B — Brier reliability / resolution decomposition

For TARGET and STOP separately, for each available v0.44 variant and fold, evaluate both:

- frozen model probability (`p_target_v41` or `p_stop_v41`);
- frozen training-only empirical baseline probability.

Use ten fixed equal-width probability bins:

`[0.0,0.1), [0.1,0.2), ..., [0.9,1.0]`.

For binary outcome `Y` and prediction `P`, report:

- Brier score;
- uncertainty `UNC = y_bar * (1-y_bar)`;
- reliability `REL = sum_k n_k/N * (p_bar_k-y_bar_k)^2`;
- resolution `RES = sum_k n_k/N * (y_bar_k-y_bar)^2`;
- reconstruction residual `BS - (REL - RES + UNC)`;
- calibration-bin table with count, mean prediction and empirical frequency.

The decomposition is descriptive. No recalibration method is fitted in v0.45.

## Diagnostic C — frozen-feature information content

For each of the five S0 test folds and each frozen feature:

- univariate ROC-AUC for TARGET-vs-rest;
- univariate ROC-AUC for STOP-vs-rest;
- signed Spearman correlation with TARGET indicator;
- signed Spearman correlation with STOP indicator;
- absolute AUC edge `abs(AUC-0.5)`.

Repeat the same diagnostics in predefined strata:

- overall;
- each of `trend`, `range`, `transition` when present;
- each corrected v0.43 event family when present.

A stratum is eligible for AUC only when:

- `n >= 200`; and
- both binary classes have at least 20 observations.

Unsupported strata produce explicit null diagnostics rather than being dropped.

No feature is selected, removed, reweighted or promoted from these results. Rankings are descriptive only and cannot be used as same-sample feature selection.

## Diagnostic D — horizon / barrier-process interaction

Without creating alternative labels, report the current fixed-label process by age:

- number at risk at each bar 1..30;
- TARGET and STOP terminal counts/hazards by bar;
- cumulative TARGET and STOP incidence proxies from observed terminal counts;
- TIME/timeout mass at the horizon;
- net-R distribution for TIME outcomes;
- same quantities by regime and event family when support permits.

This stage may identify a future hypothesis that the label process is poorly matched to the market path, but any changed horizon, stop or target requires a **new preregistered experiment ID**.

## Frozen diagnostic outputs

The canonical v0.45 diagnostic artifact must contain at least:

- `diagnostic_summary_v45.json`
- `label_distribution_v45.csv`
- `cause_age_hazard_v45.csv`
- `calibration_decomposition_v45.csv`
- `calibration_bins_v45.csv`
- `feature_information_v45.csv`
- `timeout_diagnostics_v45.csv`
- `source_integrity_v45.json`

## Integrity rules

The runner must fail closed unless all of the following are true:

- v0.44 canonical decision equals `V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE`;
- v0.44 final artifact reports `kraken_touched=false`;
- PAPER and LIVE are false;
- prepared fold index contains exactly five common folds;
- prepared event table hash matches the hash recorded by v0.44 provenance;
- the diagnostic feature list equals frozen `V41_FEATURES` exactly;
- no Kraken string appears as an instantiated development venue or data source.

## Interpretation boundary

v0.45 may conclude only that diagnostics are complete and describe where predictive information appears weak or miscalibrated. It cannot produce a `DEVELOPMENT_WINNER`.

Possible next scientific work must be registered separately after reading v0.45. Examples include a new representation or label experiment, but no such experiment is authorized by this preregistration itself.

Explicitly forbidden as same-stage rescue:

- changing CUSUM/DC thresholds;
- changing the 3R target or 30-bar horizon;
- changing stop policy;
- post-hoc feature selection;
- dropping weak assets/regimes/event families;
- Transformer / Decision Transformer;
- PPO/DQN/RL;
- on-chain/order-book/sentiment expansion;
- cost or gate relaxation;
- Kraken access;
- PAPER or LIVE activation.

Final v0.45 state must remain:

`DIAGNOSTIC_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`.
