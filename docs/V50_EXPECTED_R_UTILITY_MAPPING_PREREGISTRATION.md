# v0.50 — Expected-R / Economic Utility Mapping Failure Attribution — Preregistration

Date: 2026-09-12
Status: **PREREGISTERED BEFORE ANY v0.50 EMPIRICAL EXECUTION**

## Purpose

Diagnose whether the remaining v0.47 economic instability is attributable to the economic-utility mapping and capital-admission pipeline rather than broad forecast drift.

This stage is diagnostic-only. Candidate promotion is impossible.

## Frozen source evidence

- canonical v0.44 prepared evidence: workflow run `34700944062`
- canonical v0.47 evidence: workflow run `34705324352`
- canonical v0.47 scientific head: `802b0cde38549617589b6a4a313949361dd9425c`
- canonical v0.47 artifact digest: `sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118`
- canonical v0.49 diagnostic run: `34707339422`
- v0.49 conclusion: `V49_CONDITIONAL_INSTABILITY_NOT_SUPPORTED`

The primary prediction source is the exact canonical `C1_TEMPERATURE_R1` output from v0.47. v0.50 must not refit the predictor to create the primary expected-R series.

## Frozen universe and support

Use exactly the canonical v0.47 C1 supported fold×symbol units (`120`).

No asset, venue, side, regime, event family, month, fold or row may be removed because of outcome quality.

For a unit-level statistic requiring finite variation/support, unsupported statistics remain missing and are counted transparently. The overall diagnostic requires at least `108/120` units for the primary unit-level families.

## Family A — FIT state-utility transport

States are exactly:
- TARGET
- STOP
- TIME_POSITIVE
- TIME_NONPOSITIVE

For each fold×symbol and each state, compute realized mean `net_r` separately in FIT, CAL and TEST using the frozen v0.44 windows and v0.46 R1 encoding.

A state is eligible inside a unit only if it has at least `5` observations in each of FIT, CAL and TEST.

For a unit with at least `3/4` eligible states:

`CAL_state_MAE = mean_s |mu_FIT(s) - mu_CAL(s)|`

`TEST_state_MAE = mean_s |mu_FIT(s) - mu_TEST(s)|`

`state_transport_excess = TEST_state_MAE - CAL_state_MAE`

Fold evidence uses a one-sided exact sign test of unit `state_transport_excess > 0`, with Benjamini-Hochberg correction across the five folds at `q <= 0.10`. A fold is positive only if median excess > 0 and adjusted q <= 0.10.

`STATE_UTILITY_TRANSPORT_FAILURE_SUPPORTED = true` only if at least `3/5` folds are positive.

State-level signed biases are diagnostic only and cannot be used to select/prune states.

## Family B — expected-R calibration and ranking

Use exact canonical v0.47 C1 TEST predictions, including `expected_r_v47` and realized `net_r`, before model admission.

For each fold×symbol unit with at least `30` finite rows and nonzero expected-R variation compute:

1. Spearman correlation `rho(expected_r_v47, net_r)`.
2. Equal-frequency quintiles of expected-R; compute realized mean net-R in each quintile.
3. `top_bottom_spread = mean_R(top quintile) - mean_R(bottom quintile)`.
4. OLS diagnostic `net_r = alpha + beta * expected_r_v47` (intercept and slope are descriptive only).
5. `mean_bias = mean(expected_r_v47 - net_r)`.

For Spearman and top-bottom spread, fold-level positive and negative sign-test families are evaluated separately with BH correction across the five folds at q <= 0.10.

A ranking fold is **preserved** only when both median Spearman > 0 and median top-bottom spread > 0 and both positive-direction adjusted q values <= 0.10.

A ranking fold is **harmful** only when either median Spearman < 0 with negative-direction adjusted q <= 0.10, or median top-bottom spread < 0 with negative-direction adjusted q <= 0.10.

`EXPECTED_R_RANKING_FAILURE_SUPPORTED = true` only if at least `3/5` folds are harmful.

Absence of preserved ranking is not by itself proof of harmful ranking.

## Family C — pipeline-stage attribution

For each fold, evaluate the exact C1 TEST path at four frozen stages:

S0. `ALL_TEST`: all C1 supported TEST rows.
S1. `EXPECTED_R_POSITIVE`: exact canonical `selected_model == true` (`expected_r_v47 > 0`).
S2. `NONOVERLAP`: apply the unchanged v0.47/v0.39 non-overlap realization to S1.
S3. `GOVERNOR_EXECUTED`: apply the unchanged Financial Governor to S2.

For every stage record n, realized mean net-R and profit factor. For attribution use realized expectancy deltas:

- `admission_delta = expectancy(S1) - expectancy(S0)`
- `nonoverlap_delta = expectancy(S2) - expectancy(S1)`
- `governor_delta = expectancy(S3) - expectancy(S2)`

A pipeline layer is broadly harmful only if its delta is negative in at least `3/5` folds **and** its five-fold median delta is < 0.

Frozen conditions:
- `EXPECTED_R_ADMISSION_FAILURE_SUPPORTED`
- `NONOVERLAP_FAILURE_SUPPORTED`
- `GOVERNOR_FAILURE_SUPPORTED`

The governor reconstruction must reproduce canonical v0.47 C1 executed trade counts by fold exactly. If it does not, v0.50 is invalid and must stop before interpretation.

## Context decomposition

Report, without pruning or gating:
- expected-R ranking by venue, side, regime and event family;
- state-utility transport by state;
- stage expectancy by fold and venue where support exists.

These are hypothesis-generating only.

## Frozen decision routing

Evaluate five failure conditions:
A = state utility transport failure supported;
B = expected-R ranking failure supported;
C = expected-R admission failure supported;
D = non-overlap failure supported;
E = governor failure supported.

- if exactly one of A–E is true: emit its specific `V50_*_FAILURE_SUPPORTED` decision;
- if more than one is true: `V50_MIXED_UTILITY_PIPELINE_FAILURE_SUPPORTED`;
- if none is true: `V50_UTILITY_MAPPING_EVIDENCE_INCONCLUSIVE`.

No route can promote a candidate or open external validation.

## Engineering / provenance requirements

- exact v0.47 artifact hash verification;
- exact v0.44 prepared-event pickle hash verification;
- exact 120 canonical C1 units;
- inherited v0.47 predictor/calibration/economic-state source files must be unchanged;
- unit tests before empirical execution;
- fail closed on schema/provenance/governor-reproduction mismatch;
- deterministic statistical code.

## Safety firewall

- `candidate_promotion_allowed = false`
- `kraken_touched = false`
- `paper_execution = false`
- `live_execution = false`
- `post_result_threshold_tuning = false`
- `post_result_pruning = false`

## Forbidden after results

No threshold search, state regrouping, bin-count search, feature selection, asset/venue/side/regime/family pruning, cost changes, governor changes, model-capacity expansion or Kraken inspection may be justified from this consumed diagnostic without a new preregistration.