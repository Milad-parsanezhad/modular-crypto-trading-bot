# v0.49 — Conditional Concept Drift / Time-Varying Relationship Diagnostic — Frozen Preregistration

Date: 2026-09-12
Status: **PREREGISTERED BEFORE ANY v0.49 EMPIRICAL EXECUTION**

This document is authoritative for v0.49. No v0.49 empirical result may be interpreted if the scientific implementation does not descend from this preregistration commit.

## 1. Purpose

v0.48 did not support broad marginal feature/label/context drift as the principal explanation of v0.47 instability. v0.49 tests the narrower hypothesis that the conditional relationship between frozen predictors/probabilities and outcomes changes over time even when marginal distributions are comparatively stable.

This stage is **diagnostic only**. It cannot promote a trading candidate and cannot activate Kraken, PAPER or LIVE execution.

## 2. Frozen sources and lineage

Canonical source evidence:

- v0.44 prepared-data run: `34700944062`
- v0.44 scientific head: `5efb843d385d194c549a69dc4b2fd38d56ab906c`
- v0.47 canonical run: `34705324352`
- v0.47 scientific head used by the canonical workflow: `802b0cde38549617589b6a4a313949361dd9425c`
- v0.47 artifact ID: `10300833544`
- v0.47 artifact digest: `sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118`
- v0.48 canonical diagnostic run: `34706819566`
- v0.48 scientific head: `eb2e02f464db3ca58889a0474c453b9d378b66f4`
- v0.48 artifact ID: `10302425731`
- v0.48 artifact digest: `sha256:a07a65418529e83730c2429d330016c52e06b2251387b9f0917b6a5d2bef618e`

The v0.48 artifact is provenance/rationale evidence only. The v0.49 calculations consume exact v0.44 prepared events/folds and exact v0.47 C1 lineage/economic evidence.

Any hash/provenance mismatch invalidates the run.

## 3. Frozen evaluation universe

Primary arm: canonical `C1_TEMPERATURE_R1` from v0.47.

Evaluation units are exactly the 120 supported C1 `(fold, symbol)` units in canonical v0.47.

No unit may be removed because of its v0.49 result. If reconstruction cannot produce the preregistered diagnostics for at least 108/120 units, the run is invalid as `V49_INVALID_INSUFFICIENT_DIAGNOSTIC_SUPPORT`; no scientific routing result may be interpreted.

## 4. Frozen partitions

Use the unchanged five v0.44 common chronological folds.

For each fold×symbol:

- FIT: unchanged frozen fit partition;
- CAL: unchanged frozen calibration partition;
- TEST: unchanged frozen OOS test partition;
- CAL_EARLY and CAL_LATE: split CAL chronologically into two contiguous halves using stable signal-time ordering; earlier half gets the floor(n/2) observations and later half gets the remainder.

TEST is never used to fit the base learner or temperature calibrator.

## 5. Frozen predictor reconstruction

Reproduce the v0.47 C1 predictor exactly:

- frozen R1 four-state representation: TARGET, STOP, TIME_POSITIVE, TIME_NONPOSITIVE;
- frozen v0.46 asset-specific cross-venue multinomial logistic learner;
- frozen features: `V41_FEATURES` plus v0.43 development-venue dummies;
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)`;
- fit base learner on FIT only;
- fit one scalar v0.47 temperature on the **full CAL partition only**;
- apply that same fitted base learner + temperature to CAL_EARLY, CAL_LATE and TEST;
- collapse four-state probabilities into common TARGET / STOP / TIME probabilities exactly as v0.47.

No model refit, recalibration or threshold selection is allowed using TEST labels.

## 6. Diagnostic family A — conditional calibration-map drift

Common states: TARGET, STOP, TIME.

For each partition and each common state, fit a diagnostic binary calibration map only for measurement:

`logit(P(Y_state=1)) = intercept + slope * logit(p_state)`

Frozen implementation:

- probability clip epsilon = `1e-6`;
- one predictor: clipped probability logit;
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)`;
- unweighted fit;
- no hyperparameter search.

A state-level map is finite only if both binary classes occur in that partition. A fold×symbol calibration-map statistic requires finite maps for all three common states in CAL_EARLY, CAL_LATE and TEST.

For state `s`:

- `D_internal_s = Euclidean(theta_CAL_EARLY_s, theta_CAL_LATE_s)`
- `D_forward_s = Euclidean(theta_CAL_LATE_s, theta_TEST_s)`
- `Excess_s = D_forward_s - D_internal_s`

Unit statistic:

`conditional_calibration_excess = median_s(Excess_s)`

No state may be selected or weighted after seeing results.

## 7. Diagnostic family B — feature→residual interaction drift

For each common state:

`residual_state = I(Y=state) - p_state`

For every frozen `V41_FEATURES` column and state pair, compute Spearman rho(feature, residual) separately in CAL_EARLY, CAL_LATE and TEST.

A pair is finite only when:

- at least 20 rows exist in each compared partition;
- feature variance is nonzero in each partition;
- residual variance is nonzero in each partition;
- all three correlations are finite.

For each finite feature×state pair:

- `R_internal = abs(rho_CAL_EARLY - rho_CAL_LATE)`
- `R_forward = abs(rho_CAL_LATE - rho_TEST)`
- `Excess = R_forward - R_internal`

Unit statistic:

`interaction_excess = median(finite pair excesses)`

At least 30 finite feature×state pairs are required for a unit-level statistic. Otherwise that unit is unsupported for family B; it may not be silently dropped from overall support accounting.

## 8. Diagnostic family C — forecast-loss process change

For each event, common-space multiclass Brier loss is:

`L_t = sum_k (p_tk - I(Y_t=k))^2`, k in TARGET/STOP/TIME.

Frozen standardized CUSUM statistic for evaluation partition B relative to reference partition A:

- `mu_A = mean(L_A)`
- `sigma_A = std(L_A, ddof=1)` with floor `1e-8`
- `z_t = (L_B,t - mu_A) / max(sigma_A, 1e-8)` in chronological signal-time order
- `CUSUM(A→B) = max_t |sum_{i<=t} z_i| / sqrt(n_B)`

Then:

- internal = `CUSUM(CAL_EARLY→CAL_LATE)`
- forward = `CUSUM(CAL_LATE→TEST)`
- unit `loss_cusum_excess = forward - internal`

No alternative change-point algorithm or CUSUM threshold may be substituted after results.

## 9. Fold-level inference

For each family A/B/C independently:

1. collect unit excess values inside each fold;
2. perform a one-sided exact sign test for excess > 0, ignoring exact zeros;
3. compute the fold median excess;
4. Benjamini-Hochberg adjust the five fold p-values **within that family**;
5. frozen FDR threshold `q <= 0.10`;
6. a fold is positive iff median excess > 0 AND BH q <= 0.10.

A family has broad support iff at least `3/5` folds are positive.

## 10. Linkage diagnostics

Primary linkage variable: `conditional_calibration_excess`.

Join only by exact canonical `(fold, symbol)` keys.

### Forecast-quality links

Against canonical C1 v0.47 metrics:

- common multiclass Brier: harmful direction = positive;
- common macro OVR AUC: harmful direction = negative.

### Economic link

From canonical C1 post-governor trades, calculate unit expectancy for fold×symbol units with at least `5` executed trades.

Harmful direction:

- higher conditional calibration excess → lower unit expectancy (negative rho).

For each link:

- Spearman rho;
- fold-cluster bootstrap, resampling the five folds with replacement and retaining all units from each sampled fold;
- `3000` bootstrap replicates;
- fixed RNG seed `314`;
- 90% percentile CI.

A harmful linkage is established iff the 90% CI excludes zero in the preregistered harmful direction.

## 11. Frozen decision routing

Let:

- `A = family A broad support`
- `B = family B broad support`
- `C = family C broad support`
- `L = any preregistered harmful linkage CI excludes zero`

Decision:

1. If diagnostic support <108/120 units:
   - `V49_INVALID_INSUFFICIENT_DIAGNOSTIC_SUPPORT`
2. Else if `A AND (B OR C) AND L`:
   - `V49_CONDITIONAL_INSTABILITY_SUPPORTED`
3. Else if `A AND (B OR C) AND NOT L`:
   - `V49_CONDITIONAL_SHIFT_PRESENT_LINK_INCONCLUSIVE`
4. Else if `(NOT A) AND B AND C`:
   - `V49_RESIDUAL_PROCESS_SHIFT_ONLY`
5. Else if `(NOT A) AND (NOT B) AND (NOT C)`:
   - `V49_CONDITIONAL_INSTABILITY_NOT_SUPPORTED`
6. Else:
   - `V49_CONDITIONAL_INSTABILITY_EVIDENCE_INCONCLUSIVE`

No decision permits candidate promotion or Kraken access.

## 12. Required outputs

Canonical artifact must include at minimum:

- `decision_v49.json`
- `unit_conditional_drift_v49.csv`
- `fold_conditional_tests_v49.csv`
- `linkage_v49.csv`
- `context_decomposition_v49.csv`
- `execution_provenance_v49.json`
- `output_hashes_v49.json`

## 13. Engineering / bug-repair contract

If execution fails before a valid decision:

1. classify the failure as engineering vs scientific;
2. research the failure using documentation/source evidence where needed;
3. change only the minimal defect;
4. add a regression test;
5. record the failed/superseded run and repair rationale;
6. rerun the exact scientific contract.

A repair may not change diagnostic families, thresholds, support minima, folds, features, labels, models, costs, or decision routes after empirical results are observed.

## 14. Hard prohibitions

- no asset/venue/side/regime/event-family/fold/month pruning;
- no trading threshold optimization;
- no model-capacity increase;
- no Transformer, Decision Transformer, PatchTST, CMamba, PPO, DQN or RL;
- no on-chain, sentiment, news or LOB features;
- no Kraken read;
- PAPER=false;
- LIVE=false;
- no v0.49 result may modify v0.44–v0.48 historical evidence.

## 15. Safety state

`candidate_promotion_allowed=false`
`kraken_touched=false`
`paper_execution=false`
`live_execution=false`

Final preregistration state:

`V49_PREREGISTERED / CONDITIONAL_CONCEPT_DRIFT_DIAGNOSTIC_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`
