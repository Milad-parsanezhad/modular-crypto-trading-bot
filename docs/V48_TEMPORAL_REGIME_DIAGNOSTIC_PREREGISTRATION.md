# v0.48 — Temporal / Regime Non-Stationarity Diagnostic — Frozen Preregistration

Date frozen: 2026-09-12
Status: **PREREGISTERED BEFORE ANY v0.48 FORMAL DIAGNOSTIC EXECUTION**

## 1. Scientific purpose

v0.48 is a **failure-attribution diagnostic** on already-consumed development evidence. It cannot promote a trading candidate and cannot unlock Kraken, PAPER, or LIVE execution.

Primary question:

> After the v0.47 calibration improvement, is the remaining fold-to-fold economic instability associated with temporal distribution shift / within-regime non-stationarity in the frozen causal feature and label distributions?

The null interpretation is not “the market is stationary”; it is that the frozen diagnostics do not provide sufficient evidence to attribute the remaining failure to measurable temporal distribution shift.

## 2. Immutable source evidence

Only these canonical sources are allowed:

- v0.44 prepared evidence run: `34700944062`
  - artifact name: `v44-prepared-data`
  - expected prepared event-table hash is read from the canonical `prepared_hashes_v44.json` and verified before use.
- v0.47 canonical run: `34705324352`
  - artifact name: `v47-probability-calibration`
  - artifact ID: `10300833544`
  - artifact digest: `sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118`
  - scientific head: `802b0cde38549617589b6a4a313949361dd9425c`
  - frozen prereg head: `7959089a3816b1b3cf1b4179ee386285be4e0b5c`

The documentation-only v0.47 result commit `6d20e8c4ba3cfe2523e38747bd82756213fc07b1` is the parent of the v0.48 prereg branch.

Any hash/provenance mismatch invalidates the v0.48 run before interpretation.

## 3. Environment rule

The canonical v0.44 pickle must be consumed inside GitHub Actions with Python 3.11 and the repository's pinned/resolved dependencies. No locally converted pickle/CSV may substitute for the canonical event table.

## 4. Frozen analysis population

Development venues remain exactly:

- CoinEx
- OKX
- KuCoin

Kraken remains sealed.

The five common chronological folds remain exactly those in canonical `fold_index_v44.csv`; no fold may be rebuilt after seeing v0.48 results.

Primary diagnostic arm is `C1_TEMPERATURE_R1` because v0.47 showed improved Brier/reliability with non-inferior discrimination. This designation is **for diagnosis only**, not promotion. `C0_IDENTITY_R1` is retained as a fixed sensitivity reference. `C2_DIRICHLET_R1` is not a primary diagnostic arm because v0.47 already showed discrimination degradation.

No asset/fold can be added or removed based on v0.48 outcomes. Supported units must equal the canonical v0.47 supported fold×symbol units.

## 5. Frozen features and contexts

Continuous causal feature set is exactly `V41_FEATURES` inherited from `research_bot/event_competing_risk_v41.py`. No new feature may be added after execution starts.

Categorical contexts are exactly:

- `regime`
- `event_family_v41`
- `side`
- `venue`

Label/state distributions are exactly:

- common outcome: `TARGET`, `STOP`, `TIME`
- R1 state: `TARGET`, `STOP`, `TIME_POSITIVE`, `TIME_NONPOSITIVE`

## 6. Exact temporal slicing

For each frozen fold, use the same leakage-safe slicing as v0.47:

- FIT: `signal_time < cal_start` and `exit_time < cal_start`
- CAL: `cal_start <= signal_time < pretest_cut` and `exit_time < test_start`
- TEST: `test_start <= signal_time <= test_end`

For drift-baseline estimation only, CAL is split chronologically at its median unique signal timestamp into `CAL_EARLY` and `CAL_LATE`. This split is not used for model fitting, calibration, filtering, or trade selection.

## 7. Continuous feature-drift diagnostic

For each supported fold×symbol unit and each frozen continuous feature:

1. compute FIT-only robust scale as `max(IQR(FIT)/1.349, MAD(FIT)*1.4826, eps)`;
2. compute standardized 1-Wasserstein distance between `CAL_EARLY` and `CAL_LATE`;
3. compute standardized 1-Wasserstein distance between `CAL_LATE` and `TEST`;
4. define feature-level excess drift as `forward_distance - internal_calibration_distance`;
5. define the unit-level continuous drift score as the median feature-level excess drift across finite, non-degenerate frozen features.

No feature is selected by observed association with P/L.

Per fold, the primary test is a one-sided exact sign test across supported symbols for `unit_continuous_drift_score > 0`. Five fold p-values are corrected with Benjamini-Hochberg at `q <= 0.10`.

A fold is `CONTINUOUS_DRIFT_POSITIVE` only if:

- median unit score > 0; and
- BH-adjusted q <= 0.10.

## 8. Label/state and categorical context shift

For each fold×symbol unit:

- compute Jensen-Shannon divergence for R1 state distribution between `CAL_EARLY` vs `CAL_LATE` and `CAL_LATE` vs `TEST`;
- compute the same for common outcome distribution;
- compute the same for each frozen categorical context (`regime`, `event_family_v41`, `side`, `venue`);
- excess JS is forward JS minus internal-CAL JS.

Per diagnostic family, use the same one-sided sign-test + BH `q <= 0.10` rule across the five folds.

No category may be deleted or separately promoted because it looks favorable.

## 9. Forecast-link diagnostic

Use only canonical v0.47 `forecast_metrics_v47.csv`.

For primary C1 units, join the frozen unit-level drift scores to:

- common multiclass Brier;
- common mean reliability error;
- common macro OVR AUC.

Report Spearman correlations and a fold-cluster bootstrap 90% confidence interval using fixed seed `314159` and `2000` bootstrap resamples. Resampling unit is the fold; all symbols inside a sampled fold move together.

Directional interpretation is frozen:

- harmful drift direction for Brier/reliability: correlation `> 0`;
- harmful drift direction for AUC: correlation `< 0`.

This association is diagnostic only and cannot be used to create a filter.

## 10. Economic decomposition

Using canonical v0.47 `post_governor_trades_v47.csv.gz`, report C1 economics with no pruning by these frozen decompositions:

- fold;
- calendar month;
- venue;
- regime;
- event family;
- fold×venue;
- fold×regime.

Metrics:

- executed count;
- mean net R;
- PF;
- mean stress net R;
- stress PF.

These tables are descriptive and never serve as a selection rule.

## 11. Diagnostic decision routing

The run must emit exactly one of these decisions:

### `V48_DIAGNOSTIC_SUPPORTS_TEMPORAL_DISTRIBUTION_SHIFT`
Requires all:

1. continuous drift is positive in at least `3/5` folds under the frozen BH rule;
2. at least one frozen label/state/context family is positive in at least `3/5` folds;
3. at least one forecast-link metric has a harmful-direction 90% fold-cluster bootstrap CI excluding zero.

### `V48_SHIFT_PRESENT_LINK_TO_FORECAST_FAILURE_INCONCLUSIVE`
Use when conditions 1 and 2 pass but condition 3 does not.

### `V48_NONSTATIONARITY_EVIDENCE_INCONCLUSIVE`
Use otherwise.

None of the three decisions authorizes a trading candidate, Kraken, PAPER, or LIVE.

## 12. Explicit prohibitions

During v0.48 no one may:

- change the mother strategy;
- change v0.46/v0.47 labels, base learner, calibrator, costs or Financial Governor;
- tune thresholds after results;
- delete assets, venues, regimes, event families, folds or months based on outcomes;
- add Transformer/PatchTST/CMamba/Decision Transformer;
- add PPO/DQN/RL;
- add on-chain, order-book, sentiment or news features;
- read Kraken data;
- enable PAPER or LIVE execution.

## 13. Failure and repair policy

If execution fails before a valid diagnostic decision:

1. classify the failure as engineering/provenance vs scientific evidence;
2. investigate the root cause;
3. patch only the defect;
4. add a regression test;
5. preserve all frozen metrics, tests and decision rules;
6. mark the failed run superseded;
7. rerun the exact preregistered diagnostic;
8. record both the defect and repair in immutable documentation.

Any result-dependent change to the scientific contract invalidates v0.48 and requires a new version.

## 14. Safety state

- `kraken_touched = false`
- `paper_execution = false`
- `live_execution = false`
- `candidate_promotion_allowed = false`
- `post_result_pruning = false`

This document is the authoritative scientific contract for v0.48.
