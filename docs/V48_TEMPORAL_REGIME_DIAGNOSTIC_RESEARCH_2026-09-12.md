# v0.48 — Temporal / Regime Non-Stationarity Diagnostic — Research Note

Date: 2026-09-12
Status: RESEARCH COMPLETE / DIAGNOSTIC PREREGISTRATION NEXT

## Scope

This stage follows the canonical v0.47 decision `V47_CALIBRATION_REJECT_OR_INSUFFICIENT_EVIDENCE`.

v0.48 is **diagnostic only**. It is not a candidate-generation, model-selection, threshold-tuning, asset-pruning, venue-pruning, event-family-pruning, regime-pruning, or external-validation stage. It cannot authorize Kraken, PAPER, or LIVE execution.

Source evidence is already-consumed development evidence:

- canonical v0.44 prepared-data run: `34700944062`
- canonical v0.47 run: `34705324352`
- canonical v0.47 scientific head: `802b0cde38549617589b6a4a313949361dd9425c`
- canonical v0.47 artifact ID: `10300833544`
- canonical v0.47 artifact digest: `sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118`

The post-result documentation commit `6d20e8c4ba3cfe2523e38747bd82756213fc07b1` only records the canonical v0.47 result and is the branch base for this research note.

## Why v0.48 exists

v0.47 established a specific pattern:

- temperature scaling improved common-space Brier from approximately `0.61651` to `0.60222`;
- reliability error improved from approximately `0.02205` to `0.01598`;
- macro OVR AUC was preserved (`0.59889` to `0.59989`);
- aggregate expectancy remained slightly positive;
- but only `2/5` OOS folds were positive;
- PF remained below `1.05`;
- 36-bps stress PF remained below `1.00`;
- worst drawdown remained beyond the frozen `5%` firewall.

Therefore probability calibration was a real bottleneck but not the only bottleneck. The next scientific question is whether the remaining failure is associated with temporal distribution shift / regime non-stationarity rather than insufficient static model capacity.

## Current literature basis

Recent time-series literature explicitly distinguishes temporal distribution shift from concept drift and emphasizes that static forecast quality can degrade when deployment dynamics differ from training dynamics, even when the base predictor remains unchanged.

Relevant references include:

1. Huang, Ma & Michailidis (UAI 2026), *Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift*. The paper treats covariate shift, concept shift, and temporal dependence as separate reliability problems and uses pre-forecast target windows to quantify mismatch.
2. Zhao, Liu & Prakash (ICLR 2026), *Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift*. The work distinguishes temporal shift from concept drift and argues temporal shift should be diagnosed before more aggressive concept-drift adaptation.
3. *Non-stationarity in financial time series: A taxonomy-based survey of drift detection, adaptation, and evaluation* (Neurocomputing, 2026, DOI `10.1016/j.neucom.2026.134647`). The survey classifies covariate shift, conditional/concept drift, label shift, structural breaks and regime changes, and stresses temporal evaluation protocols in finance.

These references support a diagnostic-first step. They do **not** justify adding a Transformer, RL policy, larger feature set, or adaptive live learner at this point.

## Exploratory observations from the frozen v0.47 artifact

The observations below are descriptive and were used only to formulate the v0.48 diagnostic. They are not confirmatory v0.48 results.

### C1 fold economics

For `C1_TEMPERATURE_R1`:

| Fold | Executed trades | Expectancy R | PF | Stress PF |
|---|---:|---:|---:|---:|
| 1 | 253 | -0.05766 | 0.90734 | 0.86158 |
| 2 | 215 | -0.00810 | 0.98653 | 0.93387 |
| 3 | 233 | +0.01971 | 1.03202 | 0.98289 |
| 4 | 446 | +0.18409 | 1.34537 | 1.27266 |
| 5 | 350 | -0.13905 | 0.76396 | 0.70999 |

This is strong temporal heterogeneity: fold 4 is broadly favorable while fold 5 reverses materially.

### Venue pattern

The failure is not explained by one permanently bad venue:

- fold 1: KuCoin negative, OKX positive;
- fold 2: KuCoin negative, OKX positive;
- fold 3: CoinEx, KuCoin and OKX approximately flat-to-positive;
- fold 4: all three development venues positive;
- fold 5: CoinEx and OKX weakly negative and KuCoin strongly negative.

Thus a static venue deletion would be a post-hoc and scientifically invalid response.

### Regime-label pattern

The coarse regime mixture between folds 4 and 5 is nearly unchanged (range / transition / trend proportions are very similar), but economics reverse sharply. Therefore the frozen coarse regime label alone is not sufficient to explain the shift.

### Continuous-feature pattern

A descriptive robust-median comparison of fold 4 versus fold 5 indicates non-trivial changes in continuous causal features, including approximately:

- `ema200_gap_atr_v39`: robust standardized median shift about `+0.50`;
- `atr_pct`: about `-0.47`;
- `directional_prior_v39`: about `+0.31`;
- `stop_fraction`: about `-0.28`;
- `ema50_gap_atr_v39`: about `+0.22`.

At the same time, Jensen-Shannon changes in the coarse `regime` and `venue` proportions are very small. This is consistent with the hypothesis that market state can drift **within** a nominal regime category.

These numbers are exploratory only. v0.48 will recompute all formal diagnostics from exact canonical artifacts inside the pinned GitHub Actions environment rather than relying on local pickle conversion.

## Engineering/reproducibility finding

The v0.44 event pickle was created in the canonical GitHub environment using Python 3.11 / pandas 3.0.5. A separate local inspection environment using Python 3.13 / pandas 2.2.3 cannot deserialize the timezone-aware millisecond datetime extension representation losslessly.

This is an environment-compatibility issue, not corruption of the canonical artifact. v0.48 will therefore load the exact artifact inside GitHub Actions using the same project dependency resolution and Python 3.11 lineage. No local conversion, coercion, or rewritten event table will be used as scientific evidence.

## Research conclusion

The next justified action is a frozen, non-promotional diagnostic that measures:

1. continuous causal-feature shift from CAL to TEST;
2. state/outcome mixture shift;
3. categorical context shift (regime, event family, side, venue);
4. association between measured shift and forecast deterioration;
5. temporal concentration of economics by frozen fold/month/venue/regime/event family.

The purpose is failure attribution, not rescue tuning.

Explicitly forbidden in v0.48:

- Transformer / PatchTST / CMamba / Decision Transformer;
- PPO / DQN / any RL policy;
- on-chain, sentiment, news or order-book feature expansion;
- changing the mother strategy;
- changing v0.46/v0.47 labels or calibrators;
- changing costs or Financial Governor rules;
- choosing assets, venues, regimes or event families from observed performance;
- threshold relaxation;
- Kraken access;
- PAPER or LIVE execution.

Next artifact: `docs/V48_TEMPORAL_REGIME_DIAGNOSTIC_PREREGISTRATION.md`.
