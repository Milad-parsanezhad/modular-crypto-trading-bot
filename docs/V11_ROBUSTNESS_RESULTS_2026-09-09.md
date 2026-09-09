# v0.11 Robustness & Inference Results — 2026-09-09

## Status

**Research result only. No paper, testnet or live execution is authorized.**

GitHub Actions run `34309928748` completed successfully on commit `a6e3afbdcc319071597ddef8a676dc51d02846a6`.

The run passed deterministic/regression tests, live multi-exchange discovery, eligibility auditing, multi-seed OOS model reruns, paired moving-block bootstrap, FDR correction, regime diagnostics, evidence validation and artifact upload.

## Live data scope

- Discovered listings: **15,847**
- Unique discovered assets: **3,812**
- Final eligible assets: **15**
- Eligible symbols: `BNB/USDT`, `BTC/USDT`, `DOGE/USDT`, `DOT/USDT`, `ETH/USDT`, `HYPE/USDT`, `NEAR/USDT`, `SOL/USDT`, `SOPH/USDT`, `SUI/USDT`, `TAO/USDT`, `UNI/USDT`, `XMR/USDT`, `XRP/USDT`, `ZEC/USDT`.

### Eligibility quality-control finding

An earlier v0.11 run allowed `USDG/USDT` into the eligible universe. That violates the research requirement to exclude stablecoin bases from alpha ranking. Before the final run, the explicit stable-base policy and regression tests were hardened to reject `USDG`, `USDS`, `USDP`, `BUSD`, `GUSD`, `FRAX`, `LUSD`, `USDD`, and `EURC` in addition to the existing exclusions. The final evidence artifact contains no such stable base.

This correction is retained as an example of **Evidence Before Opinion**: universe quality errors must be fixed before interpreting model performance.

## Robustness design

- Timeframe: **4h**
- OOS structure: purged expanding folds inherited from v0.10
- Learned challengers: Logistic Regression, HistGradientBoosting, Random Forest
- Baselines: Momentum, Ichimoku cross-sectional score, Equal-Weight Market
- Seeds in the live CI pilot: **11, 42, 101**
- One-way cost assumption: **12 bps**
- Bootstrap: **300 paired moving-block resamples**
- Block length: **12 bars** (48 hours at 4h)
- Planned pairwise tests: **9**
- Multiple-testing correction: **Benjamini-Hochberg FDR**, q threshold 0.10
- Regimes: `HIGH_VOL`, `RANGE`, `TREND_DOWN`, `TREND_UP`

## Multi-seed aggregate results

| Variant | Median net return | Median Sharpe | Median max DD | Positive-seed fraction |
|---|---:|---:|---:|---:|
| **Ichimoku baseline** | **+39.69%** | **1.752** | -34.54% | 100% |
| Equal-weight market | +15.55% | 1.072 | **-27.28%** | 100% |
| Momentum baseline | +6.86% | 0.636 | -40.34% | 100% |
| Logistic Regression | -27.28% | -1.090 | -45.67% | 0% |
| HistGradientBoosting | -53.69% | -3.493 | -64.90% | 0% |
| Random Forest | -56.97% | -3.755 | -63.47% | 0% |

The simple baselines again outperform the learned challengers in this configuration. Model complexity therefore remains unjustified as a promotion criterion.

## Strongest learned model vs strongest baseline

Provisional learned model: **Logistic Regression**  
Strongest baseline: **Ichimoku baseline**

Coverage-matched paired inference used **729 common OOS periods**.

- Observed mean 4h net-return difference, Logistic − Ichimoku: **-0.0009083**
- 95% moving-block bootstrap CI: **[-0.0022737, -0.0000166]**
- Observed total-return difference: **-66.97 percentage points**
- Observed Sharpe difference: **-2.842**
- One-sided p-value for a *positive* Logistic edge: **0.9767**
- FDR-adjusted q-value: **1.000**
- Coverage-match fraction: **100%**

There is no statistical evidence that the learned candidate has a positive edge over the strongest baseline. In this run the mean-return bootstrap interval is entirely below zero.

## Regime dependence

The aggregate Ichimoku result is **not stable across regimes**:

| Regime | OOS periods | Net return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| HIGH_VOL | 208 | **+52.34%** | 4.966 | -24.36% |
| TREND_DOWN | 139 | **+28.84%** | 8.874 | -4.47% |
| RANGE | 280 | **-16.98%** | -2.236 | -28.85% |
| TREND_UP | 102 | **-14.27%** | -4.495 | -15.53% |

This is an important scientific result. The aggregate Ichimoku baseline must **not** be described as a universal profitable strategy. Its performance is concentrated in particular point-in-time regimes in this sample, and any regime gate derived from these observations becomes a **new hypothesis** that requires fresh/external validation.

The provisional learned model (Logistic Regression) was also regime-dependent: positive in `TREND_DOWN`, negative in `RANGE` and `TREND_UP`, and approximately flat/weak in `HIGH_VOL`.

## Promotion decision

**`NO_MODEL_PROMOTED`**

Promotion failed for all of the following explicit reasons:

1. `NON_POSITIVE_MEDIAN_SEED_RETURN`
2. `INSUFFICIENT_MULTI_SEED_STABILITY`
3. `DRAWDOWN_TOO_LARGE`
4. `BOOTSTRAP_EDGE_CI_INCLUDES_ZERO`
5. `NO_FDR_SIGNIFICANT_EDGE_OVER_STRONGEST_BASELINE`
6. `INSUFFICIENT_REGIME_STABILITY`

Therefore no v0.11 model output may be routed to paper execution.

## Scientific interpretation

1. The v0.10 observation that a simple model could be competitive did **not** survive stronger v0.11 inference; the learned challengers were materially worse than the simple baselines.
2. Multi-seed evaluation matters for stochastic tree models; HGB and Random Forest showed seed variation, but all tested seeds remained negative.
3. The cross-sectional Ichimoku score remains a useful **research baseline/challenger**, not a confirmed entry rule.
4. Regime dependence is now a first-class hypothesis. Reusing the same sample to tune a regime gate would be research overfitting.
5. The correct next step is not paper deployment. It is a pre-registered external/untouched validation of regime-conditioned technical hypotheses and the addition of evidence-backed derivative/order-flow features.

## Next research gate

Proposed v0.12 scope:

- freeze the v0.11 result and hypotheses;
- test the Ichimoku/regime interaction on a new validation domain rather than optimizing on this same OOS period;
- add point-in-time funding/basis/OI/order-flow features where historical availability is valid;
- run feature-family ablations (`price-only`, `+Ichimoku`, `+derivatives`, `+Ichimoku+derivatives`);
- retain cost, coverage, bootstrap/FDR and regime diagnostics;
- keep execution disabled unless a candidate clears the Great Filter.

## Evidence contract

These results are valid only for the recorded data universe, time period, cost assumptions and research protocol. They are **not a profitability guarantee, BUY/SELL signal, or authorization for live trading**.
