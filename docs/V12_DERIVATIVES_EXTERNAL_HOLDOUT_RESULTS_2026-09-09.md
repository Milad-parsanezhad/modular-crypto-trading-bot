# v0.12 Derivatives / Order-Flow External Holdout — Results

Date: 2026-09-09
GitHub Actions run: `34318656074`
Artifact: `v12-derivatives-external-holdout`
Status: **SUCCESS — NO TRADING AUTHORIZATION**

## Executive result

The pre-registered v0.12 external-holdout experiment completed successfully, but **did not find incremental derivatives evidence sufficient for promotion**.

Decision gate:

`NO_INCREMENTAL_DERIVATIVES_EVIDENCE`

The best full learned variant on the untouched holdout was Logistic Regression with price + Ichimoku + derivatives, but it still lost money net of explicit turnover costs and did not outperform its price-only comparator by the required statistical/risk-adjusted gates.

## Data / evidence scope

- Provider: Binance Vision public USD-M archive.
- Requested frozen cohort: 12 perpetuals.
- Usable cohort: **12 / 12**.
- Provider failures: **0**.
- Raw panel rows: **13,140**.
- Feature panel rows after forward-target removal: **13,128**.
- Synchronized timestamps: **1,094**.
- Coverage: **2025-09-01 08:00 UTC → 2026-08-31 16:00 UTC**.
- Development interval: **2025-09-01 08:00 UTC → 2026-06-01 08:00 UTC**.
- Untouched final holdout: **2026-06-01 16:00 UTC → 2026-08-31 16:00 UTC**.
- Final holdout rebalances: **274**.
- Explicit cost assumption: **8 bps one-way per unit turnover**.
- Seeds: `11, 42, 101`.
- Planned paired tests: **10**; realized tests: **10**.
- Multiple-testing control: Benjamini-Hochberg FDR, alpha 0.10.

Usable symbols:

`ADAUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, DOTUSDT, ETHUSDT, LINKUSDT, LTCUSDT, SOLUSDT, XRPUSDT`

Feature families were fully instantiated over the panel:

- Price family: 20 features.
- Ichimoku family: 8 features.
- Derivatives family: 30 features.

Derivatives included funding, premium-index/basis proxy, futures taker order-flow and lagged OI/OI-value features. The OI archive used the conservative next-UTC-day availability shift before point-in-time merging.

## Untouched holdout results

| Model | Variant | Net return | Sharpe | Sortino | Max DD | Explicit cost | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic | price_only | **-5.19%** | **-0.192** | -0.266 | -28.40% | 0.1752 | 0.5438 |
| Logistic | full | -5.85% | -0.298 | -0.428 | **-24.57%** | 0.1869 | 0.5213 |
| Logistic | price_ichimoku | -8.25% | -0.518 | -0.732 | -27.38% | 0.1480 | 0.5287 |
| Logistic | price_derivatives | -16.58% | -1.395 | -1.992 | -30.87% | 0.1933 | 0.5281 |
| HGB | price_only | -14.27% | -0.969 | -1.489 | -31.40% | 0.2440 | 0.5348 |
| HGB | price_ichimoku | -20.99% | -1.621 | -2.459 | -34.60% | 0.2211 | 0.5232 |
| HGB | price_derivatives | -23.55% | -1.896 | -2.771 | -35.38% | 0.2723 | 0.5344 |
| HGB | full | -25.18% | -2.099 | -3.072 | -37.13% | 0.2413 | 0.5319 |

### Interpretation

1. None of the learned feature-family variants produced positive net holdout performance.
2. The simple Logistic price-only model was the least negative learned variant.
3. Adding derivatives alone materially worsened Logistic performance in this specific frozen holdout.
4. The full Logistic model reduced max drawdown relative to price-only, but did not improve return or Sharpe.
5. HGB was materially weaker than Logistic across all four feature-family variants.
6. AUC values remained close to 0.5, consistent with weak directional classification signal.

These results do **not** prove derivatives data are useless in general. They reject the frozen v0.12 specification as sufficient incremental alpha evidence on this holdout.

## Paired bootstrap / FDR inference

### Primary promotion comparison: Logistic full vs price-only

- Common holdout periods: **274**.
- Observed mean-return difference: **-0.00003896** per 8h period.
- Total-return difference: **-0.67 percentage points**.
- Sharpe difference: **-0.106**.
- 95% moving-block bootstrap CI for mean edge: **[-0.000640, +0.000447]**.
- One-sided p-value: **0.5581**.
- BH-FDR q-value: **0.9468**.
- FDR significant: **No**.

Therefore the full model did not demonstrate an incremental edge over price-only.

### Derivatives-only incremental test: Logistic price_derivatives vs price-only

- Mean-return difference: **-0.0004844**.
- Total-return difference: **-11.40 percentage points**.
- Sharpe difference: **-1.203**.
- 95% CI: **[-0.001056, +0.000118]**.
- One-sided p-value: **0.9336**.
- FDR q-value: **0.9468**.

No positive incremental derivatives evidence was found.

### Important nuance: Logistic full vs Logistic price_derivatives

This comparison was the only planned pair that survived FDR:

- Mean-return difference: **+0.0004455**.
- Total-return difference: **+10.73 percentage points**.
- Sharpe difference: **+1.097**.
- 95% CI: **[+0.000105, +0.000742]**.
- One-sided p-value: **0.00664**.
- BH-FDR q-value: **0.06645**.

This is **not** evidence of a profitable full model. Both variants were negative. It only indicates that, conditional on this comparison, adding the price/Ichimoku information to the derivatives-only specification improved the negative derivatives-only outcome.

## Regime diagnostics — best full model (Logistic)

| Regime | Periods | Net return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| HIGH_VOL | 94 | +0.83% | +0.446 | -17.02% |
| RANGE | 123 | -7.58% | -1.614 | -14.46% |
| TREND_DOWN | 37 | +2.14% | +1.458 | -8.44% |
| TREND_UP | 20 | insufficient periods | — | — |

The full Logistic variant was positive in HIGH_VOL and TREND_DOWN buckets but negative in RANGE. This is a diagnostic observation only; it is not sufficient to introduce a new post-hoc regime gate into the same holdout.

## Promotion decision

`NO_INCREMENTAL_DERIVATIVES_EVIDENCE`

Promotion failed for these frozen reasons:

1. `NON_POSITIVE_FULL_NET_RETURN`
2. `NO_CLEAR_SHARPE_EDGE_OVER_PRICE_ONLY`
3. `BOOTSTRAP_EDGE_CI_INCLUDES_ZERO`
4. `NO_FDR_SIGNIFICANT_EDGE_OVER_PRICE_ONLY`

The drawdown gate did not fail for the best Logistic full variant, but passing one risk constraint does not override the other failed evidence gates.

## Seed note

The three requested seeds produced identical results for the frozen Logistic/HGB implementations in this run. That is expected for effectively deterministic training paths under these estimators/settings and should **not** be presented as independent stochastic robustness evidence. Multi-seed analysis remains essential for models whose optimization/initialization is stochastic (for example Random Forest variants with stochastic sampling, neural networks and RL).

## Scientific conclusion for the thesis

v0.12 provides a useful negative result:

> Under the pre-registered 8h external Binance Vision perpetual archive, frozen cohort, fixed Logistic/HGB challengers, explicit turnover cost, point-in-time derivatives alignment and untouched final holdout, the tested funding/premium/order-flow/OI feature family did not add sufficient incremental net alpha over price-only features.

This result supports the thesis principle that adding a theoretically strong data family does not guarantee tradable alpha. It also prevents the project from promoting complexity simply because derivatives/microstructure have strong literature support.

## Next justified research step

Do **not** tune v0.12 on the final holdout.

The next defensible phase is to preserve this holdout as spent and move to a new evidence source/window. Candidate directions are:

1. finer microstructure / independent order-flow source rather than kline taker-buy proxy;
2. true reconstructed spot-perpetual basis rather than premium-index proxy;
3. liquidations / higher-frequency OI deltas where point-in-time history is available;
4. a new forward paper observation window for the frozen hypotheses;
5. only after new evidence, reconsider regime-conditional gating.

Paper, testnet and live execution remain closed.
