# v0.12 Derivatives / Order-Flow External-Holdout Protocol

Date pre-registered: 2026-09-09
Status: RESEARCH ONLY — NO PAPER/LIVE AUTHORIZATION

## Motivation

v0.11 rejected all learned models for promotion after multi-seed robustness and statistical inference. The strongest learned candidate did not beat the strongest simple baseline. The next justified question is therefore not whether a more complex model can fit price data, but whether derivatives/microstructure information adds incremental out-of-sample information beyond price and Ichimoku.

This protocol is committed **before** inspecting the v0.12 final holdout result.

## Frozen research question

> Does a point-in-time derivatives feature family (funding, premium/basis proxy, futures taker order-flow and lagged open interest) add incremental risk-adjusted net performance beyond price-only and price+Ichimoku feature sets on an external USD-M perpetual archive, after explicit turnover costs and an untouched final holdout?

## Frozen data scope

- Provider: Binance Vision public USD-M archive only for this external replication.
- Market: USD-M perpetual futures.
- Bar clock: completed 8h bars.
- Development start: 2025-09.
- End: last completed archive month available at run time.
- Frozen cohort: `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,BCHUSDT,DOTUSDT,AVAXUSDT`.
- A symbol is usable only if required archive history is present and point-in-time feature quality gates pass.
- Missing provider data are reported as DATA_UNAVAILABLE; they are never backfilled from future observations.

The fixed cohort is deliberate for this external replication and is not a claim of survivorship-bias-free market coverage. Dynamic historical membership is a later dataset problem and is listed as a limitation.

## Feature-family ablation

Four frozen variants are tested under the same model, folds, timestamps and portfolio construction:

1. `price_only`
   - momentum
   - realized volatility
   - liquidity / quote-volume transforms
   - cross-sectional ranks of price/liquidity features

2. `price_ichimoku`
   - all `price_only` features
   - point-in-time Tenkan/Kijun distance
   - price/Kijun distance
   - Kumo width and price/cloud distance
   - cross-sectional Ichimoku ranks
   - no Chikou future shift

3. `price_derivatives`
   - all `price_only` features
   - funding rate and trailing funding aggregates
   - premium-index / basis proxy and trailing z-scores
   - futures taker-buy quote imbalance and trailing summaries
   - daily open interest and open-interest value, conservatively available from the next UTC day
   - cross-sectional ranks of the derivatives features

4. `full`
   - `price_only + Ichimoku + derivatives`

## Models

Frozen challengers:

- regularized Logistic Regression
- HistGradientBoostingClassifier

No hyperparameter tuning is allowed on the final holdout. Model hyperparameters are fixed in code before the first v0.12 holdout run.

## Point-in-time rules

- Every feature at decision timestamp `t` must be computable using information available no later than `t`.
- Funding and premium archive records are merged backward to completed-bar timestamps.
- Daily OI metrics use the existing conservative `+1 day` availability shift before `merge_asof`.
- Ichimoku spans are calculated from contemporaneous/past rolling windows only; no forward chart displacement is used as a model feature.
- No `bfill` after indicator creation.
- No scaler/imputer fit outside training data.

## Split / final holdout

1. Build synchronized panel.
2. Reserve the final **25% of unique timestamps** as `FINAL_HOLDOUT` before any model fit.
3. The first 75% is `DEVELOPMENT`.
4. Development diagnostics use purged expanding chronological folds only.
5. Final models are fit on all development timestamps and evaluated exactly once on the final holdout.

The final holdout is not used for threshold, model, feature or hyperparameter selection.

## Target and portfolio

- Target: next completed 8h perpetual price return.
- Ranking: cross-sectional probability score.
- Portfolio: long-only top quartile, equal weight.
- Minimum assets per timestamp: 6.
- One-way explicit execution cost: **8 bps** per unit turnover.
- Funding cash-flow is reported separately and is not silently fabricated into PnL when a directly aligned cash-flow series is unavailable.

## Statistical inference

For final-holdout incremental comparisons:

- paired moving-block bootstrap on coverage-matched net returns;
- 95% CI for mean return difference;
- one-sided p-value for positive incremental mean edge;
- Benjamini-Hochberg FDR across planned comparisons;
- final-holdout regime diagnostics (trend/range/high-volatility);
- no result is called stable from one aggregate Sharpe value.

Planned comparisons, per model:

- `price_ichimoku - price_only`
- `price_derivatives - price_only`
- `full - price_only`
- `full - price_ichimoku`
- `full - price_derivatives`

## Promotion gate

v0.12 does **not** authorize execution. It may only report one of:

- `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`
- `PROVISIONAL_DERIVATIVES_EDGE_NEEDS_FORWARD_PAPER_REPLICATION`

A provisional edge requires all of the following on the untouched holdout:

1. positive net return;
2. Sharpe > strongest simpler comparator by at least 0.10;
3. max drawdown no worse than -35%;
4. paired bootstrap lower CI > 0 against `price_only`;
5. FDR q <= 0.10 against `price_only`;
6. positive incremental result in at least two major regime buckets with sufficient observations;
7. no point-in-time/data-quality violation.

Even if all gates pass, execution remains closed until forward paper replication, independent risk approval and reconciliation tests.

## Mandatory evidence artifact

The CI artifact must include:

- commit SHA / protocol version;
- requested and usable symbols;
- archive/provider failures;
- coverage start/end;
- feature availability counts;
- development vs final-holdout boundaries;
- variant/model holdout metrics;
- paired bootstrap and FDR outputs;
- regime-conditional diagnostics;
- explicit cost totals and turnover;
- decision gate and reasons;
- a hard assertion that no `BUY_SIGNAL` or `CONFIRMED_ENTRY` exists.

## Known limitations

- Frozen modern perpetual cohort can contain survivorship selection; v0.12 is an external archive replication, not historical-universe coverage proof.
- Premium index is a basis/crowding proxy, not an independently reconstructed spot-perpetual basis series.
- Kline taker-buy imbalance is coarse order flow, not L2/L3 order-book microstructure.
- OI is daily archive data with conservative next-day availability, therefore lower frequency than 8h bars.
- This phase does not yet include liquidation history, on-chain, whale, fundamentals, tokenomics or news.
- No v0.12 result can become a live trading claim.
