# Chapter 4 — Results and Evaluation (Master Draft)

Date: 2026-09-09

> This file is an evidence-grounded master draft for thesis integration. It distinguishes historical backtest evidence, untouched-holdout evidence, engineering evidence and prospective forward evidence. It must not be rewritten to imply profitability that the artifacts do not support.

## 4.1 Evaluation philosophy

The experimental program was organized as a sequence of increasingly strict evidence gates rather than a single best-backtest selection. The evaluation therefore separates four different questions:

1. Can the software produce signals and simulate execution correctly?
2. Does a learned candidate outperform strong baselines out of sample after costs?
3. Does that superiority survive robustness, multiple-testing and untouched-holdout controls?
4. Does a frozen hypothesis replicate prospectively in forward PAPER operation?

This separation prevents engineering success from being misreported as investment alpha.

## 4.2 v0.10 — Purged OOS model tournament

The v0.10 live-universe experiment discovered 15,801 listings and 3,798 unique assets. After history, spread and data-quality gates, 17 assets were eligible for deep evaluation. The synchronized 4-hour panel contained 25,483 rows and 1,619 timestamps over 2025-12-12 to 2026-09-08.

Three learned challengers — Logistic Regression, HistGradientBoosting and Random Forest — were compared with Momentum, Ichimoku and Equal-Weight Market baselines under a 12-bp one-way transaction-cost assumption.

The strongest aggregate learned model was Logistic Regression with +8.97% net return and Sharpe 0.717. The Ichimoku baseline produced +32.04% and Sharpe 1.578, while the Equal-Weight Market baseline produced +24.90% and Sharpe 1.512. Logistic was unstable across the three OOS folds, with negative results in the first two folds and a strongly positive third fold.

The formal decision was `NO_MODEL_PROMOTED` because no clear Sharpe edge over the baselines and insufficient fold stability were observed.

## 4.3 v0.11 — Robustness and inference

v0.11 repeated the competition under stronger controls: three seeds, paired moving-block bootstrap, FDR correction, coverage matching and point-in-time regime diagnostics. The final eligible universe contained 15 assets.

Median multi-seed results showed that simple baselines dominated learned challengers. The Ichimoku baseline recorded +39.69% median net return and Sharpe 1.752; Equal-Weight Market recorded +15.55% and Sharpe 1.072. Logistic Regression recorded -27.28% and Sharpe -1.090; HistGradientBoosting and Random Forest were more negative.

For the strongest learned candidate versus the strongest baseline, 729 matched OOS periods were used. The mean 4-hour net-return difference (Logistic minus Ichimoku) was -0.0009083 and the 95% moving-block bootstrap interval was [-0.0022737, -0.0000166]. The one-sided p-value for a positive Logistic edge was 0.9767 and the FDR-adjusted q-value was 1.000.

The regime analysis was equally important. Ichimoku was strongly positive in HIGH_VOL and TREND_DOWN buckets but negative in RANGE and TREND_UP. Therefore its favorable aggregate performance could not be interpreted as universal or regime-stable.

The formal decision remained `NO_MODEL_PROMOTED`.

## 4.4 v0.12 — External untouched derivatives holdout

v0.12 tested whether derivatives and order-flow information added incremental value on a new, frozen holdout rather than retuning v0.11. The experiment used 12 perpetual assets, 13,128 feature rows and 1,094 synchronized 8-hour timestamps. The final untouched holdout covered 2026-06-01 16:00 UTC through 2026-08-31 16:00 UTC with 274 rebalances.

The feature families were price, Ichimoku and derivatives. The derivatives family included funding, premium-index/basis proxy, futures taker flow and lagged OI/OI-value variables with conservative point-in-time alignment.

No learned feature-family variant produced positive net holdout performance. Logistic price-only was least negative at -5.19% with Sharpe -0.192. Logistic full was -5.85% with Sharpe -0.298 and lower maximum drawdown than price-only. The primary full-versus-price-only mean edge was -0.00003896 per 8-hour period, with 95% bootstrap interval [-0.000640, +0.000447], one-sided p-value 0.5581 and FDR q-value 0.9468.

The decision was therefore `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.

This result does not show that derivatives data are useless in general. It shows that this frozen point-in-time feature implementation did not provide sufficient incremental tradable alpha under the specified holdout and cost model.

## 4.5 v0.13-v0.14 — New prospective evidence and production PAPER engineering

Because the v0.12 holdout was consumed, the project did not tune a post-hoc regime or feature gate on it. v0.13 opened a new prospective microstructure window for BTC/ETH using finer flow and true-basis research directions. v0.14 then established the production PAPER runtime.

The runtime separates market data, signal generation, independent risk control, simulated execution and persistence. PostgreSQL stores account state, positions, observations, fills and equity records. Duplicate-bar protection prevents the same completed bar from creating repeated simulated orders. LIVE execution remains fail-closed.

These stages provide engineering evidence, not profitability evidence.

## 4.6 v0.15-v0.16 — Prospective forward evidence

v0.15 created immutable scheduled evidence snapshots from the production PAPER system and pre-registered minimums of 168 forward hours, 100 observations and 10 simulated fills before formal review. v0.16 aggregates those artifacts, collapses rapid unchanged snapshots and generates reproducible defense outputs.

At the official v0.16 record, the system had 2 production observations, 1 simulated paper fill and only 1 independent v0.16 snapshot after deduplication. The formal state was `INSUFFICIENT_FORWARD_SAMPLE` and LIVE promotion remained `PROHIBITED`.

Sharpe and Sortino are intentionally withheld until both the pre-registered sample gate and a minimum numerical depth of independent inter-snapshot returns are satisfied. The earlier paper return near -0.0261% is descriptive only and is not interpreted as evidence of positive or negative expected performance.

## 4.7 Cross-stage interpretation

The combined results reveal a consistent methodological pattern:

- model complexity did not guarantee stronger OOS performance;
- a favorable aggregate baseline could be driven by specific regimes;
- stronger inference overturned superficially attractive model-selection conclusions;
- a theoretically strong derivatives feature family did not automatically create incremental holdout alpha;
- transaction costs and turnover materially influenced economic outcomes;
- prospective operation must be kept separate from historical optimization.

The project therefore supports an evidence-gated architecture in which complexity is promoted only when it survives increasingly strict validation.

## 4.8 Search-aware statistical interpretation

The next candidate-search cycle should extend the current bootstrap/FDR/CPCV infrastructure with an explicit registry of every tried strategy. This enables stronger control of research degrees of freedom through tools such as White’s Reality Check, Hansen’s SPA test, Deflated/Probabilistic Sharpe analysis and Probability of Backtest Overfitting where the data design is appropriate.

The purpose is not to maximize the number of statistical tests in the thesis. The purpose is to prevent a final selected model from receiving credit for a Sharpe ratio that is partly a consequence of repeated search.

## 4.9 Final result statement for the current thesis state

The implemented system is operational as a modular cryptocurrency trading-research and forward PAPER platform. It has demonstrated reproducible data processing, cost-aware OOS evaluation, robustness inference, untouched-holdout testing, independent risk control, persistent paper execution and prospective evidence capture. However, no learned candidate through v0.12 has established stable incremental alpha, and the prospective v0.15-v0.16 forward sample remains below the pre-registered minimum.

Accordingly, the defensible conclusion is that the project demonstrates a rigorous and operational AI-assisted trading research framework, while profitability, stable risk-adjusted alpha and real-money LIVE readiness remain unproven at the current evidence depth.
