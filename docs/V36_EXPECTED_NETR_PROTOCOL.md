# v0.36 — Expected Net-R + Conformal Uncertainty + Duration Utility

**Status:** preregistered research only. `PAPER replacement=false`, `LIVE=false`.

## Why this stage exists

v0.24d showed that event-level filtering could improve event economics while portfolio admission destroyed much of that advantage. v0.35 then showed that a win/loss-style asymmetric meta-filter was not robust across consumed CoinEx, OKX and KuCoin. v0.36 therefore changes the learning target rather than lowering a failed threshold: it predicts **post-cost R-multiple directly**, penalizes uncertainty, and penalizes capital lock duration. Portfolio drawdown/admission is deliberately deferred to v0.37.

## Literature basis frozen before result inspection

1. Barak, Mousavi & Hosseini (2025), *Deep Reinforcement Learning for Dynamic Learn to Rank: A Risk-Aware Framework for Cryptocurrency*, SSRN 5494646. The paper motivates treating ranking, meta-filtering and risk allocation as distinct modules in volatile crypto markets.
2. Kato (2024), *Conformal Predictive Portfolio Selection*, arXiv:2410.16333. It motivates incorporating predictive uncertainty into portfolio decisions rather than using point forecasts alone.
3. Schmitt (2026), *Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR*, arXiv:2602.03903 / SSRN 6172999. It motivates sequential uncertainty calibration under nonstationarity and regime drift.
4. Taheri Hosseinkhani (2026), *False Confidence in Machine-Learning Asset Pricing*, SSRN 6962648. It motivates explicitly penalizing forecast confidence when calibration reliability deteriorates.
5. Kim & Lim (2026), *From Predictability to Tradability: Machine-Learning Signals and Trading Frictions in Cryptocurrency Markets*, SSRN 7115197. It motivates layered evaluation from predictive ranking to net tradability instead of equating forecast accuracy with implementable alpha.

These sources motivate the architecture; none of their reported performance is imported into our result.

## Frozen data governance

Development venues are already-consumed evidence:

- CoinEx
- OKX
- KuCoin

**Kraken is reserved untouched evidence and MUST NOT be fetched by v0.36.**

The base event pool is the complete six-strategy Daily subset of the already-defined v0.30 registry, avoiding post-hoc choice of only the best prior strategy:

- `V30_D1_REGIME_MOM_20_90`
- `V30_D1_REGIME_MOM_30_120`
- `V30_D1_CUSUM_BREAKOUT_3`
- `V30_D1_CUSUM_BREAKOUT_4`
- `V30_D1_ICHIMOKU_REGIME_100`
- `V30_D1_ICHIMOKU_REGIME_125`

## Model families

Exactly three fixed candidates are tested:

- Ridge regression, `alpha=10`
- Huber regression, `epsilon=1.35`, `alpha=0.001`
- Histogram Gradient Boosting, fixed depth/learning-rate/iterations/random seed

No hyperparameter search is performed.

## Feature contract

Only information available at signal time may enter the model: side, ATR%, short/medium returns, EMA gaps, market regime, cross-sectional dispersion ratio, causal volume ratio, Ichimoku cloud position, calendar cycle, and fixed base-strategy one-hot indicators. **Symbol identity is excluded.** Outcome, entry/exit result, realized duration and future price fields are forbidden as features.

## Cross-venue + chronological design

For each target development venue, the other two venues are pooled as sources. Source unique signal timestamps are split chronologically:

- first 60%: model fit;
- next 20%: conformal calibration;
- final source 20% is not used for model fitting/calibration;
- target venue is scored only for signals strictly **after** the source calibration cutoff.

Thus target outcomes never train their own target model, and scoring is temporally later than the source fit/calibration window.

## Economic target and uncertainty

Return model target: realized **post-cost R-multiple**.

Duration model target: `log1p(holding days)` using source outcomes only.

One-sided conformal uncertainty is calibrated from source residuals. The score is:

`utility = conformal_lower_R / sqrt(1 + predicted_holding_days)`

The top 50% by predicted utility on each target venue are selected. The 50% fraction is frozen for sample sufficiency and is not outcome-tuned.

## v0.36 event-alpha gate

Every development venue must independently satisfy:

- selected events >= 200;
- Profit Factor >= 1.05;
- expectancy > 0 R;
- positive-symbol breadth >= 60%;
- moving-block CI lower bound on fixed-risk event return > 0;
- PF >= 1.0 under 36 bps round-trip stress.

A portfolio MDD gate is intentionally **not** applied in v0.36 because this stage isolates event alpha. The unchanged <=5% portfolio MDD constraint returns in v0.37.

## Multiplicity

Prior effective trials: 120. New v0.36 trials: 3. Total effective history after v0.36: **123**.

## Fail-closed semantics

Possible decisions:

- `NO_V36_ROBUST_EVENT_UTILITY_CANDIDATE`
- `V36_EVENT_UTILITY_CANDIDATE_REQUIRES_PORTFOLIO_ARBITRATION`

Even the second state does **not** touch Kraken and does not authorize PAPER replacement or LIVE execution. It only permits the already-preregistered v0.37 portfolio arbitration experiment to be interpreted as a joint event+portfolio candidate if its parent v0.36 model also passed.