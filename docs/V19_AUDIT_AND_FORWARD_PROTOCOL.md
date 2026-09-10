# v0.19 — Audit-Corrected Research & Prospective Multi-Venue Microstructure

Date: 2026-09-10

Status: **Pre-registered audit/collection stage — no signal, PAPER replacement, testnet or LIVE authorization**

## 1. Why v0.19 is necessary

The v0.18 artifact was successfully executed and correctly failed its promotion gates, but a deeper code/method audit found several issues that matter for scientific interpretation. v0.19 does not erase v0.18. It preserves the original artifact as the historical record, records the defects explicitly, performs an audit-corrected re-analysis, and starts a genuinely prospective evidence stream for the next hypothesis.

## 2. v0.18 defects found in audit

### BUG-A1 — uncertainty calibration used in-sample residuals

v0.18 fitted Ridge on the training set and then estimated residual MAD on the same training rows. Even for a regularized linear model this is an optimistic uncertainty estimate because the calibration observations also participated in model fitting.

**v0.19 fix:** estimate residual MAD from chronological expanding-window out-of-fold predictions entirely inside the training segment. The final holdout remains excluded from model fitting and uncertainty calibration.

### BUG-A2 — terminal liquidation cost could be omitted

The v0.18 transition-cost backtest charged entry/exit costs when position changed inside the evaluation frame, but if the final row ended long the forced closing transaction at the end of the experiment was not charged.

**v0.19 fix:** audit backtests explicitly force terminal flattening and charge the additional one-way turnover/cost.

### BUG-A3 — reproducibility metadata was incomplete

The v0.18 result did not persist full coverage timestamps and a deterministic dataset fingerprint for Experiment A. The numerical artifact is preserved, but exact later reconstruction is harder than necessary.

**v0.19 fix:** every audited symbol records coverage start/end and a SHA-256 fingerprint over canonical timestamp/OHLCV data.

### BUG-B1 — v0.18-B was not an exact replication of the v0.11 Ichimoku baseline

v0.11 evaluated an **Ichimoku cross-sectional top-quartile portfolio**. v0.18-B instead used a separate per-asset binary rule, `ichimoku_score > 0`. Those are different strategy geometries.

**v0.19 fix:** restore the v0.11 geometry: at every timestamp, rank the synchronized asset universe by the frozen Ichimoku score and hold the top quartile equal-weight, subject to the same one-way cost convention.

### BUG-B2 — the v0.18 regime gate was asset-level while the v0.11 diagnostic was market-level

The v0.11 regime table classified the synchronized market state from cross-sectional regime fractions. v0.18-B gated each asset using that asset's own regime flags. This changed the hypothesis.

**v0.19 fix:** classify a single market regime per synchronized timestamp using the original v0.11 `_classify_regime` logic and gate the **whole cross-sectional portfolio** only when that market regime is `HIGH_VOL` or `TREND_DOWN`.

### LIMITATION-B3 — contemporaneous external venue is not a fully independent economic domain

OKX and CoinEx spot prices for BTC/ETH reflect the same globally arbitraged crypto market. A cross-venue test is useful for provider/implementation transfer, but it should not be described as equivalent to a new prospective temporal replication.

**v0.19 policy:** the corrected retrospective run is labelled `RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`. Fresh promotion evidence must come from a pre-registered future window or a genuinely independent evidence domain.

## 3. Audit re-analysis policy

The audit-corrected v0.18-A/B calculations are intentionally **non-promotional**. The previous result has already been observed, so any corrected re-analysis is treated as software/scientific audit evidence rather than a second chance to pass the same holdout.

Allowed audit labels:

- `SPENT_HOLDOUT_AUDIT_ONLY_NOT_NEW_EVIDENCE`
- `RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`

Forbidden outcomes from the audit path:

- `VALIDATED_OOS`
- `SEARCH_AWARE_SURVIVOR`
- PAPER strategy replacement
- testnet promotion
- real-money LIVE authorization

## 4. New prospective hypothesis: multi-venue microstructure

Recent peer-reviewed evidence strengthens the case for studying richer order flow rather than merely adding deeper architectures. Anastasopoulos et al. (2026) report that international/world order flow can explain and predict the cross-section of cryptocurrency returns, while Easley et al. (2026) report economically relevant own-market and cross-market microstructure effects for major cryptocurrencies. Bysik & Ślepaczuk (2026) also show that the forecast-to-trade conversion rule and transaction costs can dominate headline model performance.

These papers motivate a **new hypothesis**, not a reinterpretation of the negative v0.12 holdout.

### H19

> Point-in-time, multi-venue trade-flow and top-of-book features may contain incremental information that is absent from candle-level taker-flow proxies, but they must first pass prospective data-quality and later untouched predictive tests after realistic costs.

## 5. Frozen prospective collection design

### Symbols

- `BTC/USDT`
- `ETH/USDT`

### Public venues

- CoinEx
- OKX
- KuCoin

The collector is provider-tolerant but fail-closed at the feature level: a symbol requires at least **two accepted venues** in a snapshot.

### Features captured per venue

- best bid / best ask;
- spread in basis points;
- top-depth bid and ask quote notional;
- depth imbalance;
- signed recent trade buy notional;
- signed recent trade sell notional;
- signed-trade coverage;
- trade imbalance.

### Cross-venue features

- median mid price;
- cross-venue mid-price dispersion in bps;
- mean spread;
- mean depth imbalance;
- mean signed trade imbalance;
- trade-imbalance sign agreement.

### Quality gates

- at least 2 accepted venues per symbol;
- no crossed/invalid order books;
- spread <= 50 bps per accepted venue;
- cross-venue mid dispersion <= 100 bps;
- signed-trade coverage >= 80% on at least 2 venues.

If a quality gate fails, `feature_authorized=false`; the snapshot remains as evidence but cannot enter a model.

## 6. Collection clock and immutability

GitHub Actions collects one immutable snapshot every four hours at minute 17. Each run uploads a uniquely named 90-day artifact.

Collection rule:

`prospective_only_no_backfill_no_signal`

The system must not fabricate historical order-book snapshots or retrospectively fill missed microstructure periods from candle data.

## 7. Maturity gates before modelling

Two gates are frozen before observing predictive results:

### Data-quality pilot gate

- >= 7 elapsed days;
- >= 42 scheduled 4h snapshots;
- >= 80% of expected snapshots with at least two-venue coverage for both BTC and ETH;
- no unresolved timestamp/unit/schema defect.

Passing this gate authorizes only feature-quality analysis.

### Predictive-evidence gate

Before any new alpha claim:

- >= 250 independent 4h decision timestamps after quality filtering;
- frozen feature definitions;
- frozen next-4h target construction;
- explicit turnover + fee + slippage assumptions;
- price-only benchmark versus price+microstructure challenger;
- chronological/purged evaluation;
- dependence-aware block bootstrap;
- full trial registry;
- FDR for planned pairwise hypotheses;
- SPA/Reality Check and DSR/PSR if a multi-trial candidate search is conducted;
- no threshold tuning on the final evaluation window.

## 8. Research references anchoring v0.19

1. Anastasopoulos, A., Gradojevic, N., Liu, F., Maynard, A., & Tsiakas, I. (2026). *Order flow and cryptocurrency returns*. Journal of Financial Markets, 79, 101047. DOI: `10.1016/j.finmar.2026.101047`.
2. Easley, D., O'Hara, M., Yang, S., & Zhang, Z. (2026). *Microstructure and market dynamics in crypto markets*. Journal of Financial Markets. DOI: `10.1016/j.finmar.2026.101071`.
3. Bysik, A., & Ślepaczuk, R. (2026). *Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting*. SSRN / arXiv working paper. The paper reports that naive sign execution can fail after costs and that cost-aware filtering materially changes turnover/economic outcomes in selected configurations.
4. Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94–107. DOI: `10.3905/jpm.2014.40.5.094`.
5. White, H. (2000). *A Reality Check for Data Snooping*. Econometrica, 68(5), 1097–1126. DOI: `10.1111/1468-0262.00152`.
6. Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. Journal of Business & Economic Statistics, 23(4), 365–380. DOI: `10.1198/073500105000000063`.

## 9. Claim contract

Through v0.19 the following may be claimed:

- specific v0.18 methodological defects were identified and corrected in an audit layer;
- the historical v0.18 result remains preserved and unpromoted;
- a prospective multi-venue microstructure evidence stream is operational if CI/collection tests pass;
- the system remains a functional research/PAPER platform.

The following remain prohibited:

- declaring the cost-aware rule profitable from the spent v0.18 holdout;
- declaring the v0.11 regime interaction replicated from the corrected retrospective audit;
- claiming multi-venue order-flow alpha before the prospective maturity and predictive gates;
- real-money LIVE execution.
