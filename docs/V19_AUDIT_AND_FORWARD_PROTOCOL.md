# v0.19 — Audit-Corrected Research & Prospective Multi-Venue Microstructure

Date: 2026-09-10

Status: **Pre-registered audit/collection stage — no signal, PAPER replacement, testnet or LIVE authorization**

## 1. Why v0.19 is necessary

The v0.18 artifact was successfully executed and correctly failed its promotion gates, but a deeper code/method audit identified defects that matter for scientific interpretation. v0.19 does not erase v0.18. It preserves the original artifact, documents the defects, performs a non-promotional reconstruction audit, and opens a genuinely prospective microstructure evidence stream under a stricter point-in-time data contract.

The guiding rule is:

> Correct bugs on spent evidence, but never recycle the corrected retrospective sample as fresh alpha evidence.

## 2. Audit corrections to v0.18

### BUG-A1 — uncertainty calibration

v0.18 estimated Ridge residual MAD on rows used to fit the model. v0.19 uses chronological expanding-window out-of-fold residuals entirely inside the training segment. Final evaluation rows do not enter model fitting or uncertainty calibration.

### BUG-A2 — terminal liquidation cost

v0.18 could finish with an open long position without charging the final flattening transaction. v0.19 charges terminal turnover/cost explicitly.

### BUG-A3 — historical reproducibility

v0.18 did not archive the raw Experiment-A OHLCV frame or its deterministic fingerprint. The original numerical artifact remains valid as a historical record, but a later rolling API request cannot recreate the exact input bytes.

v0.19 therefore records coverage timestamps and dataset SHA-256 and labels the later reconstruction:

`REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`

### BUG-B1/B2 — hypothesis fidelity

v0.11 used a synchronized **cross-sectional top-quartile Ichimoku portfolio** and a **market-level regime diagnostic**. v0.18-B changed these into a per-asset `score > 0` rule and asset-level regime gates. v0.19 restores the original portfolio geometry and market-level regime construction.

The repaired historical test remains:

`RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`

## 3. Audit path cannot promote

Allowed labels:

- `REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`
- `RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`

Forbidden transitions from the audit path:

- `VALIDATED_OOS`
- `SEARCH_AWARE_SURVIVOR`
- PAPER strategy replacement
- testnet promotion
- real-money LIVE authorization

## 4. Prospective hypothesis H19

Recent peer-reviewed work supports deeper study of order flow and market microstructure, but it does not validate this project's implementation. The frozen hypothesis is:

> **H19:** Point-in-time multi-venue liquidity and reported-trade-flow measurements may provide incremental information beyond candle-level features, but only if the measurements first satisfy prospective synchronization/quality gates and then improve a frozen price-only benchmark on a later untouched 4h decision-time evaluation after realistic costs.

This is deliberately narrower than claiming replication of “world order flow” or high-frequency L2 research.

## 5. Frozen market scope

### Assets

- `BTC/USDT`
- `ETH/USDT`

### Public venues

- CoinEx
- OKX
- KuCoin

A symbol requires at least **two quality-accepted venues** in a measurement snapshot. Provider failure is recorded; it is never silently imputed by another venue.

## 6. Measurement versus trading horizon

The thesis's **primary forecast/trading horizon remains 4 hours**.

v0.19 uses a denser **target measurement cadence of 30 minutes** because a single instantaneous REST snapshot every four hours is too sparse to characterize within-bar liquidity and flow. The scheduled collector targets minutes `17` and `47` of each UTC hour. GitHub Actions scheduling can be delayed, so actual observation timestamps, not nominal cron labels, are authoritative.

The denser measurement stream does **not** create additional independent 4h outcomes. Before prediction, sub-4h measurements must be aggregated into one frozen feature vector per 4h decision timestamp.

## 7. Fixed point-in-time trade-flow contract

### Problem being prevented

Public `fetchTrades(limit=N)` without a `since` boundary returns an exchange-specific default range. The same N can therefore correspond to materially different lookback durations across venues.

### Frozen v0.19 rule

For every venue observation at order-book time `t`:

- requested reported-trade window: **60 seconds**;
- interval: **[t − 60s, t]**;
- CCXT venues are queried with `since=t−60s` plus a maximum count;
- every provider response is locally re-filtered to the exact interval;
- any trade with timestamp `> t` is excluded and counted as future-provider overshoot;
- recent count, raw count, first/last timestamp and last-trade staleness are persisted;
- minimum recent trades per accepted venue: **5**;
- maximum last-trade staleness: **30 seconds**.

This makes cross-venue reported-trade imbalance a common-clock measurement rather than a comparison of arbitrary latest-N histories.

## 8. Trade-side semantics

The unified/public `side` value is stored as reported by the provider. It is **not** treated as independently verified aggressor classification.

Frozen semantic label:

`exchange_reported_side_unverified_aggressor`

Accordingly, thesis/report language must prefer:

- **reported trade imbalance**
- **provider-reported buy/sell side**

and must avoid claiming a definitive Lee–Ready-style aggressor reconstruction unless such a method is separately implemented and validated.

## 9. Per-venue measurements

- best bid and best ask;
- mid price;
- spread in bps;
- exact top-depth bid quote notional `sum(price_i × quantity_i)`;
- exact top-depth ask quote notional;
- depth imbalance;
- 60-second reported buy notional;
- 60-second reported sell notional;
- unknown-side notional;
- reported trade imbalance;
- signed-side coverage;
- raw/recent trade counts;
- first/last recent trade timestamp;
- trade staleness;
- excluded future-trade count;
- source/venue identity.

CoinEx base quantities remain available separately to the PAPER execution layer; cross-venue comparison uses quote notionals.

## 10. Cross-venue quality gates

Per accepted venue:

- valid non-crossed order book;
- spread <= **50 bps**;
- fixed trade window exactly **60 seconds**;
- >= **5** recent trades;
- last reported trade <= **30 seconds** old;
- non-negative notionals;
- declared side semantics;
- no future trade in the feature window.

Per symbol snapshot:

- >= **2 accepted venues**;
- cross-venue mid dispersion <= **100 bps**;
- cross-venue order-book clock skew <= **60 seconds**;
- signed-side coverage >= **80%** on at least two accepted venues.

If these gates fail, the snapshot remains immutable evidence but `feature_authorized=false`.

## 11. Cross-venue features

Only after quality gating:

- median mid price;
- mid-price dispersion bps;
- mean spread bps;
- mean depth imbalance;
- mean **reported** trade imbalance;
- reported-trade sign agreement;
- number of accepted venues;
- observation clock skew.

The legacy JSON alias `mean_trade_imbalance` is retained for compatibility, but its explicit semantic interpretation is `mean_reported_trade_imbalance`.

## 12. REST snapshot boundary

v0.19 is a **multi-venue public REST snapshot pilot**. It is not:

- a reconstructed full-depth historical order book;
- an exchange-native incremental WebSocket event stream;
- L3 order-level data;
- the 11-currency “world order flow” dataset used by Anastasopoulos et al.

If the pilot indicates useful and stable measurement quality, a future separately registered stage may add short-burst WebSocket/L2 event capture. That would be a new measurement layer, not a silent replacement of v0.19.

## 13. Prospective maturity gates

### Gate Q — data-quality pilot

Before any feature-predictiveness analysis:

- >= **7 elapsed days** after the corrected fixed-window collector is activated;
- target >= **336 nominal 30-minute measurement opportunities**;
- actual scheduler coverage reported explicitly;
- >= **80%** of expected measurement opportunities with two-venue quality coverage for both BTC and ETH;
- timestamp, units, side semantics and fixed-window validation passing;
- no unresolved schema/unit/look-ahead defect.

Passing Gate Q authorizes only construction of a frozen 4h feature table.

### Gate F — frozen 4h feature construction

After Gate Q, define once and register before target inspection how within-bar measurements are aggregated. Candidate summary statistics may include median, robust mean, dispersion and slope, but the final set must be frozen before predictive testing.

No sub-4h snapshot is treated as an independent 4h return outcome.

### Gate P — predictive evidence

Before an alpha claim:

- >= **250 independent quality-filtered 4h decision timestamps**;
- frozen next-4h target;
- frozen `price_only` benchmark;
- frozen `price_plus_microstructure` challenger;
- explicit fee, spread/slippage and turnover treatment;
- chronological/purged OOS or untouched future holdout;
- full trial registry including failures;
- dependence-aware paired block bootstrap;
- FDR for planned pairwise hypotheses;
- SPA/White Reality Check plus DSR/PSR if adaptive multi-trial search occurs;
- no holdout threshold tuning.

### Gate X — execution progression

Even a supported predictive result cannot jump directly to LIVE. It must still pass prospective PAPER replication, testnet/reconciliation, independent risk checks and explicit live review.

## 14. Literature anchors

1. Anastasopoulos, A., Gradojevic, N., Liu, F., Maynard, A., & Tsiakas, I. (2026). *Order flow and cryptocurrency returns*. **Journal of Financial Markets, 79**, 101047. DOI `10.1016/j.finmar.2026.101047`. The study uses a cross-section of 84 cryptocurrencies and international order flows denominated in 11 currencies; this is materially richer than v0.19.
2. Easley, D., O'Hara, M., Yang, S., & Zhang, Z. (2026). *Microstructure and market dynamics in crypto markets*. **Journal of Financial Markets**. DOI `10.1016/j.finmar.2026.101071`. The study reports important own-market and cross-market microstructure effects for major cryptocurrencies.
3. Pindza, E. (2026). *Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets*. **Frontiers in Blockchain, 9**, 1811716. DOI `10.3389/fbloc.2026.1811716`. More than three million minute observations are evaluated with leakage-aware controls; predictive information is found but no strategy survives realistic standard retail fees.
4. Raffaelli, D., Cestari, R. G., Marazzina, D., et al. (2026). *Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data*. **Decisions in Economics and Finance**. DOI `10.1007/s10203-026-00570-z`. The paper uses real-time LOB event streams and motivates future higher-resolution measurement without converting REST snapshots into LOB-event claims.
5. Bysik, A., & Ślepaczuk, R. (2026). *Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting*. SSRN DOI `10.2139/ssrn.6795938`; arXiv `2606.00060`. This is a preprint/working paper, not used as the sole basis for any thesis claim; it motivates cost-aware forecast-to-trade conversion.
6. Bieganowski, B., & Ślepaczuk, R. (2026). *Explainable Patterns in Cryptocurrency Microstructure*. arXiv `2602.00776`. The study uses 1-second Binance Futures order-book/trade data and highlights stable feature patterns and execution-sensitive interpretation; it is supporting, not primary peer-reviewed evidence.
7. Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. **Journal of Portfolio Management, 40(5)**, 94–107. DOI `10.3905/jpm.2014.40.5.094`.
8. White, H. (2000). *A Reality Check for Data Snooping*. **Econometrica, 68(5)**, 1097–1126. DOI `10.1111/1468-0262.00152`.
9. Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. **Journal of Business & Economic Statistics, 23(4)**, 365–380. DOI `10.1198/073500105000000063`.

## 15. Claim contract

Through v0.19 the following may be claimed:

- specified v0.18 methodological defects were identified and corrected in a non-promotional audit layer;
- the original v0.18 artifact remains preserved;
- a fixed-window, timestamped, quality-gated multi-venue REST microstructure collector has been implemented;
- the primary 4h trading horizon is unchanged;
- prospective measurement can be used for a later frozen experiment if maturity gates are satisfied.

The following remain prohibited:

- declaring v0.18 cost-aware results profitable from a spent/refreshed sample;
- declaring the v0.11 regime interaction freshly replicated from the construction audit;
- describing provider-reported side as verified aggressor ground truth;
- describing REST snapshots as complete L2/L3 event history;
- claiming microstructure alpha before Gate Q/F/P;
- PAPER strategy replacement, testnet promotion or real-money LIVE execution.
