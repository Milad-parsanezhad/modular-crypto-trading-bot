# v0.19 Deep Bug Audit Findings

Date: 2026-09-10

Status: **Engineering/scientific audit; original v0.18 artifact preserved**

## Executive finding

A second-pass audit found that v0.18 was correctly fail-closed, but several implementation/design details made its interpretation weaker than the headline tables suggested. These defects do not create a hidden profitable strategy. They make the conservative v0.18 non-promotion decision even more appropriate.

## Severity classification

| ID | Type | Severity | Effect | v0.19 action |
|---|---|---|---|---|
| A1 | Statistical calibration | High | in-sample residual MAD can understate uncertainty | expanding chronological OOF MAD |
| A2 | Execution accounting | Medium | final open position may avoid terminal closing cost | forced terminal flattening cost |
| A3 | Reproducibility | High | raw v0.18 sample cannot be reconstructed byte-for-byte | dataset SHA-256 + coverage timestamps from v0.19 onward |
| B1 | Hypothesis fidelity | Critical | v0.18-B binary per-asset rule did not match v0.11 cross-sectional top-quartile portfolio | restore original portfolio geometry |
| B2 | Hypothesis fidelity | Critical | asset-level regime gating did not match v0.11 market-level regime diagnostic | restore v0.11 market-level `_classify_regime` |
| B3 | Research design | High limitation | contemporaneous exchange transfer is not independent temporal replication | relabel as retrospective construction audit |
| M1 | Provider adapter | Medium | KuCoin rejected order-book limit 10 | normalize KuCoin depth limit to 20/100 |
| M2 | Unit/accounting | High | prototype CoinEx depth notional approximated all depth at best price | preserve exact sum(price × quantity) across levels |
| M3 | Point-in-time synchronization | High | cross-venue snapshot lacked a maximum clock-skew gate | add 60-second cross-venue clock-skew quality gate |

## Important reproducibility correction

The original v0.18 artifact contains results but not the raw CoinEx OHLCV frame or its deterministic data fingerprint. Because the source endpoint is rolling, a later request for `1800` bars is not byte-identical to the original request even if configuration is unchanged.

Therefore v0.19 uses the explicit label:

`REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`

This means the corrected audit can answer “does the repaired implementation behave sensibly on a comparable refreshed sample?” but it cannot isolate the causal numerical effect of each bug on the exact original v0.18 holdout.

The immutable v0.18 artifact remains part of the thesis record and is **not overwritten**.

## Why B1/B2 are the most consequential scientific bugs

The v0.11 Ichimoku baseline was constructed cross-sectionally: synchronized assets were ranked by an Ichimoku score and the top quartile was held. v0.11 regime diagnostics then classified a market-level regime from aggregate regime flags.

v0.18-B changed both elements:

- it turned the score into a per-asset binary `score > 0` signal;
- it allowed each asset to carry its own favorable regime flag.

That was a different strategy and therefore could not be described as an exact external replication of the v0.11 portfolio-level hypothesis. v0.19 fixes the construction and deliberately refuses to call the repaired retrospective result fresh replication evidence.

## New microstructure collector bugs caught during smoke testing

The first v0.19 smoke run was scientifically useful because it exposed provider/measurement problems before a long forward history accumulated:

1. KuCoin's CCXT adapter rejected `fetchOrderBook(limit=10)` and requires a supported limit such as 20 or 100. The adapter now normalizes the requested depth.
2. The first CoinEx collector prototype had only aggregate base quantity from `DepthSnapshot` and reconstructed quote depth at the best price. That is biased when deeper levels differ in price. `DepthSnapshot` now preserves exact quote notional as `sum(price_i * quantity_i)` while retaining base quantity for paper-execution sizing.
3. Cross-venue observations were sequential but no maximum clock-skew rule existed. v0.19 now records `venue_clock_skew_seconds` and quality-gates snapshots exceeding 60 seconds.

These are exactly the kinds of data-contract defects that should be found during a pilot rather than after model training.

## Remaining known limitations, not bugs

- The current multi-venue flow is a crypto-exchange order-flow proxy, **not** the same “world order flow” denominated across 11 fiat currencies used by Anastasopoulos et al. (2026).
- Latest-N-trades observations are not identical to a fixed-duration trade-flow window on every venue. For the prospective pilot, imbalance ratios are retained; before predictive promotion, trade-window duration/staleness must be audited and either normalized or explicitly modeled.
- Top-of-book/depth snapshots are discrete observations, not reconstructed full event streams. L2/L3 historical microstructure claims are therefore prohibited.
- A constant 12-bp backtest friction is a controlled comparison assumption, not a complete realized implementation-shortfall model. Spread/slippage/impact sensitivity remains required for any candidate approaching promotion.
- v0.18's full round-trip hurdle is conservative but not state-aware. A state-aware entry/hold/exit cost hurdle would be a **new hypothesis**, not a silent bug fix.

## Research verdict

More research is required, but it should target **data quality and genuinely new evidence**, not model complexity for its own sake.

Priority order:

1. complete the 7-day prospective microstructure data-quality pilot;
2. audit trade-window duration, staleness, venue coverage and timestamp synchronization;
3. accumulate at least 250 independent 4h decision timestamps before a new predictive comparison;
4. freeze price-only versus price+microstructure models and trial registry before evaluation;
5. apply dependence-aware bootstrap/FDR and, if a multi-trial search is performed, SPA/Reality Check and DSR/PSR;
6. only after a candidate survives those gates consider deeper DL/RL execution challengers.

## Safety conclusion

No v0.19 audit result can authorize PAPER strategy replacement, testnet promotion or real-money LIVE execution. The only fresh evidence path is the prospective v0.19 collection stream.
