# v0.19 Deep Bug Audit Findings

Date: 2026-09-10

Status: **Engineering/scientific audit; original v0.18 artifact preserved; prospective v0.19 measurement only**

## Executive finding

A second-pass audit found that v0.18 was correctly fail-closed, but several implementation/design details made its interpretation weaker than the headline tables suggested. A third measurement audit of the new v0.19 collector then found cross-venue trade-window, venue-counting and evidence-schema defects before the prospective dataset matured. None of these defects reveals a hidden profitable strategy. They reinforce the decision to keep promotion closed until fresh evidence survives stricter data contracts.

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
| M4 | Measurement comparability | **Critical** | `fetchTrades(limit=N)` represented different lookback durations on different exchanges | request with `since` and locally filter all venues to the same 60-second PIT window |
| M5 | Variable semantics | High | exchange-reported `side` could be over-described as true aggressor classification | label explicitly as `exchange_reported_side_unverified_aggressor` |
| M6 | Sampling design | High | one instantaneous REST snapshot every 4h under-sampled within-bar microstructure | target 30-minute measurement cadence while keeping the trading horizon frozen at 4h |
| M7 | Payload integrity | High | an early prototype hash could be computed before late metadata were attached | canonical final-payload SHA-256 excluding only its self-referential hash field |
| M8 | Cross-venue identity | **Critical** | accepted observations, rather than unique venues, could satisfy the venue-count gate | count unique venue names; deduplicate and fail closed on duplicate-provider observations |
| M9 | Evidence schema/universe | High | insufficient-coverage rows were structurally thinner and a missing configured symbol could disappear from the snapshot | stable fail-closed schema + explicit configured-symbol materialization |
| M10 | PIT metadata integrity | High | stored trade-window bounds were not independently validated against the frozen observation clock | validate `window_start=t-60s`, `window_end=t` and first/last trade timestamps |

## Important reproducibility correction

The original v0.18 artifact contains results but not the raw CoinEx OHLCV frame or its deterministic data fingerprint. Because the source endpoint is rolling, a later request for `1800` bars is not byte-identical to the original request even if configuration is unchanged.

Therefore v0.19 uses the explicit label:

`REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`

This means the corrected audit can answer “does the repaired implementation behave sensibly on a comparable refreshed sample?” but it cannot isolate the causal numerical effect of each bug on the exact original v0.18 holdout. The immutable v0.18 artifact remains part of the thesis record and is **not overwritten**.

## Why B1/B2 are the most consequential v0.18 scientific bugs

The v0.11 Ichimoku baseline was constructed cross-sectionally: synchronized assets were ranked by an Ichimoku score and the top quartile was held. v0.11 regime diagnostics then classified a market-level regime from aggregate regime flags.

v0.18-B changed both elements: it turned the score into a per-asset binary `score > 0` signal and allowed each asset to carry its own favorable regime flag. That was a different strategy and therefore could not be described as an exact external replication of the v0.11 portfolio-level hypothesis. v0.19 fixes the construction and deliberately refuses to call the repaired retrospective result fresh replication evidence.

## Why M4 is the most consequential prospective-collector bug

The first multi-venue snapshot used a fixed **count** of recent public trades rather than a fixed **time interval**. Public REST APIs and CCXT adapters can return different default histories and provider-specific caps. A list of 500 recent trades on one venue can represent seconds, while 100 trades on another venue can represent a materially different interval.

A cross-venue mean of those imbalances is therefore not a like-for-like measurement.

The repaired collector now:

1. freezes the order-book observation clock;
2. requests public trades using `since = observed_at - 60 seconds` where the unified provider supports it;
3. locally filters every provider to the closed PIT interval `[t-60s, t]`;
4. excludes and counts any returned trade with timestamp `> t`;
5. records first/last recent trade timestamps, recent trade count, raw count and staleness;
6. rejects venues with fewer than 5 recent trades or last-trade staleness above 30 seconds;
7. preserves the exchange-reported side but does **not** silently promote it to validated aggressor ground truth.

This converts the feature from “latest-N trade imbalance” into a more defensible **fixed-window reported trade-flow snapshot**.

## Why M8 matters even after M4 is fixed

Cross-venue evidence is meaningful only when coverage represents distinct providers. Counting observations instead of unique venue identifiers creates a subtle pseudo-replication failure: two observations from CoinEx are not two independent venues. The final v0.19 implementation therefore groups valid observations by venue, retains only the latest valid observation per venue for diagnostics, records `accepted_venue_names`, and places the whole symbol under a fail-closed `DUPLICATE_VENUE_OBSERVATIONS` quality flag when a duplicate-provider condition is detected.

The regression suite now proves both cases: duplicates cannot manufacture the minimum venue count, and duplicates cannot silently pass even when another valid venue is present.

## Stable fail-closed schema and universe materialization

A second subtle integrity risk was schema asymmetry. Earlier insufficient-venue rows did not expose all normal quality fields, which could cause downstream evaluators to fail or, worse, special-case missing values inconsistently. v0.19 now emits the same core diagnostic keys for insufficient coverage, with `None` where a statistic is not valid.

The configured universe is also explicit. If BTC or ETH has no usable observations in a collection cycle, the symbol remains in the snapshot with `NO_OBSERVATIONS_FOR_CONFIGURED_SYMBOL`; it cannot disappear and thereby improve measured coverage through omission.

## Sampling-frequency correction

Recent microstructure studies use second/minute event or LOB data. A single instantaneous REST observation every four hours is too sparse to characterize within-bar liquidity/flow dynamics. v0.19 therefore changes the **measurement cadence** to a target of 30 minutes while leaving the thesis forecast/trading horizon at **4 hours**.

This does not create eight independent 4h targets from one bar. Later, the within-bar measurements may be aggregated into frozen 4h features such as mean/median/spread/slope/dispersion, but only after the data-quality pilot is complete. Scheduled GitHub jobs can start late, so actual event timestamps — not nominal cron labels — remain authoritative.

## Literature-driven interpretation

The direction of this repair is supported by recent evidence but not validated by it:

- Anastasopoulos et al. (2026), *Journal of Financial Markets*, study a much richer “world order flow” constructed from international flows in 11 currencies and find OOS predictive content. Their object is **not** equivalent to our three-venue USDT REST snapshot. DOI `10.1016/j.finmar.2026.101047`.
- Easley, O'Hara, Yang & Zhang (2026), *Journal of Financial Markets*, report own-market and cross-market microstructure effects for major cryptocurrencies. DOI `10.1016/j.finmar.2026.101071`.
- Pindza (2026), *Frontiers in Blockchain*, uses more than three million minute observations, leakage-aware walk-forward evaluation and realistic fee analysis. The key caution for this project is that microstructure signal can be genuine yet too weak to survive retail trading costs. DOI `10.3389/fbloc.2026.1811716`.
- Raffaelli et al. (2026), *Decisions in Economics and Finance*, use real-time LOB event streams for high-frequency BTC forecasting, showing why a REST snapshot pilot must not be described as a reconstructed L2/L3 event stream. DOI `10.1007/s10203-026-00570-z`.
- Bysik & Ślepaczuk (2026) report that naive sign-based BTC trading can fail after 10-bp costs and that cost-aware execution materially changes turnover/economic outcomes in selected walk-forward configurations. SSRN DOI `10.2139/ssrn.6795938`.

Literature informs the hypothesis; only this project's prospective data can decide promotion.

## Remaining known limitations, not bugs

- The current multi-venue flow is a crypto-exchange order-flow proxy, **not** the same “world order flow” used by Anastasopoulos et al. (2026).
- Public REST history can still be provider-capped; the fixed-window metadata exposes recent-count/staleness and prevents silent interpretation as a complete consolidated tape.
- Exchange-reported trade `side` is retained as reported data. Until venue-specific semantics are independently verified, the variable is called **reported trade imbalance**, not definitive buyer/seller aggressor ground truth.
- Top-of-book/depth snapshots are discrete REST observations, not reconstructed full event streams. L2/L3 historical/event-stream claims are prohibited.
- A constant 12-bp backtest friction is a controlled comparison assumption, not a complete realized implementation-shortfall model. Spread/slippage/impact sensitivity remains required for any candidate approaching promotion.
- v0.18's full round-trip hurdle is conservative but not state-aware. A state-aware entry/hold/exit cost hurdle would be a **new hypothesis**, not a silent bug fix.

## Research verdict

More research is required, but it should target **measurement fidelity, prospective evidence and cost-aware inference**, not model complexity for its own sake.

Priority order:

1. complete at least 7 elapsed days of dense prospective microstructure measurement;
2. audit fixed-window recent-trade coverage, staleness, unique-venue coverage, scheduler delay and timestamp synchronization;
3. aggregate sub-4h measurements into a pre-registered 4h feature vector without changing the target horizon;
4. accumulate at least 250 independent quality-filtered 4h decision timestamps before a new predictive comparison;
5. freeze price-only versus price+microstructure candidates and register every trial before evaluation;
6. apply dependence-aware bootstrap/FDR and, if multi-trial search is performed, SPA/Reality Check and DSR/PSR;
7. only after a candidate survives those gates consider deeper DL/RL execution challengers or a higher-fidelity WebSocket/L2 collector.

## Safety conclusion

No v0.19 audit result can authorize PAPER strategy replacement, testnet promotion or real-money LIVE execution. The only fresh evidence path is the prospective v0.19 measurement stream.
