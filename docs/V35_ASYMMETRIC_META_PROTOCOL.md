# v0.35 — Asymmetric Regime + Cross-Sectional Meta-Labeling Protocol

Status: **research only**. Forward PAPER replacement and LIVE execution are explicitly unauthorized.

## Motivation

The completed v0.31-v0.34 chain showed an economically positive but statistically unstable daily CUSUM-breakout family. The consumed KuCoin attribution found only 55% positive-asset breadth and materially stronger short-side economics than long-side economics. The subsequent v0.33 breadth filter failed to create a robust three-venue winner. Therefore v0.35 introduces a **new preregistered hypothesis family** rather than retuning the rejected v0.33 breadth thresholds.

Recent crypto momentum research motivates two additional state variables: persistent market-regime transitions and cross-sectional dispersion. State-transition evidence suggests momentum is concentrated in persistent regimes, while more recent cross-sectional evidence suggests dispersion is associated with weaker subsequent crypto momentum. These concepts are used as hypothesis inputs, not as post-hoc excuses to relax gates.

References used for the hypothesis design:
- *State transitions and momentum effect in cryptocurrency market*, Finance Research Letters (2025), DOI: 10.1016/j.frl.2025.108356.
- *Cross-Sectional Dispersion and the State Dependence of Cryptocurrency Momentum* (2026 working paper, SSRN 6648082).
- Internal v0.24 strategy-aware meta-labeling protocol for the point-in-time/meta-labeling discipline.

## Evidence boundary

Consumed development evidence:
- CoinEx frozen v0.30 daily snapshots;
- OKX frozen v0.30 daily snapshots;
- KuCoin frozen v0.31 daily snapshot and holdout evidence.

Untouched evidence:
- **Kraken remains sealed.** v0.35 contains no Kraken downloader and no Kraken market-data request.

A v0.35 development winner may only authorize a later, separately preregistered Kraken holdout. v0.35 itself cannot consume Kraken.

## Frozen base economics

The new event family inherits the bracket economics of `V30_D1_CUSUM_BREAKOUT_3`:
- timeframe: 1d;
- CUSUM threshold: inherited and frozen;
- breakout lookback: inherited and frozen;
- stop ATR multiple / reward-risk / maximum holding period: inherited and frozen;
- round-trip friction: 24 bps;
- v0.25 portfolio limits: 2.0% aggregate and 1.5% same-direction open loss-at-stop;
- v0.31 pre-entry hard-drawdown firewall: 5%.

The rejected v0.33 breadth thresholds are **not** tuned or reused as the v0.35 selection mechanism.

## New asymmetric regime hypothesis

Long and short signals no longer share identical regime semantics:
- long: persistent BTC `UP-UP` regime only;
- short: persistent `DOWN-DOWN` or transition state (`regime <= 0`).

The short-side asymmetry is preregistered because consumed v0.32 evidence showed stronger short-side economics. It is not evidence that future short trades will be profitable.

## Cross-sectional context

For every signal timestamp and asset, v0.35 computes causally available:
- 20-day return percentile rank;
- 60-day return percentile rank;
- their equal-weight relative-strength score;
- cross-sectional 20-day momentum dispersion;
- a lagged 90-bar rolling 80th-percentile dispersion reference;
- the actual point-in-time BTC persistent-regime state.

No future row participates in historical ranks or the lagged dispersion reference.

## Causal meta-label

The meta-label target is whether a completed event has **positive post-cost R-multiple**. The score is a duration-weighted empirical-Bayes probability estimate with a 0.50 prior and prior strength 4.

Crucial availability rule:

`historical exit_time < current signal_time`

An event whose exit occurs exactly at the new signal timestamp remains unavailable to that signal. This prevents same-timestamp outcome leakage.

The score hierarchy is deterministic:
1. side + actual regime + relative-strength bucket + dispersion bucket;
2. side + actual regime;
3. side;
4. global settled-event history.

Minimum history is 20 settled events before a trade may pass the meta layer.

## Six preregistered candidates

Exactly six side-specific rank/meta combinations are tested:

1. `V35_D1_ASYM_META_L55_S45_M50_45`
2. `V35_D1_ASYM_META_L60_S45_M50_45`
3. `V35_D1_ASYM_META_L60_S40_M50_45`
4. `V35_D1_ASYM_META_L60_S40_M55_45`
5. `V35_D1_ASYM_META_L65_S40_M55_45`
6. `V35_D1_ASYM_META_L65_S35_M55_50`

This increases the cumulative effective-trial count from 114 to **120**.

## Three-venue development gate

A candidate is eligible only if **every one** of CoinEx, OKX and KuCoin satisfies the unchanged gate:
- at least 200 executed trades;
- Profit Factor >= 1.05;
- expectancy > 0 R;
- positive-asset fraction >= 60%;
- absolute maximum drawdown <= 5%;
- moving-block CI lower bound > 0.

Winner ranking among eligible candidates is deterministic and lexicographic:
1. highest worst-venue block-CI lower bound;
2. highest worst-venue breadth;
3. highest worst-venue PF;
4. highest worst-venue expectancy;
5. highest minimum trade count;
6. lowest worst-venue drawdown.

## Decision states

If no candidate passes:
- `NO_V35_ROBUST_DEVELOPMENT_CANDIDATE`
- Kraken remains untouched.

If one candidate passes and is deterministically locked:
- `V35_ASYMMETRIC_META_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT`
- Kraken still remains untouched until a new fail-closed holdout workflow is preregistered.

Neither state authorizes PAPER replacement or LIVE execution.

## Prohibited actions

- no threshold relaxation after results;
- no retuning the rejected v0.33 candidates;
- no winner reselection after future Kraken inspection;
- no historical holdout recycling;
- no claim of profitability from a green CI run;
- no PAPER replacement or LIVE execution authorization.
