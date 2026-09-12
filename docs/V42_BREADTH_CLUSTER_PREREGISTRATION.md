# v0.42 Breadth Expansion + Cluster-Robust Validation — Preregistration

## Status

**FROZEN BEFORE v0.42 OUTCOME INSPECTION**

This experiment follows the frozen negative/insufficient v0.41 qualification result. v0.41 showed promising HistGradientBoosting cause-specific competing-risk economics, but no venue reached the preregistered minimum of 200 financially executed trades; CoinEx also had a negative moving-block CI lower bound because its available history was materially shorter.

v0.42 tests whether the v0.41 HistGB signal generalizes across a broader development universe. It is not a threshold-rescue experiment.

## Frozen hypothesis

The unchanged v0.41 HistGB cause-specific competing-risk architecture will retain positive post-cost expectancy and robustness when applied to a broader, outcome-independently eligible asset universe on the already-consumed development venues.

## What is unchanged from v0.41

- mother strategy semantics: ICT + SMC + Ichimoku + expanded Al Brooks + HTF context;
- event-family definitions;
- v0.41 feature representation;
- `V41_HISTGB_CAUSE_SPECIFIC` model family and hyperparameters;
- frozen seeds: `314, 1618, 2718`;
- row-wise median seed aggregation; no best-seed selection;
- discrete-time competing TARGET/STOP hazards;
- TIME/right-censoring and timeout-R head;
- 30-bar horizon;
- family-conditional conformal lower bound with global fallback;
- admission rule: `lower_expected_r > 0` and `P(target) > P(stop)`;
- 5 purged temporal folds;
- 30-bar embargo;
- base round-trip cost = 24 bps;
- stress round-trip cost = 36 bps;
- risk policy: 0.25% base risk, 2% aggregate stop-risk cap, 1.5% same-direction cap, 5% hard drawdown firewall;
- independent venue financial accounting;
- PAPER=false; LIVE=false; Kraken SEALED.

## Candidate asset universe — frozen before data outcomes

The following 25 USDT spot pairs are the only v0.42 candidates:

`BTC, ETH, SOL, XRP, DOGE, ADA, LINK, AVAX, LTC, BCH, DOT, TRX, ATOM, NEAR, ETC, FIL, UNI, AAVE, SUI, TON, XLM, ALGO, ICP, ARB, OP`.

No asset may be added because it performed well after the run starts.

### Outcome-independent eligibility

A candidate symbol is included only if:

1. it is available on **all three** development venues: CoinEx, OKX, KuCoin; and
2. each venue provides at least **900 valid 4h bars** in the frozen request window.

Eligibility is decided from market availability/bar count only, before inspecting mother-event outcomes, model scores, expectancy, PF, or drawdown.

At least **18 common eligible symbols** must remain. If fewer than 18 satisfy availability, the experiment is `DATA_UNAVAILABLE`; thresholds will not be relaxed.

## Frozen date window

- request start: `2025-01-01T00:00:00Z`
- request end: `2026-09-10T23:59:59Z`
- timeframe: `4h`

CoinEx may expose less historical depth than requested. That limitation is recorded in the data manifest and is not repaired by outcome-based symbol selection.

## New robustness test: hierarchical symbol × time-block bootstrap

Simply increasing the number of correlated crypto assets can inflate nominal sample size. Therefore v0.42 adds a new gate:

1. within each development venue, sample eligible symbols with replacement;
2. for every sampled symbol, resample its time-ordered realized `net_r` values using a circular moving-block bootstrap;
3. combine the resampled symbol paths and compute mean expectancy;
4. repeat 1,000 times using frozen seed 314;
5. require the 2.5% lower quantile of bootstrap expectancy to be strictly positive.

Frozen within-symbol block length: **10 executed trades**.

This gate is in addition to—not a replacement for—the original moving-block CI.

## Per-venue qualification gates

Every development venue must independently satisfy all of the following:

- financially executed trades >= 200;
- profit factor >= 1.05;
- expectancy > 0R;
- positive-asset fraction >= 0.60;
- original moving-block CI lower bound > 0;
- hierarchical symbol×time cluster-bootstrap lower bound > 0;
- stress-cost PF at 36 bps >= 1.00;
- positive-quarter fraction >= 0.60;
- max account drawdown <= 5%.

## Cross-fold / seed gates

- positive fold fraction >= 0.60 across exactly five frozen folds;
- positive frozen-seed fraction >= 2/3;
- no selection of the best seed.

## Anti-overfit prohibitions

After outcomes are observed under experiment v0.42, none of the following may be changed under the same experiment ID:

- lower-bound admission threshold;
- `P(target)>P(stop)` rule;
- event-family priority;
- event-family pruning;
- direction-specific pruning;
- HistGB hyperparameters;
- seeds;
- costs;
- horizon;
- risk limits;
- minimum 200 trade gate;
- bootstrap block lengths;
- symbol eligibility minimum;
- candidate symbol list.

Any change requires a new preregistered experiment.

## Kraken governance

Kraken is the reserved external holdout and **must not be instantiated, queried, or fetched in v0.42**. Kraken can be authorized only if the complete frozen v0.42 development protocol produces a winner that passes all development gates.

## Interpretation

A v0.42 pass would establish a development winner, not live-trading authorization. The next admissible step would be a one-shot, preregistered Kraken holdout. A v0.42 failure remains valid evidence and PAPER/LIVE stay disabled.
