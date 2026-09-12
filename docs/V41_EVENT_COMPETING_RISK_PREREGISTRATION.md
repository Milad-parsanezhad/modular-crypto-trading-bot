# v0.41 Event-Specific Competing Risk — Frozen Preregistration

Date frozen: 2026-09-12

## Motivation

v0.40 showed that pooled binary meta-labeling has essentially no OOS discrimination under the current target definition. The next hypothesis is therefore a representation/target change, not a capacity increase.

v0.41 preserves the v0.39 mother-strategy feature lineage but assigns every causal mother event to one frozen semantic setup family and models the trade path as two competing absorbing causes:

- TARGET first;
- STOP first;
- TIME is right-censoring / no absorbing cause before the frozen horizon.

## Frozen event families and priority

Exactly one family is assigned at signal time. Highest priority wins:

1. ICT_MSS
2. BROOKS_H2L2
3. ICHIMOKU_BREAKOUT
4. ICHIMOKU_PULLBACK
5. SMC_OB_RETEST
6. BROOKS_FAILED_BREAKOUT
7. SMC_FVG_STRUCTURE
8. LIQUIDITY_SWEEP
9. OTHER_MOTHER_EVENT

Direction is modeled separately as LONG or SHORT. Regime (`trend`, `range`, `transition`) is an independently causal input.

## Target construction

For each event, the existing v0.39 conservative trade realization is reused unchanged:

- next-open entry;
- structural stop with frozen minimum distance;
- target = 3R;
- maximum hold = 30 four-hour bars;
- stop-first if stop and target are touched on the same bar;
- 24 bps base round-trip costs;
- 36 bps stress costs.

The event is expanded into discrete at-risk rows from bar 1 through its observed event/censoring duration. Only signal-time covariates plus deterministic elapsed-time features are available to the hazard models.

## Competing-risk model

Two cause-specific hazards are fitted:

- `h_target(t | X)`;
- `h_stop(t | X)`.

The cumulative incidence functions are reconstructed sequentially under survival:

`CIF_target += S(t-1) * h_target(t)`

`CIF_stop += S(t-1) * h_stop(t)`

`S(t) = S(t-1) * (1 - h_target(t) - h_stop(t))`

Remaining survival at bar 30 is the TIME probability.

A separate timeout-value head estimates post-cost R conditional on TIME. Expected post-cost R is:

`P(target)*(3R-cost_R) + P(stop)*(-1R-cost_R) + P(timeout)*E[R_timeout]`

## Candidate order

1. `V41_LOGIT_CAUSE_SPECIFIC`
   - L2 pooled-logistic target and stop hazards;
   - Huber timeout-R head.
2. `V41_HISTGB_CAUSE_SPECIFIC`
   - regularized HistGradientBoosting target and stop hazards;
   - HistGradientBoosting timeout-R head.

The first/simplest candidate passing every frozen gate is the development winner.

## Hierarchical experts

The pooled model always exists. A family × side expert is fitted only when the fit sample has:

- at least 300 events in that family × side;
- at least 40 TARGET events;
- at least 40 STOP events.

Otherwise prediction falls back to the pooled model. These minima are frozen before results.

## Frozen semantic feature vector

v0.41 uses a compact, preregistered subset of the causally normalized v0.39 mother-strategy representation:

- ATR/returns/trend/volume context;
- ICT/SMC/Ichimoku/Brooks/MTF engine scores;
- engine agreement/dispersion/prior;
- liquidity sweep and MSS flags;
- displacement and premium/discount;
- OB/supply-demand retests;
- Ichimoku breakout/pullback;
- Brooks H1/H2/L1/L2, failed breakout and follow-through.

Family, side, regime and deterministic elapsed-time features are appended. No feature may be added or removed after observing v0.41 results under this experiment ID.

## Uncertainty and admission

The existing purged fold calibration segment is used only after model fitting. A one-sided split-conformal lower expected-R buffer is calculated at alpha = 0.20.

Family-specific conformal buffers are used when a family has at least 30 calibration observations; otherwise a global buffer is used.

An OOS event is selected only when:

- conformal lower expected R > 0;
- cumulative P(TARGET) > cumulative P(STOP).

There is no top-k selection and no post-result threshold search.

## Seed policy

Frozen seeds: 314, 1618, 2718.

Three seed predictions are combined by row-wise median. Best-seed selection is forbidden. At least 2/3 seed-level OOS expectancy diagnostics must be positive.

## Development data

Consumed development venues only:

- CoinEx
- OKX
- KuCoin

Reserved external holdout:

- Kraken — SEALED / FORBIDDEN

Symbols:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- DOGE/USDT

Timeframe: 4h.

Requested window: `2025-01-01T00:00:00Z` through `2026-09-10T23:59:59Z`.

## Validation

Unchanged from v0.39/v0.40:

- five purged chronological folds;
- 30-bar embargo;
- zero-selection folds count as non-positive;
- venue-independent capital/risk accounting;
- no same-test rescue tuning.

Every venue must pass all frozen gates:

- executed trades >= 200;
- Profit Factor >= 1.05;
- expectancy > 0R;
- positive-asset breadth >= 0.60;
- moving-block bootstrap CI lower bound > 0;
- positive-quarter fraction >= 0.60;
- stress 36bps PF >= 1.00;
- maximum account drawdown <= 5%.

Cross-fold / seed gates:

- positive fold fraction >= 0.60;
- positive seed fraction >= 2/3.

Additional diagnostics, not extra pass/fail gates:

- target/stop/time calibration by family;
- Brier scores for target and stop cumulative incidence;
- family-specific expectancy and sample counts;
- venue × quarter × regime worst groups;
- fraction of predictions using family-specific experts.

## Financial governance

Unchanged:

- 0.25% base stop-risk per trade;
- 0.50% max risk per trade;
- 2.00% aggregate open stop-risk;
- 1.50% same-direction open risk;
- 35% max asset nominal weight;
- 70% max gross;
- risk reduction at 2% and 3.5% drawdown;
- 5% hard drawdown firewall;
- two-day cooldown after three consecutive losses;
- no martingale, averaging down, revenge trading, discretionary override, or automatic risk increase after loss.

## Anti-overfitting prohibitions

After observing v0.41 results, the following are forbidden under this experiment ID:

- changing family priority;
- deleting weak families, sides, regimes, venues or quarters;
- changing expert sample minima;
- changing conformal alpha;
- changing target, stop or horizon;
- selecting the best seed;
- increasing leverage or risk;
- adding neural sequence models;
- touching Kraken.

Any change requires a new experiment identifier and preregistration.

## Execution state

`PAPER = false`

`LIVE = false`

`KRAKEN_TOUCHED = false`
