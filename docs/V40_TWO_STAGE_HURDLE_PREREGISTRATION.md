# v0.40 Two-Stage Hurdle — Frozen Preregistration

Date frozen: 2026-09-12

## Scientific motivation

v0.39 repaired the v0.39R zero-event failure but exposed a bias/variance-admission frontier:

- Ridge: too few admitted trades after conservative uncertainty gating;
- HistGB: modest positive venue means but insufficient temporal/breadth/bootstrap robustness;
- shallow MLP: more admissions but negative OOS economics and seed instability.

The new hypothesis is **not** that more model capacity is required. The new hypothesis is that a single regression target is structurally mismatched to the problem. v0.40 therefore separates event admissibility from conditional payoff magnitude.

## Frozen architecture

### Stage 1 — meta-label

Target:

`y_meta = 1(net_r > 0)`

Two candidate model families, evaluated in this exact order:

1. `V40_LOGIT_HUBER_HURDLE`
   - L2 logistic classifier;
   - Huber positive-magnitude regressor;
   - Huber loss-magnitude regressor;
   - Huber duration regressor.
2. `V40_HISTGB_HURDLE`
   - HistGradientBoosting classifier;
   - HistGradientBoosting positive/loss magnitude regressors;
   - HistGradientBoosting duration regressor.

The first/simplest candidate passing every frozen gate is the development winner.

### Stage 2 — conditional economics

Separate heads estimate:

- positive payoff magnitude `E[R+ | X]`;
- absolute non-positive payoff magnitude `E[|R-| | X]`;
- holding duration in bars.

Expected post-cost R is reconstructed as:

`E[R|X] = p_win * E[R+|X] - (1-p_win) * E[|R-| | X]`

This avoids forcing one regressor to model the discontinuous trade/no-trade and payoff-magnitude problems simultaneously.

## Calibration and uncertainty

Each purged fold retains the v0.39 fit / calibration / test separation.

The calibration segment is split chronologically and **without overlap**:

- first 50%: Platt probability calibration only;
- second 50%: conformal residual calibration only.

No target-test label is used for fitting or calibration.

Conformal lower and upper buffers are computed separately by market regime (`trend`, `range`, `transition`) when the regime has at least 30 calibration observations. Otherwise the global calibration buffer is used.

Frozen conformal alpha: `0.20`.

## Admission rule

An OOS event is selected only when all conditions hold:

- calibrated `p_win >= 0.55`;
- expected post-cost R `> 0`;
- regime-aware conformal lower expected R `> 0`.

There is no post-result percentile selection and no top-k selection.

## Seed policy

Frozen seeds:

- 314
- 1618
- 2718

Predictions are combined by row-wise median. The best seed is never selected after observing results.

At least 2/3 seed-level OOS expectancy diagnostics must be positive.

## Mother strategy and data

The v0.39 mother-strategy feature definitions are frozen and reused unchanged:

- ICT;
- SMC;
- Ichimoku;
- expanded Brooks engine;
- completed higher-timeframe context;
- causal robust normalization.

No v0.40 feature may be added after seeing the v0.40 result.

Development venues only:

- CoinEx
- OKX
- KuCoin

Reserved external holdout:

- Kraken — **SEALED / FORBIDDEN**

Frozen symbols:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- DOGE/USDT

Timeframe: `4h`.

Requested window: `2025-01-01T00:00:00Z` through `2026-09-10T23:59:59Z`.

## Walk-forward protocol

- five purged chronological folds;
- 30-bar embargo;
- event training requires event exit before the next protected period;
- zero-selection folds count as non-positive;
- financial accounting is independent for each venue.

## Costs and trade realization

Unchanged from v0.39:

- base round-trip cost: `24 bps`;
- stress round-trip cost: `36 bps`;
- target: `3R`;
- max hold: `30` 4h bars;
- structural stop with minimum `1.5 ATR` and 0.30% stop fraction;
- conservative stop-first ordering if stop and target are both touched in one bar.

## Financial governance

Unchanged from v0.39:

- base stop-risk/trade: `0.25%` equity;
- max stop-risk/trade: `0.50%`;
- aggregate open stop-risk: `2.00%`;
- same-direction open stop-risk: `1.50%`;
- max asset nominal weight: `35%`;
- max portfolio gross: `70%`;
- drawdown scaling at 2% and 3.5%;
- hard drawdown firewall: `5%`;
- after three consecutive losses: two-day cooldown;
- no martingale, averaging down, revenge trading, discretionary override, or automatic risk increase after loss.

## Frozen development qualification gates

Every development venue must pass all of the following:

- financially executed trades >= 200;
- Profit Factor >= 1.05;
- expectancy > 0R;
- positive-asset breadth >= 0.60;
- moving-block bootstrap CI lower bound > 0;
- positive-quarter fraction >= 0.60;
- stress-36bps Profit Factor >= 1.00;
- max account drawdown <= 5%.

Cross-fold / seed gates:

- positive fold fraction >= 0.60;
- positive seed fraction >= 2/3.

Brier score and Brier-skill versus fold climatology are recorded as diagnostics for Stage 1 but are not additional pass/fail gates in this experiment.

## Anti-overfitting prohibitions

After results are observed, v0.40 does **not** authorize:

- lowering `p_win` below 0.55;
- changing conformal alpha;
- merging probability and conformal calibration segments;
- choosing the best seed;
- deleting bad quarters/regimes/venues;
- changing v0.39 mother-strategy features;
- increasing risk/leverage;
- introducing PatchTST/CMamba under the same experiment ID;
- touching Kraken.

Any such change is a new hypothesis and requires a new experiment identifier and preregistration.

## Execution authorization

`PAPER = false`

`LIVE = false`

`KRAKEN_TOUCHED = false`
