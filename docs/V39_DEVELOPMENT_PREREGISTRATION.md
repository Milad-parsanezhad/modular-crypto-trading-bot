# v0.39 Mother Strategy — Frozen Development Preregistration

Status: **FROZEN BEFORE EMPIRICAL EXECUTION**

Purpose: evaluate the executable v0.39 mother-strategy architecture under a causal, cost-aware, regularized learning protocol without modifying the frozen v0.39R result.

This experiment cannot authorize PAPER/LIVE execution and cannot touch Kraken.

## 1. Frozen development universe

Development venues only:

- CoinEx
- OKX
- KuCoin

Reserved external holdout:

- Kraken — **SEALED / UNTOUCHED**

Symbols:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- DOGE/USDT

Timeframe: `4h`

Fixed sample window:

- start: `2025-01-01T00:00:00Z`
- end: `2026-09-10T23:59:59Z`

No later candle may be appended to this experiment.

## 2. Frozen mother-strategy feature architecture

Use `research_bot.mother_strategy_v39` unchanged for this experiment.

Independent engines:

1. ICT — liquidity sweep, MSS/structure, displacement, premium/discount.
2. SMC — BOS/CHoCH, FVG, order-block mitigation, supply/demand retest.
3. Ichimoku — Kumo, Tenkan/Kijun, Kijun slope, breakout, pullback.
4. Brooks-inspired causal engine — Always-In, trend/range, breakout/failed-breakout, H1/H2, L1/L2, wedge, micro double top/bottom, signal bar, follow-through, measured-move context.
5. completed higher-timeframe context only.

No hard `k-of-n`, 6/7, unanimity, or post-result hand-written conjunction is allowed.

Candidate events are generated only by the frozen broad event pool in `mother_strategy_v39.py`.

## 3. Frozen label / trade-realization policy

For every candidate event at closed bar `t`:

- candidate side is `research_candidate_side_v39` and must be `-1` or `+1`;
- entry is `open[t+1]`;
- stop distance is the v0.39 structural/ATR stop fraction, never below `max(1.5 × ATR, 0.30% of price)`;
- target is `+3.0R`;
- maximum holding period is `30` 4h bars;
- if stop and target are both touched in the same bar, stop is assumed first;
- otherwise exit at the last allowed bar close;
- base round-trip cost = `24 bps`, converted to R by initial stop distance;
- stress round-trip cost = `36 bps`, converted the same way.

Training labels may overlap because they are supervised event outcomes; development strategy metrics use a causal one-active-trade-per-symbol realization after model selection.

## 4. Frozen causal preprocessing

Use `causal_robust_normalize` from `financial_system_v39.py`:

- rolling window = 180 bars;
- minimum history = 60 bars;
- location = shifted rolling median;
- scale = shifted rolling MAD × 1.4826 with prior-only standard-deviation fallback;
- normalized values clipped to `[-5,+5]`;
- missingness indicators included;
- neural input is finite `float32`.

Current/future observations may not contribute to their own scaler state.

## 5. Frozen model ladder

Sequential complexity ladder:

1. `V39_RIDGE`
2. `V39_HISTGB`
3. `V39_SHALLOW_MLP`

Selection rule: **the first / simplest model that passes every frozen development gate is the development winner.**

We do not choose the model with the highest return, Sharpe, or PF among the three.

If Ridge passes, HistGB/MLP cannot replace it in this experiment merely because they report larger returns.

PatchTST/CMamba are not candidates in this experiment. They require a new preregistered experiment only if the frozen ladder fails or if a separately stated representation-learning hypothesis is justified.

Frozen model settings:

### Ridge

- `Ridge(alpha=10.0)`

### HistGradientBoosting

- learning rate `0.05`
- max iterations `180`
- max leaf nodes `15`
- L2 regularization `1.0`
- deterministic seed `314`

### Shallow MLP

Three fixed seeds: `314`, `1618`, `2718`.

- two hidden layers: `(64, 32)`
- activation: ReLU
- optimizer: Adam/AdamW-equivalent regularized training policy available in current sklearn implementation
- L2 alpha: `1e-4`
- learning rate init: `3e-4`
- maximum iterations: `80`
- early stopping: enabled
- validation fraction: `0.15`
- no best-seed selection; final prediction is median of the three fixed-seed predictions.

If the production PyTorch MLP is substituted later, it is a new experiment unless predictions are demonstrated implementation-equivalent under this frozen protocol.

## 6. Frozen temporal evaluation

Use five expanding purged walk-forward folds over development time.

- initial history: approximately first 40% of unique signal timestamps;
- remaining time is split into five contiguous OOS test folds;
- embargo before every test fold: `30` bars;
- within every pre-test history, the most recent 20% is calibration and the earlier 80% is model fitting;
- training events must have `exit_time < calibration_start`;
- calibration events must have `exit_time < test_start`;
- test events are never used for fitting, scaling decisions, threshold selection, conformal calibration, or model-family selection.

## 7. Frozen uncertainty / capital-admission rule

For each fold/model:

- point target = post-cost R-multiple;
- one-sided lower confidence/conformal buffer is calibrated only on the calibration segment;
- model prediction is eligible only if `lower_expected_r > 0`;
- positive mean prediction alone is insufficient;
- no post-result change to the lower-bound rule is permitted.

The financial allocator then applies the frozen v0.39 policy:

- base stop-risk per trade: `0.25%` equity;
- max stop-risk per trade: `0.50%`;
- aggregate open stop-risk: `2.00%`;
- same-direction open stop-risk: `1.50%`;
- max nominal asset weight: `35%`;
- max portfolio gross: `70%`;
- drawdown risk scale: 1.00 below 2%, 0.75 at 2–3.5%, 0.50 at 3.5–5%;
- no new risk at 5% drawdown;
- no martingale, averaging down, revenge trading, discretionary override, or automatic risk increase after losses.

## 8. Frozen development gates

Each venue must independently pass all gates:

- realized selected trades `>= 200`;
- base-cost profit factor `>= 1.05`;
- base-cost mean net expectancy `> 0R`;
- positive-asset breadth `>= 0.60`;
- moving-block bootstrap 95% CI lower bound of account/trade mean `> 0`;
- positive-quarter fraction `>= 0.60`;
- 36 bps stress profit factor `>= 1.00`.

Cross-fold/seed gates:

- at least five purged folds;
- positive-fold fraction `>= 0.60`;
- for the stochastic MLP, at least 2/3 fixed seeds must have positive aggregate OOS expectancy;
- worst-group diagnostics are reported over venue × quarter × regime; no group result may be silently removed.

Bootstrap:

- 750 resamples;
- moving block length = 20 trades;
- deterministic seed = 314.

## 9. Frozen model-selection decision

Sequential decision:

1. evaluate Ridge;
2. if and only if Ridge fails one or more frozen gates, evaluate HistGB;
3. if and only if HistGB fails, evaluate shallow MLP;
4. first model passing every gate becomes `V39_DEVELOPMENT_WINNER`;
5. if none pass: `V39_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE`.

No threshold relaxation, model-family addition, feature deletion, feature revival, seed cherry-picking, or hand-selected regime exclusion is allowed after results.

## 10. Holdout and execution governance

Regardless of development outcome in this run:

- Kraken is not fetched by the v0.39 development runner;
- `kraken_touched = false`;
- PAPER execution = disabled;
- LIVE execution = disabled.

A development winner may justify a **separate holdout-authorization decision**, not automatic Kraken use.

## 11. Interpretation

This experiment asks:

> Can the reconstructed ICT + SMC + Ichimoku + Brooks + MTF mother strategy, when fused by a regularized causal learning process and constrained by the frozen financial governor, produce broad and temporally persistent post-cost development evidence without post-hoc tuning?

A negative result is retained as valid thesis evidence and cannot be converted into a positive result by changing the protocol under the same experiment identifier.
