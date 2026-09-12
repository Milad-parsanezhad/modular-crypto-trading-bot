# v0.39 Mother Strategy — Preregistered Architecture

Status: **research/development only**  
PAPER execution: **disabled**  
LIVE execution: **disabled**  
Reserved external holdout: **Kraken SEALED**

## Objective

Build one executable research architecture from the thesis strategy lineage while preventing the v0.39R zero-trade failure caused by a hard conjunctive admission rule. ICT, SMC, Ichimoku, Brooks and completed higher-timeframe context therefore remain independent causal engines. Their outputs become model features; no engine can silently veto all other engines.

## Base clock and causality

- Base timeframe: `4h`.
- Higher-timeframe context for 4h decisions: completed Daily bars only.
- Decisions use closed bar `t`; no feature may use future candles.
- Any future perturbation after a cutoff must leave all earlier engine outputs unchanged.
- Kraken data are not fetched in v0.39 development.

## Engine 1 — ICT

Objective proxies:

- liquidity sweep / raid;
- recent market-structure shift (MSS) / CHoCH or BOS after a sweep;
- displacement;
- premium/discount location.

Output: signed `ict_score_v39` plus component flags. ICT is not merged semantically with SMC.

## Engine 2 — SMC

Objective proxies:

- BOS / CHoCH structure;
- FVG / imbalance;
- order-block mitigation;
- supply/demand retest.

Output: signed `smc_score_v39` plus component flags.

## Engine 3 — Ichimoku

Causal 9/26/52 state with:

- Kumo direction;
- Tenkan/Kijun alignment;
- Kijun slope;
- cloud breakout;
- Kijun pullback / continuation context.

Output: signed `ichimoku_score_v39` plus component flags.

## Engine 4 — Al Brooks-inspired price action

The implementation is an auditable research proxy, not a claim that discretionary Brooks chart reading has been fully automated.

Formalized components:

- Always-In long / short;
- Trend vs Trading Range via path-efficiency state;
- Breakout / failed breakout;
- H1 / H2 and L1 / L2 pullback-attempt proxies;
- three-push Wedge top / bottom;
- Micro Double Top / Bottom;
- Signal Bar quality;
- Follow-Through, known only after the follow-through bar closes;
- Measured Move progress / target context.

Output: signed `brooks_score_v39` plus the full component vector.

## Engine 5 — Multi-Timeframe context

Completed Daily context is attached to 4h observations only after the Daily source candle is closed. Direction is based on completed HTF price/EMA structure and EMA200 slope.

Output: `mtf_score_v39`.

## Fusion rule

There is **no fixed 5-engine AND gate** and no post-result `k-of-n` optimization.

The engine vector is exposed to the frozen v0.39 learning ladder:

1. Ridge;
2. HistGradientBoosting;
3. shallow MLP;
4. PatchTST or CMamba only if incremental OOS evidence justifies higher capacity.

The raw engine mean is retained only as `directional_prior_v39` for research/event orientation. It is not sufficient for capital allocation.

## Neural input contract

- robust rolling median/MAD normalization;
- statistics estimated from observations strictly before the current bar;
- clipping at the frozen v0.39 normalization bound;
- missingness indicators;
- finite `float32` matrix;
- fixed 3-seed median ensemble (`314`, `1618`, `2718`);
- no best-seed cherry-picking.

## Learning target

The learning system is not optimized for raw classification accuracy.

Primary outputs:

- expected **post-cost R**;
- conservative lower-tail / conformal R bound;
- uncertainty width;
- expected holding duration.

A positive point estimate is insufficient. Capital admission requires `lower_expected_r > 0` after the model/calibration layer.

## Capital and money-management policy

Frozen inherited controls:

- base stop-risk per trade: `0.25%` of equity;
- maximum per-trade stop-risk: `0.50%`;
- aggregate open stop-risk cap: `2.00%`;
- same-direction open stop-risk cap: `1.50%`;
- maximum nominal asset weight: `35%`;
- maximum portfolio gross exposure: `70%`;
- drawdown warning 1: `2.0%` → risk scale `0.75`;
- drawdown warning 2: `3.5%` → risk scale `0.50`;
- hard drawdown firewall: `5.0%` → no new risk;
- base round-trip cost assumption: `24 bps`;
- stress round-trip cost: `36 bps`.

Uncertainty shrinks risk monotonically. A non-positive conservative R bound receives zero capital.

## Stop and exit base policy

For research characterization:

- initial stop distance is at least `1.5 × ATR`;
- if a valid structural anchor (swing / OB / supply-demand) requires a wider stop, the wider causal structural distance is used;
- minimum stop fraction: `0.30%` to prevent pathological leverage from tiny stops;
- base target: `3.0R`;
- maximum holding period: `30` 4h bars;
- exits remain subject to a separately auditable Triple-Barrier/structural realization layer.

These settings are frozen for the first v0.39 characterization and are not changed after observing results under the same experiment ID.

## Psychology governance

Psychology is implemented as anti-discretion rules:

- no revenge trading;
- no martingale;
- no averaging down;
- no discretionary override;
- no automatic risk increase after a loss;
- block new entries at three consecutive losses until the defined cooldown/reset policy is satisfied;
- maximum four new entries per day as a generic anti-overtrading ceiling.

The model cannot override these controls.

## Anti-overfit / anti-underfit / local-optimum controls

- purged temporal folds with embargo;
- complexity ladder rather than unrestricted hyperparameter search;
- fixed three-seed median ensemble;
- early stopping, AdamW, weight decay, dropout and gradient clipping for neural candidates;
- higher model capacity must add incremental OOS value over the frozen simpler model;
- venue × quarter × regime robustness diagnostics;
- no threshold relaxation after results;
- irreversible experiment state machine to prevent result-driven retraining loops.

## Development qualification gates

Each of CoinEx, OKX and KuCoin must independently satisfy the inherited development requirements before any holdout authorization:

- selected events/trades `>= 200`;
- profit factor `>= 1.05`;
- mean net expectancy `> 0R`;
- positive-asset breadth `>= 0.60`;
- moving-block CI lower bound `> 0`;
- positive-quarter fraction `>= 0.60`;
- 36 bps stress PF `>= 1.00`;
- at least 60% positive purged folds;
- at least 2/3 positive fixed seeds.

A technical CI pass is not a profitability result. No v0.39 development result authorizes PAPER/LIVE automatically.

## Deferred external intelligence

Whales/Smart Traders, institutional flow, news/sentiment, on-chain and derivatives can be added only after timestamped historical availability and survivorship/leakage controls are proven. They are not allowed to contaminate the first core v0.39 attribution experiment.

## Scientific decision rule

The first v0.39 experiment asks whether the canonical engines, when treated as independent causal information rather than a hard conjunction, support a regularized and economically positive post-cost learning signal that generalizes across consumed development venues and time regimes.

Failure is retained as evidence; it does not authorize silent retuning.
