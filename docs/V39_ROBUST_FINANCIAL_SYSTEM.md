# v0.39 — Robust Financial Learning System

Status: **research-only / preregistration architecture**  
Base lineage: `v0.39R` canonical mother-strategy reconstruction  
Development venues: CoinEx, OKX, KuCoin  
Reserved holdout: **Kraken SEALED**  
PAPER: disabled  
LIVE: disabled

## Objective

v0.39 is designed around four failure modes that must be controlled explicitly rather than patched after a backtest:

1. **overfitting** — memorizing historical noise, venue quirks, quarters or regimes;
2. **underfitting** — using a model too weak to represent reproducible nonlinear interactions;
3. **local-optimum dependence** — relying on one lucky neural-network initialization;
4. **research loops / post-result tuning** — repeatedly changing thresholds after observing results.

The system therefore separates market knowledge, representation learning, uncertainty, validation and capital allocation.

## Architecture

```text
Canonical causal engines
  ├─ Ichimoku regime
  ├─ ICT/SMC structure
  ├─ Brooks engine (independent)
  ├─ Multi-timeframe context
  └─ derivatives/external features when historically point-in-time valid
          │
          ▼
Causal feature layer
          │
          ▼
Robust causal normalization
  ├─ prior-window median
  ├─ prior-window MAD / fallback std
  ├─ clipping
  └─ missingness indicators
          │
          ▼
Complexity ladder
  Ridge → HistGB → shallow MLP → PatchTST/CMamba
          │
          ▼
Fixed three-seed median ensemble
          │
          ▼
Expected Net-R + lower-tail estimate + duration
          │
          ▼
Calibration / conformal uncertainty
          │
          ▼
Purged temporal validation + embargo
          │
          ▼
Worst-group robustness
  venue × quarter × regime
          │
          ▼
Financial risk allocator
  ├─ lower-bound edge required
  ├─ uncertainty-scaled risk
  ├─ aggregate open risk ≤ 2.0%
  ├─ same-direction open risk ≤ 1.5%
  ├─ max asset nominal weight ≤ 35%
  ├─ max portfolio gross ≤ 70%
  └─ hard drawdown firewall = 5%
```

## 1. Normalization

Global full-sample standardization is prohibited because it leaks future distribution information into earlier observations.

`research_bot.financial_system_v39.causal_robust_normalize` uses, for every symbol and feature:

- rolling location = median of observations strictly before bar `t`;
- rolling scale = 1.4826 × MAD of observations strictly before `t`;
- fallback = prior rolling standard deviation when MAD collapses;
- robust z-score clipped to `[-5, +5]`;
- optional missingness indicator;
- missing normalized values filled with zero only after the indicator is emitted;
- output converted to finite `float32` for neural-network input.

The default normalization memory is 180 observations with at least 60 historical observations. For 4-hour data this is a local, moving distribution estimate rather than a fixed full-history scaler.

For sequence models, RevIN may additionally be used inside the network. It is complementary to the causal feature normalizer: the first protects feature construction from leakage and extreme scaling; RevIN helps the sequence model cope with instance-level non-stationarity.

## 2. Complexity ladder: anti-overfit and anti-underfit at the same time

We do **not** start with the most powerful network.

Frozen capacity order:

1. Ridge regression;
2. HistGradientBoosting;
3. shallow MLP;
4. PatchTST or CMamba sequence model.

A higher-capacity model is allowed to replace a lower-capacity model only when it provides incremental **out-of-sample** evidence while still satisfying all robustness gates. This creates a falsifiable response to both errors:

- if a simple model is sufficient, extra neural capacity is rejected as unnecessary variance;
- if the simple model demonstrably underfits and the neural model improves robust OOS evidence, the added capacity is justified.

No architecture is promoted because of training loss alone.

## 3. Neural-network regularization policy

The frozen neural optimizer/regularization defaults are:

- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- dropout: `0.15`;
- gradient clipping: `1.0`;
- maximum epochs: `80`;
- early-stopping patience: `8`;
- warm-up: `5%` of optimization steps;
- maximum one restart per seed;
- fixed seeds: `314`, `1618`, `2718`.

The three seeds are not a hyperparameter search. Predictions are combined by the **median**. The best seed may never be selected after the fact.

This policy reduces dependence on a lucky local optimum and makes initialization sensitivity measurable.

## 4. Learning target

The main financial target is not raw direction accuracy. The primary target remains **post-cost R-multiple**.

The neural output contract should support at least:

- expected post-cost `R`;
- lower-tail / quantile estimate of `R`;
- expected holding duration;
- optional direction probability for calibration diagnostics.

Recommended multi-task objective:

```text
L = Huber(expected_R, realized_R)
  + 0.25 × Pinball(q20_R, realized_R)
  + 0.15 × Huber(log_duration, realized_log_duration)
```

This keeps the learning objective economically aligned with the prior v0.36 Expected-Net-R lineage rather than optimizing classification accuracy disconnected from P/L.

## 5. Purging, embargo and backtest-overfit control

Temporal validation must be leakage-safe.

Minimum requirements:

- at least five purged temporal folds;
- 30-bar embargo around fold boundaries;
- event labels must be fully settled before entering a training set;
- validation and test statistics are never used for parameter fitting;
- at least 60% of purged folds must have positive OOS expectancy;
- all pre-existing venue-level economic gates remain unchanged.

For model-comparison work, CPCV / PBO / Deflated-Sharpe diagnostics should be reported where the sample geometry permits. These diagnostics are supplementary to, not replacements for, the already frozen development gates.

## 6. Distribution shift and Group-DRO logic

Average performance can hide a model that only works in one exchange, one quarter or one regime. v0.39 therefore treats the following as robustness groups:

- venue;
- calendar quarter;
- detected market regime.

Training may use Group-DRO / worst-group weighting, but development qualification is simpler and harder to game: the model must pass the frozen metrics at each development venue and must show temporal persistence.

A Group-DRO objective is a training regularizer; it does not relax the qualification gate.

## 7. Financial capital system

### Trade admission

A trade may receive risk only when its calibrated/lower-bound expected net R is positive. A positive point estimate with a non-positive lower bound receives zero risk.

### Per-trade risk

Base risk at stop:

`0.25% of realized equity`

Maximum per-trade stop risk:

`0.50%`

Uncertainty monotonically reduces this budget. It never increases risk above the frozen maximum.

### Aggregate open risk

Maximum loss-at-stop across all simultaneously open positions:

`2.0% of realized equity`

### Same-direction concentration

Maximum aggregate stop risk for all longs together or all shorts together:

`1.5% of realized equity`

### Asset and gross exposure

- maximum nominal weight per asset: `35%`;
- maximum portfolio gross exposure: `70%`.

### Drawdown scaling

From realized equity peak:

- drawdown `< 2.0%`: risk scale `1.00`;
- drawdown `2.0–3.5%`: risk scale `0.75`;
- drawdown `3.5–5.0%`: risk scale `0.50`;
- drawdown `>= 5.0%`: **no new risk**.

The pre-entry firewall must also reserve sufficient loss-at-stop headroom so that already-open risk plus proposed risk cannot cross the 5% floor at admission time.

### Transaction costs

Base round-trip cost remains `24 bps`; stress remains `36 bps` unless a later preregistration changes market/exchange assumptions before observing results.

## 8. Loop prevention

v0.39 has an irreversible experiment state machine:

```text
CREATED
  → TRAINED
  → CALIBRATED
  → VALIDATED
  → FROZEN
  → DEVELOPMENT_TESTED
```

Any stage can be rejected. A `DEVELOPMENT_TESTED` or `REJECTED` experiment cannot return to training under the same experiment identifier.

Consequences:

- no lowering a threshold after seeing a result;
- no adding a feature because the latest run failed;
- no selecting the best random seed;
- no repeated retries until a profitable result appears;
- any new hypothesis receives a new experiment ID and preregistration.

This is the main protection against a human/automation optimization loop.

## 9. Local-optimum control

Local optima cannot be eliminated for non-convex neural networks, but dependence on them can be reduced and audited:

- three fixed initializations;
- median prediction ensemble;
- AdamW + warm-up + clipping;
- early stopping on purged validation only;
- seed stability reported explicitly;
- no best-seed cherry-picking;
- if fewer than two of three seeds have positive OOS expectancy, development qualification fails.

## 10. Frozen development gates

All three development venues must independently satisfy:

- selected events `>= 200`;
- PF `>= 1.05`;
- mean net expectancy `> 0R`;
- positive-asset breadth `>= 0.60`;
- moving-block CI lower bound `> 0`;
- positive-quarter fraction `>= 0.60`;
- 36 bps stress PF `>= 1.0`.

In addition:

- positive purged-fold fraction `>= 0.60`;
- positive seed fraction `>= 2/3`.

Failure of any required gate means development rejection. No gate may be weakened after results.

## 11. What remains intentionally sealed

Kraken is not a development data source, validation source, normalization-fit source or model-selection source. It stays untouched until a frozen development candidate passes every required gate and a separate holdout preregistration authorizes one final evaluation.

## 12. Interpretation

This design does not claim that overfitting, underfitting or local optima are impossible. It turns them into measurable failure modes with explicit controls and makes post-result optimization loops structurally illegal under the same experiment identity.

The objective is not the model with the highest backtest return. The objective is the **simplest model that demonstrates persistent post-cost economic information across time, venues, regimes, random initializations and cost stress while respecting a hard capital-loss budget**.
