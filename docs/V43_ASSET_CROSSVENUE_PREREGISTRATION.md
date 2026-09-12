# v0.43 Preregistration — Asset-Specific Cross-Venue Stability

Date frozen: 2026-09-12
Execution state: **PREREGISTERED / NOT YET EMPIRICALLY RUN**

## Motivation

v0.41 produced promising economics but insufficient per-venue sample size. v0.42 expands the development universe and tests pooled breadth. Independent 2026 evidence indicates that cryptocurrency predictive relationships may transfer substantially better across venues for the **same underlying asset** than across different assets. In addition, the current HistGradientBoosting configuration is effectively deterministic across the nominal random seeds, so identical seed outputs must not be interpreted as independent stability evidence.

v0.43 therefore changes **only the learning/stability structure**. It does not change the mother strategy, event semantics, target/stop horizon, costs, risk governor, or development/holdout governance.

## Frozen hypothesis

For a given asset, combining its development observations across CoinEx, OKX and KuCoin while preserving venue context will generalize better than learning a single cross-asset decision surface. Robustness must be demonstrated across deterministic moving-block perturbations of the training sample rather than by choosing a favorable pseudo-random initialization.

## Frozen inputs

- Mother strategy: v0.39 causal ICT + SMC + Ichimoku + Brooks + MTF representation.
- Event families and competing-risk target: unchanged from v0.41/v0.42.
- Timeframe: 4h.
- Development venues: CoinEx, OKX, KuCoin.
- Reserved holdout: Kraken — **SEALED**.
- Base round-trip cost: 24 bps.
- Stress round-trip cost: 36 bps.
- Maximum event horizon: 30 bars.
- Purged walk-forward folds: 5.
- Embargo: 30 bars.
- Learner family: the v0.41 HistGradientBoosting cause-specific learner; no Transformer/RL capacity increase.

## Outcome-independent data-quality screen

A venue/symbol series can enter v0.43 only when all of the following are satisfied before inspecting model outcomes:

1. at least 900 valid closed 4h bars;
2. timestamps are unique and strictly increasing after deterministic sort;
3. all OHLCV fields required by the mother strategy are finite;
4. volume is non-negative;
5. `low <= min(open, close) <= max(open, close) <= high` for every bar;
6. missing expected 4h bars over the observed interval <= 2%;
7. stale-close fraction (`close[t] == close[t-1]`) <= 5%;
8. the final 30 bars cannot generate settled labels and are excluded from labeled training/evaluation events;
9. no observation is selected or rejected using expectancy, PF, future returns, target/stop outcome, model score, or Kraken data.

Every accept/reject decision must be written to a Data Quality Manifest with a reason code.

## Asset-specific training rule

- One model family is trained **per asset** using that asset's observations from CoinEx, OKX and KuCoin only.
- Venue is represented as a categorical/context feature; no other asset's observations may enter that asset's model.
- A fold-specific asset model is admissible only if its training partition has:
  - >=600 settled events in total; and
  - >=100 settled events from at least two development venues.
- If this availability condition fails, that asset/fold is marked `INSUFFICIENT_TRAINING_SUPPORT`; there is no outcome-based fallback and no pooled cross-asset rescue.

## True perturbation stability

Nominal estimator seeds are not used as independent evidence. For each asset/fold, construct exactly three moving-block training perturbations using frozen seeds:

- 314
- 1618
- 2718

Frozen block length: **64 consecutive training events**, sampled with replacement until the original training-row count is reached. Blocks preserve local temporal dependence; test and calibration rows are never resampled into training.

For each perturbation:
- fit the identical HistGradientBoosting cause-specific model;
- calibrate only on the fold's untouched calibration partition;
- obtain the same competing-risk and lower-Expected-R outputs.

No best perturbation may be selected. The final forecast is the row-wise median across the three perturbation forecasts. A trade is perturbation-stable only when at least **2 of 3** perturbations independently satisfy the frozen v0.41 admission logic for that event.

## Forecast-skill benchmark

A model is not credited merely for producing profitable selected trades. Each fold must also be compared with a training-only empirical baseline stratified by:

`event_family × side × regime`

with backoff to `side × regime`, then global training hazard when a cell is sparse.

Metrics:
- target Brier score;
- stop Brier score;
- Brier Skill Score = `1 - model_brier / baseline_brier`.

Frozen skill gate:
- target Brier skill > 0 in at least 3/5 folds; and
- stop Brier skill > 0 in at least 3/5 folds; and
- median target and stop Brier skill across the five folds > 0.

This benchmark uses no test outcomes for fitting or cell construction.

## Economic/robustness gates

The existing development gates remain unchanged and apply after non-overlap realization and the Financial Governor:

- >=200 executed trades per development venue;
- PF >=1.05 per venue;
- expectancy >0 per venue;
- positive asset breadth >=0.60;
- moving-block CI low >0;
- hierarchical symbol × block CI low >0;
- stress PF at 36 bps >=1.0;
- positive-quarter fraction >=0.60;
- hard account drawdown <=5%;
- positive fold fraction >=0.60.

Additional v0.43 stability gate:
- >=60% of financially executed events must have perturbation agreement >=2/3.

All gates must pass before a separate Kraken holdout preregistration is authorized.

## Anti-overfit prohibitions

Under experiment ID v0.43 it is prohibited to:

- lower any admission or economic threshold after observing results;
- change the 64-event perturbation block length after observing results;
- remove weak assets, venues, quarters or event families because of performance;
- tune event-family definitions from test outcomes;
- add Transformer, RL, on-chain, sentiment or LOB features;
- inspect Kraken;
- select the best perturbation/seed/model run.

Any such change requires a new experiment ID and new preregistration.

## Interpretation

Passing v0.43 would establish development evidence for asset-specific cross-venue generalization under true training-data perturbations. It would **not** establish live profitability. Kraken would then be eligible only for a separately frozen, one-shot external holdout test.
