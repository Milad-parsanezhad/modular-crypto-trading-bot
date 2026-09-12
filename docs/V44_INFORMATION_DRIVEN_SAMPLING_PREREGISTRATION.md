# v0.44 Preregistration — Information-Driven / Intrinsic-Time Sampling

Date frozen: 2026-09-12
Status: **PREREGISTERED / NOT EMPIRICALLY EVALUATED**

## Motivation

v0.43 produced high perturbation stability but failed both target/stop Brier-skill gates and temporal/economic qualification. The next isolated hypothesis is that the 4h clock-time mother-event panel contains too many low-information observations. v0.44 changes the **event-sampling clock only** while preserving the mother strategy, competing-risk target semantics, downstream learner family, cost assumptions, Financial Governor and development/holdout governance as far as practical.

This experiment is motivated by screened evidence on CUSUM/information-driven sampling, Triple-Barrier labeling, Directional Change / Intrinsic Time, and the project's own v0.43 failure attribution.

## Scientific isolation

v0.44 MUST NOT simultaneously add:
- Transformer / PatchTST / CMamba capacity;
- PPO / DQN / other RL policy optimization;
- on-chain features;
- order-book / LOB features;
- sentiment/news inputs;
- outcome-based asset pruning;
- relaxed economic or uncertainty thresholds;
- Kraken data.

True volume bars, dollar bars or tick imbalance bars are **not** claimed from 4h OHLCV. They require lower-level transaction/tick data and belong to a separate experiment if later available.

## Frozen development venues and holdout

Development only:
- CoinEx
- OKX
- KuCoin

Reserved external holdout:
- Kraken — **SEALED / never instantiated in v0.44 development**

PAPER=false. LIVE=false.

## Frozen data-quality contract

Inherit v0.43 outcome-independent data-quality screening:
- timeframe: 4h;
- minimum 900 bars per venue-symbol series;
- missing expected bars <=2%;
- stale close fraction <=5%;
- finite OHLCV, nonnegative volume, valid OHLC geometry;
- at least two quality-passed development venues per asset;
- label horizon must reserve 30 **actual observed future bars**.

Data screening may never use expectancy, PF, returns, labels or model outputs.

## Frozen mother strategy and labels

- source strategy: v0.39 Mother Strategy (ICT/SMC/Ichimoku/Brooks/MTF semantic features);
- source target: v0.41 event-specific discrete-time competing risks (TARGET vs STOP, TIME censored/timeout value);
- target R: 3R;
- maximum hold: 30 bars;
- base round-trip costs: 24 bps;
- stress round-trip costs: 36 bps;
- v0.43 corrected first-match event-family semantics retained.

No label target is re-optimized in v0.44.

## Frozen sampling variants

Exactly three preregistered variants are compared. No additional threshold/variant may be added after empirical results are observed.

### S0 — CLOCK_MOTHER_BASELINE

All causally generated `mother_event_v39 == 1` rows satisfying settled-label and data-quality rules.

This is the reference sampling clock and reproduces the v0.43 event universe semantics before the asset-specific support gate.

### S1 — CUSUM_LAGGED_VOL

A mother event is retained only when the current bar triggers a two-sided CUSUM information event.

Frozen construction:
1. `r_t = log(close_t / close_{t-1})`.
2. Scale is **known before bar t closes**: `scale_t = ATR14_{t-1} / close_{t-1}`.
3. Normalized innovation: `z_t = r_t / max(scale_t, eps)`.
4. Positive/negative cumulative sums reset after an event.
5. Frozen threshold `h = 0.75`, inherited from the historical project CUSUM contract; it is not tuned in v0.44.
6. `CUSUM_EVENT_t = 1` when positive sum >=0.75 or negative sum <=-0.75.
7. Sampling admission is `mother_event_v39_t == 1 AND CUSUM_EVENT_t == 1`.

The lagged scale prevents the current shock from simultaneously defining and testing its own threshold.

### S2 — DIRECTIONAL_CHANGE_LAGGED_ATR

A mother event is retained only when the close path completes a Directional-Change event relative to the running extremum.

Frozen construction:
1. Threshold for bar t is known from t-1: `theta_t = ATR14_{t-1} / close_{t-1}`.
2. `theta_t` is clipped to `[0.002, 0.08]` only for numerical/pathological-volatility safety, reusing project volatility bounds rather than tuning outcome performance.
3. In up mode, track the running high close; a downward change of at least `theta_t` confirms a bearish DC event.
4. In down mode, track the running low close; an upward change of at least `theta_t` confirms a bullish DC event.
5. State changes only after the event is observed at bar close; no future confirmation is used.
6. Sampling admission is `mother_event_v39_t == 1 AND DC_EVENT_t == 1`.

No overshoot-derived future feature is used in the selection decision.

## Learner and perturbation contract

To isolate sampling, use the v0.43 downstream learner unchanged in principle:
- asset-specific HistGradientBoosting cause-specific competing-risk model;
- same asset may learn jointly across CoinEx/OKX/KuCoin only;
- venue context retained;
- three deterministic timestamp-cluster moving-block perturbations;
- perturbation seeds: 314, 1618, 2718;
- target block size: 64 training events;
- row-wise median predictions;
- >=2/3 perturbation agreement for final model admission;
- no best-seed / best-perturbation selection.

If a sampling variant produces insufficient fold-local support for an asset, it fails closed for that asset/fold; no pooled cross-asset rescue is allowed.

## Baseline forecast-skill contract

Every sampling variant must beat the same training-only empirical hazard benchmark used in v0.43:
- target Brier skill median >0;
- stop Brier skill median >0;
- target Brier skill positive in >=60% of folds;
- stop Brier skill positive in >=60% of folds.

Forecast skill is a required gate, not a descriptive metric.

## Purged walk-forward contract

- five chronological purged walk-forward folds;
- 30-bar embargo;
- all training labels must be settled before calibration/test boundaries;
- calibration remains separate from test;
- test outcomes are never used for sampling thresholds, normalization, fit, calibration or asset eligibility.

### Common fold-boundary rule — frozen before empirical evaluation

Sampling variants MUST NOT create their own time folds after filtering. That would confound the sampling ablation by exposing each variant to different test periods.

The exact rule is:
1. Build the fully settled, quality-screened **S0 clock-time mother-event baseline** first.
2. Compute the five purged fit/calibration/test timestamp boundaries **once from S0**, before applying S1 or S2 sampling masks.
3. Store those common boundaries in the prepared-data manifest and hash them.
4. Apply S0/S1/S2 masks **inside the same frozen fit/calibration/test windows**.
5. A sparse variant may have fewer events or unsupported assets inside a fold, but it may not shift, rebuild, merge or skip the common fold boundary to improve support.
6. If a variant lacks the preregistered fold-local support inside a common window, that asset/variant/fold fails closed.

Thus all three variants face the same chronological market regimes and OOS periods.

## Economic gates

Per development venue, retain the frozen qualification gates:
- financially executed trades >=200;
- PF >=1.05;
- expectancy >0;
- positive-asset breadth >=0.60;
- moving-block CI lower bound >0;
- hierarchical asset/time cluster CI lower bound >0;
- stress PF at 36 bps >=1.0;
- positive-quarter fraction >=0.60;
- hard max drawdown <=5% through the Financial Governor.

Additionally:
- positive OOS fold fraction >=0.60;
- perturbation-stable fraction >=0.60 on median-admissible pre-agreement rows.

## Multiplicity / selection rule

Three sampling variants are being compared, so v0.44 may not simply choose the largest backtest statistic.

Frozen rule:
1. Each variant is independently evaluated against **all** forecast, temporal, perturbation, venue and economic gates.
2. A variant that fails any required gate is ineligible regardless of relative rank.
3. If exactly one passes, it is the development winner.
4. If multiple pass, choose the simplest in the preregistered order: `S0 -> S1 -> S2` unless a paired OOS utility comparison demonstrates that the later variant has a positive lower confidence bound over the earlier passing variant. The paired comparison is diagnostic and cannot rescue a gate failure.
5. If none pass, decision is `V44_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE`.

This prevents threshold fishing and multiple-comparison winner selection.

## Workflow/provenance contract

All v0.44 scientific GitHub Actions jobs MUST:
- checkout `${{ github.event.pull_request.head.sha || github.sha }}` explicitly;
- run with `permissions: contents: read`;
- use `concurrency.cancel-in-progress: true`;
- record `git rev-parse HEAD` in `execution_provenance.json`;
- record workflow run ID, event head SHA, Python/package/dependency versions, preregistration SHA-256 and prepared-event-table SHA-256;
- download public development data once in `prepare-v44`;
- fan out five fold jobs from the same prepared checkpoint;
- finalize into one canonical 90-day artifact that **copies every preparation manifest and hash** alongside fold/model/decision evidence.

The workflow must contain an explicit runtime guard preventing Kraken instantiation and tests must assert PAPER=false and LIVE=false.

## Decision discipline

No post-result changes under experiment v0.44 to:
- CUSUM threshold;
- Directional-Change threshold definition;
- event-window tolerance;
- common fold boundaries;
- features;
- labels;
- model capacity;
- costs;
- economic gates;
- asset list based on outcomes.

Any new idea after observing v0.44 results requires a new experiment ID.
