# v0.44 Preregistration — Causal Information-Driven Sampling Ablation

Status: PREREGISTRATION ONLY. Empirical execution is forbidden until a frozen v0.43 result document exists.

## Motivation

v0.42 showed that broad cross-asset pooling did not solve temporal persistence. v0.43 tests same-asset cross-venue learning with stricter data-quality and robustness controls. The next isolated hypothesis tests whether *when* examples are sampled is itself a bottleneck.

Recent evidence motivating this experiment includes: (i) information-driven/CUSUM sampling with Triple-Barrier labels in crypto research; (ii) doctoral work on Directional Change / intrinsic time; (iii) leakage-aware theses showing that model complexity often does not survive honest walk-forward testing; and (iv) transaction-cost-aware studies showing that prediction-to-trade conversion is often more important than neural capacity.

This experiment deliberately does **not** add a Transformer, RL policy, on-chain data, sentiment, LOB data, new risk limits, or new economic thresholds.

## Data-resolution constraint

The current canonical development dataset is 4h OHLCV. Tick-level dollar bars, volume bars and order-flow bars cannot be faithfully reconstructed from 4h aggregates. v0.44 therefore does **not** claim to implement tick-level information-driven bars. The only novel sampling mechanism is a causal volatility-adaptive symmetric CUSUM filter computed from 4h closes.

Directional Change / intrinsic-time sampling remains a separate future experiment because adding both CUSUM and DC in one stage would create another model-selection problem.

## Frozen development universe

- Development venues: CoinEx, OKX, KuCoin only.
- Reserved external holdout: Kraken, SEALED.
- Symbols/data-quality universe: inherited from the frozen v0.43 data-quality policy; no symbol may be added/removed using returns, PF, expectancy, model score or future labels.
- Timeframe: 4h.
- Costs: inherited unchanged from v0.39-v0.43 (base round-trip 24 bps, stress round-trip 36 bps).
- Target/stop semantics: inherited frozen competing-risk / Triple-Barrier semantics (target +3R, stop -1R, maximum hold 30 bars).
- PAPER=false; LIVE=false.

## Experimental comparison

Exactly two sampling arms are allowed:

1. `V44_CONTROL_V43_EVENTS`
   - Frozen v0.43 event construction and model/economic pipeline.
   - Recomputed only as a contemporaneous control on the same frozen data snapshot.

2. `V44_CUSUM_ACTIVITY_EVENTS`
   - Causal symmetric CUSUM activity sampling.
   - No threshold search; no alternative CUSUM multiplier may be tried under this experiment ID.

The novel arm is promoted only if it independently passes every frozen gate. A relative improvement over the control is descriptive unless both arms are compared with a preregistered paired fold statistic.

## Frozen CUSUM definition

For each venue-symbol series sorted by timestamp:

- `r_t = log(close_t / close_{t-1})`.
- Volatility scale uses only prior information: exponentially weighted standard deviation of log returns with span=48 bars and minimum 48 observations, shifted by one bar.
- Threshold: `h_t = max(1e-8, 1.0 * sigma_{t-1})`.
- Positive accumulator: `S+_t = max(0, S+_{t-1} + r_t)`.
- Negative accumulator: `S-_t = min(0, S-_{t-1} + r_t)`.
- Trigger when `S+_t >= h_t` or `S-_t <= -h_t`; after any trigger both accumulators reset to zero.
- The CUSUM trigger is an **activity clock only**, not a directional trading signal.
- A labeled candidate is admissible at trigger `t` only when the frozen mother-strategy candidate side at `t` is +1 or -1. Direction comes from the mother strategy, never from the sign of CUSUM.
- Entry remains next-bar open (`t+1`), so the current close used by CUSUM is known before entry.
- Final 30 actual observed bars are reserved before label generation.

## Model held fixed

The primary learner is held fixed to the final v0.43 asset-specific cross-venue cause-specific HistGradientBoosting design, including:

- same-asset learning across development venues;
- independent calibration split;
- competing-risk TARGET vs STOP hazards;
- conformal Expected-Net-R lower bound;
- timestamp-cluster non-circular training perturbations;
- row-wise median / >=2-of-3 perturbation agreement;
- training-only naive Brier benchmark;
- Financial Governor unchanged.

If v0.43 is invalidated technically, v0.44 may not run until the inherited implementation is repaired and re-frozen.

## Validation and robustness

The following remain mandatory:

- purged chronological walk-forward folds + embargo;
- no fitting or normalization on future/test data;
- target and stop Brier skill vs training-only naive baseline;
- >=60% positive OOS folds;
- non-tautological perturbation stability gate measured before agreement filtering;
- venue-level minimum selected/executed sample gate;
- PF >=1.05;
- expectancy >0;
- positive-asset breadth >=0.60;
- moving-block CI lower bound >0;
- symbol/time cluster CI lower bound >0;
- stress PF >=1.0 at 36 bps;
- positive-quarter fraction >=0.60;
- DD firewall unchanged.

## Multiple-testing / robustness discipline

No CUSUM threshold, EWMA span, warm-up, target, stop, holding horizon, transaction cost, feature list, model architecture or risk limit may be altered after outcome inspection under `v0.44`.

Parameter-stability and robustness diagnostics may perturb the frozen CUSUM multiplier in a **diagnostic-only** neighborhood (0.8, 1.0, 1.2) after the primary decision, but those diagnostics may not rescue a failed primary candidate or become a new winner. Any promotion claim must come from the frozen 1.0 multiplier only.

The number of research variants tried across v0.39-v0.44 must be reported in the thesis. Deflated-Sharpe / PBO-style multiple-testing diagnostics are recommended as secondary evidence, never as substitutes for the frozen economic gates.

## Execution guard

An empirical v0.44 runner MUST fail closed unless all of the following exist:

1. `docs/V43_RESULTS_2026-09-12.md` with a frozen v0.43 decision;
2. the exact v0.43 tested commit and artifact digest are recorded;
3. Kraken remains untouched;
4. no post-result threshold relaxation was committed to the v0.43 lineage.

Until those conditions hold, v0.44 is documentation + unit-tested causal sampling primitives only.
