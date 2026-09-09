# v0.17 Research Gap & Hypothesis Register

Date: 2026-09-09

Status: **Prioritized research hypotheses for future experiments; no signal authorization**

## 1. Purpose

This register converts the current evidence gaps into explicit hypotheses with falsification criteria. It is designed to prevent post-hoc storytelling after results are observed.

## 2. Priority hypotheses

### H1 — Global order flow may contain incremental cross-sectional information that the v0.12 local taker-flow proxy did not capture

**Rationale.** Recent peer-reviewed crypto evidence reports that cross-market/world order flow predicts cryptocurrency returns and can add out-of-sample value, especially in non-linear models. The v0.12 implementation used futures taker-flow proxies from a single archive, which is materially different from a broad multi-currency/world-flow measure.

**Falsifiable test.** Build a point-in-time cross-sectional order-flow panel from an independent source, freeze a cohort and horizon, and compare `price-only` vs `price+global_order_flow` under matched costs and untouched holdout.

**Reject H1 if.** Incremental mean edge is non-positive, the dependence-aware CI includes zero in the wrong direction, SPA/Reality-Check evidence fails after search correction, or net performance collapses after realistic costs.

**Priority.** HIGH, but data acquisition quality is the gating constraint.

### H2 — Cost-aware abstention is likely more valuable than naive directional execution

**Rationale.** Both the project’s v0.10 results and recent 2026 BTC walk-forward evidence show that transaction costs can erase apparently positive forecasting value. A prediction should become a trade only when expected edge exceeds estimated cost plus risk/uncertainty penalties.

**Falsifiable test.** Freeze a forecasting model and compare naive sign trading against a cost-aware threshold/abstention policy on the same untouched window.

**Reject H2 if.** Cost-aware abstention does not improve net return, turnover-adjusted Sharpe, or drawdown relative to the naive rule across the pre-registered test family.

**Priority.** VERY HIGH because it leverages the existing Net-Alpha architecture without adding model complexity.

### H3 — Ichimoku’s apparent edge is conditional on regime rather than universal

**Rationale.** v0.11 showed large positive outcomes in HIGH_VOL/TREND_DOWN and negative outcomes in RANGE/TREND_UP. This creates a real hypothesis but not a valid same-sample regime filter.

**Falsifiable test.** Freeze the regime definition and Ichimoku rule exactly as specified, then test on a new external or prospective domain.

**Reject H3 if.** The pre-specified regime interaction fails to replicate or the regime-conditioned improvement disappears after costs and multiple-testing correction.

**Priority.** HIGH, because the hypothesis comes directly from project evidence, but it must not reuse the v0.11/v0.12 samples for tuning.

### H4 — Better microstructure inputs may matter more than deeper architecture at short horizons

**Rationale.** Recent crypto microstructure work reports stable predictive structure in order-book imbalance, spread and trade features across assets, and other recent studies argue that preprocessing/feature quality can match or exceed extra neural depth. This is consistent with the project’s repeated failure of more complex learned models to beat simple baselines.

**Falsifiable test.** Hold model class fixed and compare input families: candles only, trade imbalance, top-of-book imbalance, depth imbalance, spread/adverse-selection proxies, and combined microstructure.

**Reject H4 if.** Input-family gains fail untouched OOS, are not stable across assets, or disappear under executable top-of-book cost assumptions.

**Priority.** HIGH for research value, MEDIUM for immediate thesis completion because high-frequency data engineering is expensive.

### H5 — RL should target execution/allocation only after a predictive signal survives search-aware validation

**Rationale.** The current project has no learned alpha candidate that survived the existing Great Filter. Using RL to discover both signal and execution policy simultaneously would enlarge the search space and weaken causal interpretation.

**Falsifiable test.** First freeze an externally validated alpha signal. Then compare deterministic sizing/execution with PPO/DDQN execution/allocation agents under multi-seed OOS evaluation and identical market-impact/cost assumptions.

**Reject H5 if.** RL does not produce stable incremental utility after costs across seeds and regimes.

**Priority.** GATED. Do not promote merely to satisfy an “AI complexity” narrative.

## 3. Research priority order

1. **H2 Cost-aware abstention** — highest feasibility and directly supported by current code/results.
2. **H3 External regime-conditioned Ichimoku replication** — direct hypothesis from v0.11; requires genuinely new evidence.
3. **H1 Higher-quality global order flow** — strongest new-data hypothesis, but provider quality is critical.
4. **H4 Rich limit-order-book microstructure** — potentially valuable, engineering-heavy.
5. **H5 RL execution/allocation** — remains gated until a signal survives earlier stages.

## 4. Stop rules

A hypothesis is stopped rather than endlessly tuned when any of the following occurs on the frozen evaluation domain:

- non-positive net edge after costs;
- no improvement over a strong simple baseline;
- dependence-aware CI fails the pre-specified superiority direction;
- FDR / SPA / Reality-Check evidence does not survive the planned search family;
- unacceptable drawdown/capacity deterioration;
- result depends on a single asset, single regime or isolated period outside the claimed scope.

Stopped hypotheses remain in the thesis as negative evidence.

## 5. External research anchors

- Anastasopoulos et al. (2026), *Order flow and cryptocurrency returns*, Journal of Financial Markets 79, 101047. DOI `10.1016/j.finmar.2026.101047`.
- Bysik & Ślepaczuk (2026), *Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting* — recent working-paper evidence emphasizing transaction-cost sensitivity and cost-aware trading filters.
- White (2000), Hansen (2005), Bailey & López de Prado (2014) provide the search-aware statistical framework used to decide whether future apparent edges are likely to survive data snooping and selection bias.

## 6. Current decision

No hypothesis in this register changes the current project status. Through v0.16/v0.17, LIVE promotion remains prohibited and the forward PAPER sample remains insufficient for profitability/alpha claims.
