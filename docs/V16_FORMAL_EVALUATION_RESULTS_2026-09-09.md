# v0.16 Formal Forward Evaluation — Official Results Record

Date: 2026-09-09

Status: **Engineering complete; formal scientific sample not yet sufficient**

## 1. Purpose

v0.16 is the formal reporting and defense-evidence layer built on top of the prospective v0.15 forward-paper evidence stream. It is intentionally not a strategy optimizer and it does not modify any earlier pre-registered gate.

Its goals are to:

- aggregate immutable v0.15 evidence artifacts;
- remove rapid duplicate/manual snapshots from the independent time series;
- produce a reproducible forward-paper evaluation record;
- generate Chapter 4 wording and defense artifacts automatically;
- keep all profitability, alpha and LIVE-readiness claims fail-closed until the sample protocol is satisfied.

## 2. Reproducibility anchor

Validated workflow run: `34366141078`

Validated head commit: `e4191284fc85a116c04dc4af7b06bf810b22f739`

Final artifact: `v16-formal-forward-defense-4`

Artifact ID: `10109951758`

Artifact SHA-256: `77cc5ecb4a849eb62196001b9b72d2ddf9586c89f46aab99c61406cfa6a18d84`

CI result: **SUCCESS**

Deterministic v0.15 + v0.16 tests: `7 passed in 0.06s`

## 3. Data lineage

The evidence chain is:

`Railway PAPER runtime -> PostgreSQL -> v0.15 scheduled evidence snapshot -> GitHub Actions artifact -> v0.16 independent-snapshot filter -> formal evaluation -> defense package`

No real-money fills are represented in this chain.

The v0.16 pipeline reads previously generated v0.15 artifacts rather than reconstructing historical values from a mutable database state. This preserves an auditable prospective evidence trail.

## 4. Independence control

At the validated run, three raw v0.15 artifacts were available. They were generated within minutes during implementation and smoke verification and therefore did not represent three independent market observations.

The hardened v0.16 filter retained only **1 independent snapshot** for the formal time series.

Minimum spacing applied to unchanged states: `3.5 hours`.

This prevents repeated manual/CI reruns from inflating sample size or manufacturing annualized Sharpe/Sortino estimates from duplicated state.

## 5. Current formal forward state

Current production-paper evidence:

- observations: `2`;
- simulated paper fills: `1`;
- independent v0.16 snapshots: `1`;
- pre-registered sample gate passed: `false`;
- formal state: `INSUFFICIENT_FORWARD_SAMPLE`;
- LIVE promotion: `PROHIBITED`.

The paper account remains close to the initial simulated balance of 10,000 USDT. Earlier v0.15 snapshots showed approximately `-0.0261%` cumulative paper return/current drawdown. This value is descriptive only and is not interpreted as evidence of positive or negative expected performance.

## 6. Pre-registered forward gate

v0.16 preserves the v0.15 minimums unchanged:

- minimum forward duration: `168 hours`;
- minimum observations: `100`;
- minimum simulated fills: `10`.

All three conditions must pass before the formal sample-size gate can open.

Additionally, Sharpe and Sortino remain withheld until there are at least `20` independent inter-snapshot returns. Passing that numerical-depth condition still does **not** establish alpha or profitability; it only makes formal descriptive risk-ratio review numerically admissible.

## 7. Metric policy

Current policy state:

- `descriptive_metrics_authorized = true` while the safety contract remains intact;
- `risk_ratios_authorized = false`;
- `alpha_claim_authorized = false`;
- `profitability_claim_authorized = false`;
- `live_promotion_authorized = false`;
- `live_promotion = PROHIBITED`.

Therefore, the following are currently allowed in the thesis/defense:

- engineering demonstration evidence;
- proof of persistent forward-paper operation;
- sample counts;
- simulated account equity and drawdown snapshots;
- audit trail and reproducibility evidence;
- current gate status.

The following are not yet allowed:

- claiming the strategy is profitable;
- claiming statistically significant alpha;
- claiming stable Sharpe/Sortino;
- claiming readiness for real-money LIVE trading.

## 8. Relationship to earlier research results

v0.16 does not overwrite the prior negative research evidence.

In particular, the v0.12 decision remains:

`NO_INCREMENTAL_DERIVATIVES_EVIDENCE`

That result remains part of the thesis evidence base. Forward-paper observations are a new prospective validation layer, not a justification for deleting or tuning away prior negative results.

## 9. Generated defense outputs

The validated v0.16 artifact contains:

1. `v16_formal_forward_evaluation.json`
2. `chapter4_forward_results.md`
3. `defense_summary.md`
4. `forward_timeline.csv`
5. `equity_curve.svg`
6. `drawdown_curve.svg`
7. `sample_growth.svg`
8. `defense_manifest.json`

The manifest provides SHA-256 integrity hashes for the defense package files.

## 10. Thesis interpretation

The correct formal conclusion at this stage is:

> The implemented trading-research system is operational in forward PAPER mode and produces persistent, reproducible and auditable evidence. However, the prospective sample remains below the pre-registered minimum. Therefore, the current evidence is sufficient to demonstrate engineering functionality and research-process integrity, but insufficient to support a claim of trading profitability, stable risk-adjusted performance, alpha, or readiness for real-money execution.

This is the official v0.16 interpretation and should be used consistently in Chapter 4, the defense presentation and oral responses.

## 11. Next transition condition

v0.16 remains in continuous evidence-collection mode. The next research state is entered only when:

1. forward duration >= 168 hours;
2. observations >= 100;
3. simulated fills >= 10;
4. safety contract remains intact;
5. enough independent snapshot returns exist for meaningful risk-ratio review.

Until then, the system remains **PAPER only** and LIVE promotion stays **PROHIBITED**.
