# v0.16 Formal Forward Evaluation & Defense Package — Initial Results

Date: 2026-09-09

Workflow run: `34366141078`

Head commit: `e4191284fc85a116c04dc4af7b06bf810b22f739`

Artifact: `v16-formal-forward-defense-4`

Artifact ID: `10109951758`

Artifact digest: `sha256:77cc5ecb4a849eb62196001b9b72d2ddf9586c89f46aab99c61406cfa6a18d84`

## CI result

The hardened v0.16 workflow completed successfully.

Deterministic v0.15 + v0.16 tests:

`7 passed in 0.06s`

The workflow also completed all of the following stages:

- collected immutable v0.15 GitHub Actions artifacts;
- generated the formal forward evaluation;
- generated Chapter 4 forward-results wording;
- generated defense summary and audit CSV;
- generated equity, drawdown and sample-growth SVG charts;
- generated a SHA-256 defense manifest;
- validated fail-closed scientific/safety semantics;
- uploaded the final eight-file v0.16 defense artifact.

## Independence hardening

Three raw v0.15 evidence artifacts were available at this initial run. They were generated within minutes of one another during implementation/smoke verification and contained no independent four-hour market evidence.

v0.16 therefore collapses rapid unchanged/manual/CI snapshots rather than counting them as independent observations. The final hardened run reported:

- raw v0.15 snapshots collected: `3`;
- independent v0.16 snapshots used for the formal time series: `1`;
- minimum independence spacing for an unchanged state: `3.5 hours`.

This prevents manual reruns from artificially increasing sample size or creating meaningless annualized Sharpe/Sortino values.

## Current forward evidence state

The latest production paper evidence remains:

- observations: `2`;
- simulated paper fills: `1`;
- current forward sample gate: `false`;
- formal state: `INSUFFICIENT_FORWARD_SAMPLE`;
- LIVE promotion: `PROHIBITED`.

The latest paper account remains close to the initial 10,000 USDT simulated balance. Earlier v0.15 evidence recorded approximately `-0.0261%` cumulative paper return/drawdown; v0.16 does not reinterpret that tiny sample as evidence of profitability or loss expectancy.

## Scientific interpretation

v0.16 is an evaluation/reporting layer, not a strategy optimizer. It does not change the pre-registered v0.15 minimums:

- 168 forward hours;
- 100 observations;
- 10 simulated paper fills.

Formal Sharpe/Sortino remain withheld until the sample gate is satisfied and at least 20 independent inter-snapshot returns are available. Even after those descriptive risk metrics are unlocked, a separate statistical review is required before any alpha/profitability conclusion.

The following remain hard-coded as false/prohibited in the v0.16 contract:

- `alpha_claim_authorized = false`;
- `profitability_claim_authorized = false`;
- `live_promotion_authorized = false`;
- `live_promotion = PROHIBITED`.

The prior v0.12 result `NO_INCREMENTAL_DERIVATIVES_EVIDENCE` remains part of the thesis evidence and is not overwritten by paper observations.

## Defense package contents

The final artifact contains eight files:

1. `v16_formal_forward_evaluation.json`;
2. `chapter4_forward_results.md`;
3. `defense_summary.md`;
4. `forward_timeline.csv`;
5. `equity_curve.svg`;
6. `drawdown_curve.svg`;
7. `sample_growth.svg`;
8. `defense_manifest.json`.

This package is scheduled to regenerate every four hours as new v0.15 prospective evidence accumulates.
