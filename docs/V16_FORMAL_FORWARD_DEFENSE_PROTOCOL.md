# v0.16 Formal Forward Evaluation & Defense Package Protocol

Date: 2026-09-09

## Purpose

v0.16 converts the prospective v0.15 evidence stream into a reproducible thesis/defense package without changing the pre-registered scientific gate and without retroactively tuning the paper strategy.

The v0.15 minimums remain unchanged:

- minimum forward window: 168 hours;
- minimum observations: 100;
- minimum simulated paper fills: 10.

Passing those minimums means only that a formal forward review may begin. It does not, by itself, establish alpha, profitability, or live readiness.

## Inputs

v0.16 reads immutable GitHub Actions artifacts produced by `.github/workflows/v15-forward-evidence.yml`.

Every v0.15 artifact contributes a normalized evidence snapshot containing:

- production PAPER safety contract;
- PostgreSQL-backed observation/fill/equity counts;
- current paper equity and drawdown;
- sample-gate state;
- explicit `live_promotion = PROHIBITED` semantics.

The v0.16 collector downloads prior non-expired `v15-forward-evidence-*` artifacts through the GitHub Actions API and deduplicates them by `captured_at`.

## Evaluation policy

v0.16 separates three questions:

1. **Engineering demonstration readiness** — is the deployed PAPER runtime operating with reproducible evidence and fail-closed LIVE semantics?
2. **Forward sample sufficiency** — have the pre-registered v0.15 minimums been reached?
3. **Statistical review readiness** — after the sample gate passes, are enough independent snapshot returns available to calculate descriptive risk ratios without presenting numerically meaningless values?

Formal Sharpe/Sortino are withheld until the v0.15 sample gate has passed and at least 20 inter-snapshot returns are available.

Even after those ratios become available:

- `alpha_claim_authorized = false`;
- `profitability_claim_authorized = false`;
- `live_promotion_authorized = false`;
- `live_promotion = PROHIBITED`.

A separate post-gate statistical review is required before any scientific promotion decision.

## Generated defense package

Each successful v0.16 run generates:

1. `v16_formal_forward_evaluation.json` — machine-readable formal evaluation;
2. `chapter4_forward_results.md` — thesis Chapter 4 forward-results section;
3. `defense_summary.md` — concise defense wording and current evidence status;
4. `forward_timeline.csv` — snapshot-by-snapshot audit trail;
5. `equity_curve.svg` — prospective paper-equity evidence chart;
6. `drawdown_curve.svg` — prospective paper-drawdown chart;
7. `sample_growth.svg` — growth of forward observations;
8. `defense_manifest.json` — file sizes and SHA-256 digests.

The package is regenerated every four hours and retained as a GitHub Actions artifact for 90 days.

## Interpretation rules for the thesis

The thesis may report immediately that:

- the production PAPER service is operational;
- evidence is persisted in PostgreSQL;
- scheduled prospective snapshots are collected independently by GitHub Actions;
- the defense package is reproducible from immutable artifacts;
- safety and fail-closed execution semantics are continuously checked.

The thesis must not convert an insufficient forward sample into a claim of positive alpha, profitability, stable Sharpe, or real-money readiness.

The v0.12 result `NO_INCREMENTAL_DERIVATIVES_EVIDENCE` remains part of the evidence record and is not overwritten by later paper observations.
