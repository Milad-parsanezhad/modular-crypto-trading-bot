# v0.15 Forward Evidence & Defense Protocol

Date started: 2026-09-09

## Purpose

v0.15 converts the running Railway forward-paper service into a reproducible evidence stream for the thesis and defense package. It does not change the trading hypothesis and does not authorize real-money execution.

The scientific state entering this phase is intentionally conservative:

- v0.12 completed external derivatives holdout: `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`;
- v0.13 forward microstructure collection: active;
- v0.14 Railway forward paper engine: active;
- strategy label: `HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA`;
- LIVE execution: disabled/fail-closed.

## Production evidence source

The scheduled workflow reads only the public production paper/research endpoints:

- `/health`
- `/research/status`
- `/paper/status`
- `/paper/observations?limit=500`
- `/paper/fills?limit=500`

No exchange secret, API credential, database password, or Railway token is required by the evidence collector.

## Snapshot cadence

GitHub Actions workflow:

`.github/workflows/v15-forward-evidence.yml`

runs every four hours at minute 17 and can also be run manually. Each execution uploads a 90-day retained artifact containing:

- `v15_forward_evidence.json`
- `v15_forward_evidence.md`
- `raw_production_endpoints.json`

## Descriptive metrics captured

Each snapshot records:

- service version and execution mode;
- production safety-contract checks;
- PostgreSQL persistence state;
- cumulative paper return and current drawdown;
- cash/equity/peak equity and open paper positions;
- total observations, fills, and equity points;
- recent action and symbol counts;
- recent mean rule score and spread;
- simulated fill notional and fee drag;
- recent forward observation-window length.

These are operational/forward-paper diagnostics, not proof of alpha.

## Pre-registered minimum sample gate

Before v0.15 may even be described as ready for a formal statistical evaluation, all of the following must be true:

1. at least 168 hours (7 days) of forward observations in the evidence window;
2. at least 100 persisted observations;
3. at least 10 simulated paper fills;
4. all fail-closed production safety checks remain true.

If any condition is missing, the required state is:

`INSUFFICIENT_FORWARD_SAMPLE`

If all conditions are met, the state becomes only:

`READY_FOR_FORMAL_STATISTICAL_EVALUATION`

This is not a promotion decision.

## Live-promotion contract

Throughout v0.15:

`live_promotion = PROHIBITED`

Even after the minimum sample gate passes, a separate pre-registered statistical review must evaluate cost-adjusted performance, drawdown, tail risk, regime stability, and sensitivity before any later execution-readiness discussion.

## Defense evidence policy

The thesis defense package should preserve:

- Git commits and PRs for v0.10-v0.15;
- GitHub Actions run IDs and downloadable artifacts;
- Railway deployment and health evidence;
- PostgreSQL-backed paper observations/fills/equity counts;
- screenshots of the dashboard and API status;
- both positive and negative empirical results;
- explicit separation between engineering success and strategy profitability.

The central claim allowed at this phase is:

> The system has been implemented as a persistent, cost/risk-aware forward-paper trading research service and is prospectively collecting evidence. Profitability and live readiness remain empirical questions, not assumptions.
