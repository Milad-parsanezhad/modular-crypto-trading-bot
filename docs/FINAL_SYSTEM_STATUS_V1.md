# Final Engineering Status — v1.0.0-rc1

Date: 2026-09-09

## What is finalized

The thesis repository now has an end-to-end research-to-production-paper architecture:

1. dynamic universe discovery and eligibility;
2. point-in-time evidence contracts;
3. leakage-safe technical/Ichimoku features;
4. cost-aware ML tournaments;
5. external-holdout derivatives ablation;
6. moving-block bootstrap and FDR;
7. independent risk engine with drawdown/CVaR/liquidity gates;
8. guarded paper execution;
9. PostgreSQL forward-paper persistence and restart deduplication;
10. CoinEx BTC/ETH forward paper observer;
11. post-v0.12 forward microstructure collection;
12. FastAPI research/paper service;
13. live research dashboard;
14. Railway production deployment;
15. CI regression/evidence artifacts;
16. independent production smoke validation from GitHub Actions.

## Production verification

Railway successfully deployed the release candidate from repository commit:

`1ddeaab39f2b66b8b3c1d0f4c5c8f5cbea8a3853`

Deployment:

`a74994ff-b1f0-4cde-b2ec-6316c781a2cd`

The Railway health check returned HTTP 200 and the independent GitHub Actions production smoke run `34360057064` subsequently verified:

- API version `1.0.0-rc1`;
- forward paper status `RUNNING`;
- paper execution enabled;
- LIVE execution false;
- PostgreSQL persistence active;
- two persisted observations;
- one simulated paper fill;
- two equity records;
- no forward-cycle runtime error;
- already-observed 4h bars were deduplicated rather than traded twice.

Detailed evidence is recorded in `docs/V1_PRODUCTION_SMOKE_RESULTS_2026-09-09.md`.

## Scientific state

The latest completed external research gate remains v0.12: `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.

No learned model is promoted to live execution. The strongest aggregate historical baseline observed in earlier work was Ichimoku, but it showed regime instability. Therefore v0.14 uses an explicitly labelled **shadow paper hypothesis** rather than describing it as a validated production strategy.

The paper account has already produced its first simulated fill in the production forward observer. That proves the runtime/persistence pipeline works; it does **not** prove profitability or alpha.

## What "final" means here

Engineering is release-candidate complete for the thesis demonstration and the paper system is operational on Railway. Empirical forward validation is time-dependent and cannot be manufactured retrospectively. v0.13/v0.14 now collect that evidence prospectively.

## Execution ladder

- Backtest: implemented and historically evaluated.
- Paper: **implemented, deployed, PostgreSQL-backed and production-smoke verified**.
- Testnet: interface/readiness stage; depends on an exchange-provided test environment/credentials and remains gated by forward evidence/reconciliation requirements.
- Live: **disabled by design / fail-closed**.

## Defense evidence

Preserve Git history/PRs, GitHub Actions artifacts, v0.11/v0.12 negative results, v0.13 forward artifacts, Railway deployment/health evidence, production smoke run `34360057064`, PostgreSQL paper observations/fills/equity, dashboard screenshots, paper risk reports, and the explicit statement that negative results were not tuned away.

## Remaining empirical gate

A future thesis update may promote a strategy only after a pre-registered forward window demonstrates stable net performance after costs and risk controls. Until then:

`LIVE_EXECUTION = FALSE`
