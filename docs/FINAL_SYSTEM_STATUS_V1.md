# Final Engineering Status — v1.0.0-rc1

Date: 2026-09-09

## What is finalized

The thesis repository now has an end-to-end research-to-paper architecture:

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
13. live dashboard;
14. Railway deployment configuration;
15. CI regression and evidence artifacts.

## Scientific state

The latest completed external research gate is v0.12: `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.

No learned model is promoted to live execution. The strongest aggregate historical baseline observed in earlier work was Ichimoku, but it showed regime instability. Therefore v0.14 uses an explicitly labelled **shadow paper hypothesis** rather than describing it as a validated production strategy.

## What "final" means here

Engineering is release-candidate complete for the thesis demonstration. Empirical forward validation is time-dependent and cannot be manufactured retrospectively. v0.13/v0.14 now collect that evidence prospectively.

## Execution ladder

- Backtest: implemented.
- Paper: implemented and deployable.
- Testnet: interface/readiness stage; depends on an exchange-provided test environment/credentials.
- Live: disabled by design.

## Defense evidence

Preserve Git history/PRs, GitHub Actions artifacts, v0.11/v0.12 negative results, v0.13 daily artifacts, Railway health/dashboard screenshots, PostgreSQL paper observations/fills, paper equity/risk reports, and the explicit statement that negative results were not tuned away.

## Remaining empirical gate

A future thesis update may promote a strategy only after a pre-registered forward window demonstrates stable net performance after costs and risk controls. Until then: `LIVE_EXECUTION = FALSE`.
