# Agent Operating Contract — Repository Routing and Source of Truth

> **Effective date:** 2026-09-13  
> **Project:** MSc AI cryptocurrency trading-bot research  
> **Primary repository:** `parsa314/modular-crypto-trading-bot`

This file exists specifically to prevent future coding/research agents from entering repository-selection loops, duplicating fixes, or treating the public deployment shell as the scientific codebase.

## 1. Single Source of Truth

`parsa314/modular-crypto-trading-bot` is the **canonical scientific and engineering repository** for this project.

All substantive development belongs here, including:

- research code and experiment logic;
- strategy research (Ichimoku, market structure, ICT/SMC-derived causal hypotheses, etc.);
- feature engineering and model research;
- backtesting, OOS validation and statistical testing;
- risk, portfolio and CVaR research;
- ML / deep-learning / reinforcement-learning modules;
- paper-trading research logic when scientifically authorized;
- thesis evidence, reproducibility assets and defense documentation;
- the authoritative project dependency definition (`pyproject.toml`).

**Do not create a second scientific implementation in another repository unless the owner explicitly asks for it.**

## 2. Relationship to `parsa314/miladchicomobot`

`parsa314/miladchicomobot` is a **lightweight public deployment / monitoring shell**, not the canonical research codebase.

Its purpose is to expose a small fail-closed FastAPI service for deployment verification and public monitoring. It may intentionally contain only a subset of dependencies and endpoints.

Therefore:

- do **not** treat missing scientific modules in `miladchicomobot` as evidence that the main research code was deleted;
- do **not** port the entire scientific project into `miladchicomobot` merely because Railway currently deploys that repository;
- do **not** use `miladchicomobot` as the authoritative dependency manifest for the thesis project;
- do **not** infer the state of the scientific project from the deployment shell alone;
- when a runtime dependency is mirrored into `miladchicomobot`, it is only to keep the public shell deployable.

## 3. Current Railway Routing

As of 2026-09-13, the Railway service named `thesis-trading-bot-v08` is sourced from:

`parsa314/miladchicomobot` → branch `main`

This is a deployment-routing fact, **not** a statement that `miladchicomobot` is the main project repository.

Before changing Railway source routing, verify the current Railway service configuration and obtain explicit approval for any migration/repointing action.

## 4. Intentional Fail-Closed Behavior

The public deployment shell is intentionally research-only and fail-closed.

The following state is intentional unless a later scientific gate explicitly changes it:

- `LIVE_EXECUTION = false`
- `PAPER_EXECUTION = false`
- forward-paper execution disabled
- no exchange-order endpoint enabled

A missing endpoint such as `/paper/observations` in the public shell is **not automatically a bug**. The deployed shell has historically exposed `/paper/status` while keeping execution endpoints locked. Inspect the actual source and scientific gate before attempting to "fix" such behavior.

Do not enable live or paper execution merely to make a smoke test pass.

## 5. Dependency Authority

For the scientific bot, use the dependency declarations in this repository as authoritative. At the time this contract was written, the core runtime includes the project stack around:

- CCXT
- NumPy
- Pandas
- scikit-learn
- FastAPI
- Uvicorn
- Psycopg

Do not add large ML frameworks to a deployment image simply because they may be used in future research. Add PyTorch / Transformers / RL frameworks only when the corresponding code path is actually being integrated and tested.

## 6. Infrastructure Errors Are Not Automatically Code Errors

On 2026-09-13, a Railway PostgreSQL instance entered repeated recovery because its storage was exhausted (`No space left on device`). A read-only verification service then failed with PostgreSQL `57P03` (`database system is in recovery mode`).

If similar failures appear, check infrastructure/storage/database health before changing trading or research logic.

Neon was independently reachable and contained the Crypto Intelligence data model at the time of that investigation, but agents must re-verify current database routing before making migrations or changing `DATABASE_URL`.

## 7. Agent Decision Procedure

When continuing this project:

1. Start in `parsa314/modular-crypto-trading-bot` for scientific/research/trading-bot work.
2. Read `README.md`, this `AGENTS.md`, the latest project-status document, and the relevant experiment/workflow before editing code.
3. Verify the latest commit and CI state; do not assume an older chat-reported version is current.
4. Treat `miladchicomobot` as a deployment shell unless the owner explicitly changes repository governance.
5. For deployment failures, separate **code**, **dependency**, **database**, **Railway**, and **external-provider** causes before patching.
6. Preserve `LIVE_EXECUTION=false` unless explicit authorization and the project governance gate both permit otherwise.
7. Never rewrite or discard negative experimental evidence merely to obtain a passing result.
8. Prefer reproducible fixes in the canonical repository over ad-hoc changes made only in a running environment.

## 8. Anti-Loop Rules

A future agent must **not** repeatedly:

- search for the "real" bot inside `miladchicomobot` after this repository has been identified;
- recreate dependencies already declared in the canonical project;
- interpret a fail-closed endpoint as a missing-feature regression without checking governance;
- blame CoinMarketCap or another provider for a database/storage failure;
- migrate databases or change Railway source repositories simply to clear an error without first identifying the root cause;
- duplicate the same research implementation across both repositories.

If repository roles appear inconsistent with this document, **verify current GitHub and Railway state first**, then update this contract as part of the same change.

## 9. Governance Summary

```text
parsa314/modular-crypto-trading-bot
    = CANONICAL SCIENTIFIC / ENGINEERING REPOSITORY
    = source of truth for research, models, experiments, thesis evidence

parsa314/miladchicomobot
    = PUBLIC DEPLOYMENT / MONITORING SHELL
    = intentionally lightweight and fail-closed

Railway deployment source
    = may point to the shell for operational reasons
    != project source of truth
```

**Evidence Before Opinion. Reproduce before promoting. Infrastructure failures must be diagnosed before research logic is changed.**
