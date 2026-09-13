# AI Cryptocurrency Trading Bot Research — Machine Learning, Deep Learning & Reinforcement Learning

**Evidence-driven MSc research platform for an AI cryptocurrency trading bot using financial machine learning, deep learning, portfolio risk management and reproducible out-of-sample validation.**

> **Scientific contract:** Evidence Before Opinion  
> **Execution status:** `LIVE_EXECUTION = false`  
> **Research status:** promising signals exist, but **no strategy is claimed as guaranteed profitable alpha**.

> **Agent / repository routing:** This repository is the **canonical scientific and engineering source of truth**. `parsa314/miladchicomobot` is a lightweight public deployment/monitoring shell, not the main research codebase. Any coding or research agent must read [`AGENTS.md`](AGENTS.md) before changing repository roles, Railway routing, execution flags, dependencies, or scientific logic.

## What this project is

This repository is a public research program for building and falsifying an **AI/ML crypto trading system** under realistic market constraints. It covers strategy discovery, causal feature engineering, meta-labeling, LSTM/GRU/TCN/Transformer challengers, candlestick/market-structure vision research, portfolio risk, future reinforcement-learning allocation, and defense-grade reproducibility.

The research question is deliberately stricter than “does the backtest look good?”:

> **Can an AI cryptocurrency trading hypothesis survive leakage controls, realistic costs, model-selection bias, regime change, portfolio overlap, CVaR/drawdown limits and genuinely fresh evidence?**

Negative, rejected and blocked experiments are preserved because they are part of the thesis evidence trail.

## Research architecture

`Literature & Hypothesis → Causal/PIT Data → Strategy Events → Tabular/Temporal/Vision ML → Purged Validation → Search-Aware Statistics → External/Future Evidence → MTM/CVaR Portfolio Risk → Forward PAPER → RL Allocation/Execution → Thesis Defense`

Core principles:

- point-in-time / closed-bar feature construction;
- next-open execution semantics where applicable;
- realistic fees, slippage and 24/36/60 bps cost stress;
- development / validation / test separation with purge + embargo;
- simple baselines before complex ML;
- multi-seed evaluation for stochastic models;
- ablation and incremental-value tests;
- moving-block bootstrap and multiple-testing controls;
- CPCV / PBO / Deflated-Sharpe-style diagnostics where applicable;
- mark-to-market portfolio risk, correlation and CVaR constraints;
- immutable manifests, artifact digests and dependency provenance;
- no same-test rescue tuning;
- LIVE execution remains fail-closed.

## Current research state — 2026-09-11

The active research chain is maintained through stacked, auditable pull requests:

- **v0.23r — ML rebuild and leakage hardening:** panel split, feature deny-list, provenance and research-CI hardening. [PR #21](https://github.com/parsa314/modular-crypto-trading-bot/pull/21)
- **v0.24 — strategy-aware meta-labeling:** local metrics improved but breadth, bootstrap, PBO/DSR and cost-stress gates failed. **`NO_META_MODEL_PROMOTED`.** [PR #22](https://github.com/parsa314/modular-crypto-trading-bot/pull/22)
- **v0.24b — family-specific meta-labeling + portfolio risk:** some family uplift appeared, but the filtered portfolio underperformed and hit the drawdown kill. **`NO_FAMILY_META_PROMOTION`.** [PR #23](https://github.com/parsa314/modular-crypto-trading-bot/pull/23)
- **v0.24c — temporal challengers + fresh-evidence ladder:** validation-only TCN/LSTM challengers were frozen; external scoring stopped on a one-event model-reproduction mismatch. [PR #24](https://github.com/parsa314/modular-crypto-trading-bot/pull/24)
- **v0.24d — exact frozen replay + dual-venue external triangulation:** exact persisted-model replay passed. OKX and KuCoin event-level economics were positive after cost, but portfolio PF / MTM-DD / bootstrap gates failed. **`EXTERNAL_REPLICATION_FAILED`.** [PR #25](https://github.com/parsa314/modular-crypto-trading-bot/pull/25)
- **v0.25 — risk-aware portfolio arbitration:** active redesign separates “is this trade acceptable?” from “which simultaneous trade deserves scarce portfolio risk?” using learning-to-rank, uncertainty-aware admission and future-time validation.

### Key v0.24d discovery

The exact frozen filter retained positive event-level economics on two venues, yet portfolio-level performance failed the frozen risk gate. That creates a new research hypothesis:

**Signal alpha and portfolio arbitration are different learning problems.**

The current allocator uses deterministic tie-breaking when several selected events compete for limited risk budget. v0.25 therefore studies risk-aware ranking before any reinforcement-learning allocator is allowed.

## Model families under research

- Logistic / Ridge / SGD baselines
- Random Forest / Extra Trees / Gradient Boosting / HistGradientBoosting
- optional XGBoost / LightGBM / CatBoost arms
- LSTM / GRU / CNN-LSTM / TCN / Transformer temporal models
- computer-vision / multimodal representation research
- unsupervised regime and anomaly context
- learning-to-rank / uncertainty-aware portfolio admission
- reinforcement learning only after simpler allocation baselines survive fresh evidence

## Strategy families under research

- causal Ichimoku variants including S6 and Kumo/triangle hypotheses
- Order Block / BOS / retest variants
- Supply & Demand
- liquidity sweep / ICT/TTrades-derived causal adaptations
- correlation divergence / SMT-style hypotheses
- time-series momentum and higher-timeframe regime filters

Source-derived trading concepts are treated as **hypothesis generators**, never as proof of alpha.

## Research / defense navigation

- [`docs/PROJECT_STATUS_2026-09-11.md`](docs/PROJECT_STATUS_2026-09-11.md) — current evidence snapshot
- [`docs/RESEARCH_TRACEABILITY_MATRIX.md`](docs/RESEARCH_TRACEABILITY_MATRIX.md) — research-to-code/status matrix
- [`docs/DEFENSE_EVIDENCE_INDEX.md`](docs/DEFENSE_EVIDENCE_INDEX.md) — examiner question → evidence map
- [`docs/RESEARCH_CHANGELOG.md`](docs/RESEARCH_CHANGELOG.md) — stage-by-stage research updates
- [`docs/AI_CRYPTO_TRADING_BOT_RESEARCH.md`](docs/AI_CRYPTO_TRADING_BOT_RESEARCH.md) — search-friendly project overview

## Install

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Optional research stacks:

```bash
python -m pip install -e '.[ml]'
python -m pip install -e '.[deep]'
python -m pip install -e '.[rl]'
```

## Reproducibility rule

Every promotable experiment preserves source SHA, frozen hypothesis/config, data provenance, dependency environment, dataset/artifact fingerprints, validation-only selection logic, fresh evaluation evidence, cost/risk assumptions, machine-readable decision output, and explicit PAPER/LIVE authorization flags.

A successful workflow means the experiment executed according to contract. It does **not** automatically mean the strategy is economically valid.

## SEO / discovery keywords

**AI cryptocurrency trading bot · crypto trading bot machine learning · reinforcement learning trading bot · financial machine learning · algorithmic crypto trading · LSTM crypto trading · Transformer trading model · portfolio optimization · CVaR risk management · Ichimoku AI trading · ICT market structure · quantitative trading research · MSc artificial intelligence thesis**

## Thesis / defense labels

- **TESTED** — executed under a frozen protocol
- **REJECTED** — tested and failed a scientific gate
- **BLOCKED** — infrastructure/data/reproducibility prevented a valid read
- **CHALLENGER / HYPOTHESIS** — promising but not externally/forward validated
- **DATA_UNAVAILABLE** — evidence is not fabricated or silently proxied

## Safety

Predictive models do not directly authorize exchange execution. Portfolio/risk controls are independent, and research branches keep real-money execution disabled. Nothing in this repository is financial advice or a guarantee of returns.

## Repository governance

- [Agent operating contract](AGENTS.md)
- [Contribution guidelines](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [Research PR template](.github/PULL_REQUEST_TEMPLATE.md)

---

Maintained by **Milad Parsanezhad** as an academic financial-ML, AI cryptocurrency trading-bot and quantitative-research project.