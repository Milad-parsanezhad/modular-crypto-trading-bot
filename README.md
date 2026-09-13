# AI Cryptocurrency Trading Bot Research — Machine Learning, Deep Learning & Reinforcement Learning

**Evidence-driven MSc research platform for an AI cryptocurrency trading bot using financial machine learning, deep learning, portfolio risk management and reproducible out-of-sample validation.**

> **Scientific contract:** Evidence Before Opinion  
> **Execution status:** `RESEARCH_ONLY / PAPER_OFF / LIVE_OFF`  
> **Latest completed scientific result:** `v0.50 — V50_NONOVERLAP_FAILURE_SUPPORTED`  
> **Active prospective hypothesis:** `v0.51 — overlap-conflict arbitration`  
> **Research claim:** no strategy is claimed as guaranteed profitable alpha.

## What this project is

This repository is a public research program for building and falsifying an **AI/ML crypto trading system** under realistic market constraints. It covers strategy discovery, causal feature engineering, meta-labeling, temporal/deep challengers, portfolio risk, failure attribution, prospective validation and defense-grade reproducibility.

The research question is deliberately stricter than “does the backtest look good?”:

> **Can an AI cryptocurrency trading hypothesis survive leakage controls, realistic costs, model-selection bias, regime change, portfolio overlap, CVaR/drawdown limits and genuinely fresh evidence?**

Negative, rejected and blocked experiments are preserved because they are part of the thesis evidence trail.

## Canonical current state — 2026-09-13

The exact current navigation record is:

- [`docs/CANONICAL_RESEARCH_STATUS_2026-09-13.md`](docs/CANONICAL_RESEARCH_STATUS_2026-09-13.md)

The repository intentionally preserves later experiments as **stacked, auditable scientific branches/PRs** rather than flattening preregistration and empirical stages into one destructive merge.

### v0.50 — latest completed scientific result

Canonical decision: **`V50_NONOVERLAP_FAILURE_SUPPORTED`**.

The frozen v0.50 diagnostic traced the canonical Expected-R pipeline stage-by-stage and found that the **earliest-first non-overlap transformation** was the only pipeline stage satisfying the preregistered broad-harm criterion across the five consumed development folds. That is a mechanistic failure-attribution result, not evidence that any replacement arbitration rule is profitable.

Canonical provenance:

- workflow run `34707823108`;
- scientific head `1dd0b1fe506fc51ceec4ff8934b77090f86b6cc2`;
- artifact `10302830689`;
- artifact digest `sha256:da5813a8c031f9da6cc940fda942efc846ca536dbd952f4930584c30732373e9`.

### v0.51 — active prospective question

v0.51 was preregistered before admissible prospective outcomes. It compares:

1. **A0:** frozen earliest-first overlap control;
2. **A1:** `SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION` — when flat, simultaneous same-entry candidates are ranked by the already-frozen Expected-R; active trades cannot be pre-empted.

Frozen preregistration commit: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`.  
Verification run: `34738934441` — SUCCESS.  
Prospective boundary: `2026-09-13T08:00:00Z`.  
Kraken remains sealed; PAPER and LIVE remain disabled.

No v0.51 economic result is valid before its frozen prospective support conditions are satisfied.

## Research architecture

`Literature & Hypothesis → Causal/PIT Data → Strategy Events → ML/Calibration → Purged Validation → Failure Attribution → Search-Aware Statistics → External/Future Evidence → Portfolio Arbitration & MTM/CVaR Risk → PAPER Review → Thesis Defense`

Core principles:

- point-in-time / closed-bar feature construction;
- next-open execution semantics where applicable;
- realistic fees, slippage and cost stress;
- development / validation / test separation with purge + embargo;
- simple baselines before complex ML;
- multi-seed or perturbation evaluation where stochasticity is material;
- ablation and incremental-value tests;
- moving-block bootstrap and multiple-testing controls;
- CPCV / PBO / Deflated-Sharpe-style diagnostics where applicable;
- mark-to-market portfolio risk, correlation and CVaR constraints;
- immutable manifests, artifact digests and dependency provenance;
- no same-test rescue tuning;
- PAPER/LIVE execution remains fail-closed until a separate promotion path is satisfied.

## Research / defense navigation

- [`docs/CANONICAL_RESEARCH_STATUS_2026-09-13.md`](docs/CANONICAL_RESEARCH_STATUS_2026-09-13.md) — canonical current scientific state
- [`docs/PROJECT_STATUS_2026-09-11.md`](docs/PROJECT_STATUS_2026-09-11.md) — earlier v0.25-era integrated snapshot
- [`docs/RESEARCH_TRACEABILITY_MATRIX.md`](docs/RESEARCH_TRACEABILITY_MATRIX.md) — research-to-code/status matrix
- [`docs/DEFENSE_EVIDENCE_INDEX.md`](docs/DEFENSE_EVIDENCE_INDEX.md) — examiner question → evidence map
- [`docs/RESEARCH_CHANGELOG.md`](docs/RESEARCH_CHANGELOG.md) — stage-by-stage research updates
- [`docs/AI_CRYPTO_TRADING_BOT_RESEARCH.md`](docs/AI_CRYPTO_TRADING_BOT_RESEARCH.md) — project overview

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

Every promotable experiment preserves source SHA, frozen hypothesis/config, data provenance, dependency environment, dataset/artifact fingerprints, validation-only selection logic, fresh evaluation evidence, cost/risk assumptions, machine-readable decision output and explicit PAPER/LIVE authorization flags.

A successful workflow means the experiment executed according to contract. It does **not** automatically mean the strategy is economically valid.

## Model and strategy families investigated

The program has evaluated or instrumented logistic/Ridge baselines, random forests and gradient boosting, temporal/deep challengers, calibrated competing-risk models, Ichimoku, ICT/SMC/Brooks-inspired causal event families, order-flow/microstructure hypotheses and portfolio-ranking/arbitration layers. Reinforcement learning remains gated until simpler signal/allocation hypotheses survive stronger fresh-evidence tests.

Source-derived trading concepts are treated as **hypothesis generators**, never as proof of alpha.

## Thesis / defense labels

- **TESTED** — executed under a frozen protocol
- **REJECTED** — tested and failed a scientific gate
- **BLOCKED** — infrastructure/data/reproducibility prevented a valid read
- **CHALLENGER / HYPOTHESIS** — promising but not externally/forward validated
- **DATA_UNAVAILABLE** — evidence is not fabricated or silently proxied
- **PREREGISTERED / PROSPECTIVE** — design frozen before admissible future outcomes

## Safety

Predictive models do not directly authorize exchange execution. Portfolio/risk controls are independent. The canonical current deployment is research/monitoring only, with PAPER and real-money LIVE execution disabled. Nothing in this repository is financial advice or a guarantee of returns.

## Repository governance

- [Contribution guidelines](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [Research PR template](.github/PULL_REQUEST_TEMPLATE.md)

---

Maintained by **Milad Parsanezhad** as an academic financial-ML, AI cryptocurrency trading-bot and quantitative-research project.
