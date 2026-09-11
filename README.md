# Modular Crypto Trading Bot

**Evidence-driven MSc research platform for cryptocurrency strategy discovery, financial machine learning, portfolio risk and reproducible out-of-sample validation.**

> **Scientific contract:** Evidence Before Opinion  
> **Execution status:** `LIVE_EXECUTION = false`  
> **Research status:** promising candidates exist, but **no strategy is claimed as validated guaranteed alpha or profit**.

## Why this repository exists

This project is not a collection of indicator screenshots or a backtest leaderboard. It is an academic research system designed to answer a harder question:

> **Can a cryptocurrency trading hypothesis survive realistic costs, leakage controls, model-selection bias, regime change, portfolio overlap and genuinely fresh evidence?**

The repository intentionally preserves **negative, rejected and blocked experiments** because they are part of the thesis evidence trail.

## Research architecture

`Hypothesis & Literature → Causal/PIT Data → Strategy Event Engine → Baselines → Tabular/Temporal ML → Purged Validation → Multiple-Testing Audit → Fresh External/Forward Evidence → MTM/CVaR Portfolio Risk → Forward PAPER → Defense Evidence`

Core principles:

- point-in-time / closed-bar feature construction;
- next-open execution semantics where applicable;
- fees, slippage and cost-stress scenarios;
- train/development, validation and test separation with embargo/purging;
- simple baselines before complex ML;
- multi-seed evaluation for stochastic models;
- ablation and incremental-value tests;
- moving-block bootstrap and search-aware diagnostics;
- PBO / Deflated-Sharpe-style diagnostics and CPCV when applicable;
- mark-to-market portfolio risk, correlation and CVaR constraints;
- immutable manifests, artifact digests and CI provenance;
- no same-test rescue tuning;
- LIVE execution remains fail-closed.

## Current research state — 2026-09-11

The active research chain is maintained through stacked, auditable pull requests:

- **v0.23r — ML rebuild and leakage hardening:** rigorous panel splitting, feature deny-list, deterministic provenance and research-CI hardening. [PR #21](https://github.com/parsa314/modular-crypto-trading-bot/pull/21)
- **v0.24 — strategy-aware meta-labeling:** internal test improved some local metrics but failed breadth, bootstrap, PBO/DSR and cost-stress promotion gates. **Decision: `NO_META_MODEL_PROMOTED`.** [PR #22](https://github.com/parsa314/modular-crypto-trading-bot/pull/22)
- **v0.24b — family-specific meta-labeling + overlap-aware portfolio risk:** some family-level shadow uplift appeared, but the filtered portfolio underperformed and breached the research DD kill. **Decision: `NO_FAMILY_META_PROMOTION`.** [PR #23](https://github.com/parsa314/modular-crypto-trading-bot/pull/23)
- **v0.24c — fresh-evidence ladder + MTM risk + temporal challengers:** external validation is fail-closed until exact frozen-model reproducibility is guaranteed. Temporal LSTM/GRU/TCN/CNN-LSTM/Transformer challengers are selected on validation only; no previous terminal test is reused for promotion. [PR #24](https://github.com/parsa314/modular-crypto-trading-bot/pull/24)

The v0.24c temporal run completed successfully as an engineering/validation experiment. Frozen validation challengers are **TCN (H4_S6_BREAKOUT)** and **LSTM (H4_D1_OB_BOS_RISK)**, but neither is promoted until fresh external or future-time evidence is consumed.

The external path deliberately stopped when exact frozen-model reproduction differed by one selected validation event. That mismatch is treated as a **reproducibility defect**, not silently rounded away.

See [`docs/PROJECT_STATUS_2026-09-11.md`](docs/PROJECT_STATUS_2026-09-11.md) for the defense-oriented evidence map.

## Major modules

- `research_bot/` — causal features, strategy labs, ML, risk and evidence contracts
- `scripts/` — reproducible experiment entry points
- `tests/` — leakage, causality, risk, provenance and regression tests
- `.github/workflows/` — CI evidence pipelines and prospective collection jobs
- `docs/` — protocols, dated result records, traceability and thesis/defense evidence

## Model families under research

The project treats models as challengers, not assumed winners:

- Logistic / Ridge / SGD baselines
- Random Forest / Extra Trees / Gradient Boosting / HistGradientBoosting
- optional XGBoost / LightGBM / CatBoost research arms
- LSTM / GRU / CNN-LSTM / TCN / Transformer temporal models
- unsupervised regime/anomaly context
- future ensemble and portfolio-allocation layers
- reinforcement learning only after upstream portfolio evidence passes

## Strategy families under research

- causal Ichimoku variants, including S6 and Kumo/triangle hypotheses
- Order Block / BOS / retest variants
- Supply & Demand
- correlation divergence / SMT-style hypotheses
- ICT/TTrades-derived causal adaptations
- time-series momentum and higher-timeframe regime filters

Source-derived trading concepts are treated as **hypothesis generators**, never as proof of alpha.

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

Every promotable experiment should preserve:

1. source commit SHA;
2. frozen configuration / hypothesis;
3. data provenance and coverage;
4. dependency environment;
5. dataset or artifact fingerprints;
6. validation-only selection logic;
7. untouched/fresh evaluation evidence;
8. cost and risk assumptions;
9. machine-readable decision output;
10. explicit PAPER/LIVE authorization flags.

A successful workflow means the experiment executed according to contract. It does **not** automatically mean the hypothesis is economically valid.

## Thesis / defense orientation

This repository is being maintained as an evidence package for an MSc thesis. Results are organized so that Chapter 4 and the defense can distinguish:

- **TESTED** — executed under a frozen protocol;
- **REJECTED** — scientifically tested and failed a gate;
- **BLOCKED** — infrastructure/data/reproducibility prevented a valid read;
- **CHALLENGER / HYPOTHESIS** — promising but not yet externally/forward validated;
- **DATA_UNAVAILABLE** — evidence is not fabricated or proxied without explicit labeling.

## Safety

Predictive models do not directly authorize exchange execution. Portfolio/risk controls are independent, and research branches keep real-money execution disabled. No result in this repository should be interpreted as financial advice or a guarantee of returns.

## Repository governance

- [Contribution guidelines](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)

---

Maintained by **Milad Parsanezhad** as an academic financial-ML and cryptocurrency research project.
