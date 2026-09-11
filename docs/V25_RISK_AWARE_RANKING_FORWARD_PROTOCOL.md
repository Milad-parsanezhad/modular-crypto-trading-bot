# v0.25 — Risk-Aware Portfolio Admission, Learning-to-Rank and Future-Time Validation

Date frozen: 2026-09-11  
Future evidence boundary: **2026-09-11T12:00:00Z**  
Status at freeze: **HYPOTHESIS / CHALLENGER DEVELOPMENT**  
LIVE execution: **disabled**

## 1. Motivation from v0.24d

v0.24d exactly replayed the persisted v0.24b event filters and applied them without refit to two pre-registered external venues. Event-level filtered economics remained positive after cost stress on both OKX and KuCoin, but the overlap-aware portfolio failed frozen portfolio-PF, MTM-drawdown and bootstrap gates.

The post-hoc diagnostic identified a specific architectural question: the event meta-model determines whether a candidate is admissible, but simultaneous admissible events are currently processed with deterministic strategy/symbol tie-breaking. When portfolio risk is scarce, **event selection and portfolio admission are not the same learning problem**.

Because this hypothesis was formed after reading v0.24d outcomes, all v0.24d OKX/KuCoin observations are scientifically **SPENT** for v0.25 promotion. They may be used for diagnosis only.

## 2. Research question

> Given a fixed, already-frozen event filter, can a causal portfolio-ranking layer improve the allocation of scarce risk among simultaneous candidate trades without increasing search overfitting, turnover fragility or tail risk?

## 3. Literature basis

The design is motivated by recent work in several adjacent areas:

1. **Learning-to-rank portfolio selection.** Recent portfolio research uses ranking rather than isolated binary predictions to express relative desirability across simultaneous candidates. See *Momentum portfolio selection based on learning-to-rank algorithms with heterogeneous knowledge graphs*, Applied Intelligence (2024), DOI: `10.1007/s10489-024-05377-2`.
2. **Risk-aware ranking / portfolio construction.** Ranking, portfolio cardinality and risk-aware allocation should be evaluated jointly rather than assuming a classifier score is an allocator objective. v0.25 therefore keeps a simple frozen-score ranking baseline and compares learned ranking challengers under the same risk contract.
3. **Trading-signal survival.** Hu & Zhou, *Trading Signal Survival Analysis*, Computational Economics 64 (2024), DOI: `10.1007/s10614-024-10567-8`, shows that signal duration/survival is a distinct predictive object that can filter technical signals. v0.25 does not yet use future duration outcomes as decision features; survival-aware capital-lock utility is reserved as a preregistered later challenger.
4. **Backtest-overfitting control.** Arian et al., *Backtest overfitting in the machine learning era*, Knowledge-Based Systems 305 (2024), DOI: `10.1016/j.knosys.2024.112477`, reports superior false-discovery control for combinatorial purged validation in its controlled experiments. A future v0.25 survivor must therefore pass search-aware CPCV/PBO/DSR-style review before PAPER replacement.
5. **Reinforcement learning is not the first allocator baseline.** Kruthof & Müller, *Can deep reinforcement learning beat 1/N?*, Finance Research Letters 75 (2025), DOI: `10.1016/j.frl.2025.106866`, finds SAC does not systematically dominate 1/N and can lose after modest costs because of turnover. v0.25 therefore requires deterministic/supervised ranking baselines before any RL allocator.
6. **Risk-aware DRL remains a later hypothesis.** Multi-period DRL studies explicitly embed risk aversion, costs and asset dependence. If v0.25 ranking survives future evidence, these constraints become mandatory components of any later RL state/reward/action contract rather than optional post-processing.

## 4. Comparable-system bug audit

Before implementation, comparable open-source failure modes were reviewed and translated into guards:

- **FinRL CCXT timestamp drift:** local-naive timestamp conversion can offset crypto bars across machines. v0.25 requires UTC-aware timestamps at every ingestion/join boundary and rejects ambiguous timestamps.
- **Qlib dependency-chain failures:** LightGBM/default and pandas/index behavior changes can reveal sequential incompatibilities. v0.25 freezes the persisted-model environment before deserialization and uses fail-fast tests rather than continuing after dependency drift.
- **Persisted estimator identity:** v0.24c already demonstrated that model-name/seed/threshold recreation is not equivalent to a frozen binary model. v0.25 inherits exact v0.24b artifact hashes and environment checks.
- **Backtest rescue bias:** no failed future window may be retuned and relabeled as untouched. A model change after future evidence spends that window permanently.

## 5. Frozen event-filter identity

v0.25 consumes the exact v0.24b snapshot verified by v0.24d:

- source run `34578494059`
- source artifact `v24b-family-portfolio-34578494059`
- model SHA-256 `ec4a81b7d708b0ffc7a80238668fa1bb34cccdac0408c51e4aff082075a5a22a`
- dataset SHA-256 `fe1a48a8854b9550283db41cc43821ba635b7a63b07eb48df894dd06f337e338`
- persistence environment: scikit-learn 1.9.1 / NumPy 2.5.3 / pandas 3.0.5 / joblib 1.6.0

The primary ranking experiment is limited to:

- `H4_S6_BREAKOUT`
- `H4_D1_OB_BOS_RISK`

No v0.25 ranking model may refit, recalibrate or alter those frozen event filters.

## 6. Ranking challengers

The initial tournament is intentionally compact to control the research search space:

- **B0 — frozen event-model score**: deterministic baseline;
- **B1 — frozen score margin** above the original threshold;
- **M1 — Ridge expected-R regressor**;
- **M2 — HistGradientBoosting expected-R regressor**;
- **M3 — lower-quartile HistGradientBoosting** as an uncertainty-aware conservative score;
- **M4 — pairwise logistic ranker** trained on within-timestamp relative realized-R orderings.

Training uses **development only**. Ranker selection uses **validation only**. v0.24d external observations cannot participate in model or hyperparameter selection.

If no learned ranker beats the frozen-score baseline under the validation allocator objective, the baseline is frozen instead of forcing a complex model.

## 7. Causal feature contract

Ranking features must be known at admission time. Permitted inputs are causal `f_*` features plus:

- strategy identity encoded deterministically;
- trade direction;
- exact frozen event-model probability/score;
- exact frozen score margin.

Explicitly prohibited as ranking inputs:

- `r_multiple` or realized return;
- exit price/time/reason as predictive features;
- terminal equity/drawdown;
- future bars;
- later portfolio acceptance state;
- post-entry information.

Realized outcomes are labels only during development/validation.

## 8. Validation allocator

Ranker selection uses an overlap-aware realized-event allocator with the existing research risk budget:

- risk per trade: 0.25%;
- max open portfolio risk: 1.00%;
- max per-strategy open risk: 0.50%;
- max directional open risk: 0.75%;
- max concurrent positions: 5;
- hard realized-DD research kill: 5%.

This allocator is only a **selection instrument**. Final scientific evidence must use the mark-to-market correlation/CVaR simulator on future data.

## 9. Future-time evidence gate

No observation before `2026-09-11T12:00:00Z` can count as v0.25 promotion evidence.

The frozen v0.25 ranker must be applied without refit to prospective 4h data. The future decision gate requires, at minimum:

- adequate independent event/admission sample size;
- filtered/allocated PF >= 1.05;
- positive mean-R and portfolio return after modeled costs;
- MTM / adverse-bar drawdown <= 5%;
- no unexplained data-quality gaps or timestamp drift;
- breadth across symbols/market states rather than one-asset dependence;
- positive paired moving-block uplift lower bound versus the frozen-score allocator;
- survival under 36 bps cost stress;
- later CPCV/PBO/DSR/search-aware review before any PAPER replacement claim.

A green engineering workflow is not a scientific PASS.

## 10. RL gate

RL remains **locked**. It may become an allocator challenger only after a deterministic/supervised v0.25 allocator survives fresh future evidence. Any later RL reward must penalize turnover, transaction costs, concentration, correlation/tail risk and drawdown; RL will not be permitted to optimize raw historical return without these constraints.

## 11. Defense contribution

v0.25 turns a negative external result into a falsifiable systems hypothesis:

> A financial ML pipeline can preserve useful event-level information while losing economic value at the portfolio admission layer.

This distinction gives the thesis an explicit decomposition of **signal quality → admission ranking → portfolio risk → execution evidence**, and prevents classification metrics from being misreported as trading-system profitability.
