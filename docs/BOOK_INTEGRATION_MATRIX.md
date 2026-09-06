# AI/ML Trading Book Integration Matrix

This corpus distinguishes **legally open full text**, **commercial books with official companion code**, and **secondary/community implementations**. Commercial books are not copied into the repository. Their ideas are implemented only from legally accessible materials and official/public code.

## Legally Open Access — full-book study permitted

### 1) Signature Methods in Finance: An Introduction with Computational Applications (Springer Finance, 2026)
DOI: `10.1007/978-3-031-97239-3` — full book Open Access.

Chapter map and robot relevance:
- Primer / tensors / signature kernel → path-representation research library.
- Market Generators + Signature MMD → simulation and distribution-shift testing.
- Signature-based models → non-Markovian feature challenger.
- **Signature Trading Strategies** → explicit separation of alpha extraction from execution/risk; path-dependent strategies.
- Optimal stopping for non-Markovian processes → future execution/exit challenger.

**Implementation decision:** add a `path_signature` challenger after v0.5, beginning with spot + basis + funding paths. It must beat simple trailing moments under identical validation. A 2026 SSRN preprint specifically applies path signatures to crypto regime detection, so this is scientifically testable but remains experimental.

### 2) Agent AI for Finance: From Financial Argument Mining to Agent-Based Modeling (Springer, 2025)
DOI: `10.1007/978-3-031-94687-5` — 83 pages, all seven chapters Open Access.

- Financial Argument Mining → structured claims/evidence/stance from news and research.
- Single-Agent/Model Design → expert features + historical event memory; compare end-to-end vs engineered inputs.
- Multi-agent Interaction → later research agents with independent roles, not trading agents voting without evidence.
- Multi-scale Model Synergy → combine small specialist models with LLM/NLP rather than replacing quantitative models.
- Generative AI scenarios → report/research layer, not autonomous alpha by default.

**Implementation decision:** build a future `research_agent` layer that creates timestamped structured event features and provenance. It cannot place orders; its incremental value must be tested like any other feature family.

### 3) Liquidity, Markets and Trading in Action (Springer, 2022)
DOI: `10.1007/978-3-030-74817-3` — 103 pages, full Open Access.

- Liquidity / price determination → spread, depth, market impact, execution probability.
- Information shocks → event-risk/regime gates.
- Trading and Technology → automated-market safeguards, anomalous/spoofing-aware microstructure monitoring.
- TraderEx simulation → motivates market-structure simulation before production.

**Already reflected:** explicit transaction costs, slippage, order-flow ingestion and the rule that predictability is not tradability.
**Next implementation:** depth/impact/capacity model and market simulator calibration.

### 4) Data Science for Economics and Finance: Methodologies and Applications (Springer, 2021)
DOI: `10.1007/978-3-030-66891-4` — 355 pages, full Open Access.

Relevant themes: forecasting, interpretability, inference, text/social data and financial stability.
**Implementation decision:** SHAP/permutation/feature-family ablation and interpretable macro/event features; explanation is diagnostic, never evidence of causality.

## Commercial books — use official previews/code, not unauthorized full text

### Machine Learning for Algorithmic Trading — Stefan Jansen
Official public repository: `stefan-jansen/machine-learning-for-trading` (2nd edition >150 notebooks; 3rd-edition work is present in the public project).
Key integrations: point-in-time data, feature engineering, boosting baselines, Bayesian/time-series models, NLP, DL/RL challengers, portfolio and execution workflow.

### Machine Learning in Finance: From Theory to Practice — Dixon, Halperin, Bilokon
Commercial Springer book; official public code repository: `mfrdixon/ML_Finance_Codes`.
Key integrations: interpretability, sequence/probabilistic modeling and RL as challengers.

### Advances in Financial Machine Learning — Marcos López de Prado
Commercial Wiley book; use publisher sample material and legally accessible implementations.
Already reflected: event sampling, triple-barrier labels, purging/embargo concepts, backtest-overfitting controls. Every technique is independently validated rather than treated as doctrine.

### Machine Learning for Asset Managers — Marcos López de Prado
Commercial Cambridge book.
Key future challengers: denoising/detoning, clustering, economic feature importance, portfolio construction, overfitting controls.

### Artificial Intelligence in Finance — Yves Hilpisch
Commercial O'Reilly book; official public repository `yhilpisch/aiif`.
Key integration: controlled neural/RL experiments, financial AI workflows; production decisions remain evidence-gated.

### Python for Algorithmic Trading — Yves Hilpisch
Commercial O'Reilly book; official repository `yhilpisch/py4at`.
Key integration: event streaming, backtesting, deployment, monitoring and risk/execution engineering.

### Deep Learning for Finance — Sofien Kaabar
Commercial O'Reilly book; official public repository `sofienkaabar/deep-learning-for-finance`.
Key integration: DL models become challengers only after a feature family shows repeatable information content.

## Research rule for books
A book can propose an idea; it cannot certify alpha. Each extracted method enters the same pipeline:
`economic hypothesis → point-in-time implementation → simple baseline → ablation → robust OOS tests → costs/capacity → paper trading`.
