# v0.24c — Multi-Plan ML Validation, External Evidence and Forward Protocol

## Objective

Build a research process that never depends on one fragile path. Every stage has pre-registered fallback plans, but a failed scientific test is **not** rescued by retuning on the same terminal data. Fallback changes the evidence source or returns the hypothesis to development; it does not cherry-pick the test.

The project priority is machine-learning quality, validation integrity, forward evidence, portfolio economics and reproducibility. Green CI alone is never promotion evidence.

## Evidence ladder

### Plan A — Untouched external venue

Primary route. Freeze strategy definition, features, model family, seed, threshold, risk contract and allocator before reading the terminal sample. Evaluate on an external venue/source that was not used for tuning.

If the external source is technically unavailable or structurally incompatible, classify `BLOCKED` or `DATA_INSUFFICIENT`; do not call it a fail and do not tune around the missing data.

If the hypothesis economically fails on valid external data, classify `FAIL` and route to Plan D, not to another model on the same external sample.

### Plan B — Future-time forward validation

Used only when Plan A is blocked/data-insufficient. Freeze the full candidate and accumulate genuinely future bars after the freeze timestamp. No parameter, threshold, feature or allocator changes are allowed while the evidence window accumulates.

### Plan C — Sequential PAPER/shadow accumulation

Used when a large-enough terminal sample is still unavailable. Store every decision, model score, abstention, intended risk, fill assumption and outcome. The model stays frozen. This route can build evidence but cannot silently redefine a previously failed hypothesis.

### Plan D — Research redesign

Used after a valid scientific failure. Return to development data, diagnose the failure, change one or more components, and issue a new experiment version. The redesigned hypothesis must later consume **new fresh evidence**. A previously seen terminal test cannot become untouched again.

## ML research tracks

1. **Tabular strategy-aware meta-learning** — logistic/ridge/SGD/tree/boosting and optional major GBDT families; development fit, validation-only selection, test once.
2. **Temporal sequence learning** — LSTM, GRU, TCN, Transformer/Patch-style sequence encoders on causal tensors with multi-seed evaluation. Sequence models are not compared on the same hyperparameter budget as tabular models without multiplicity accounting.
3. **Vision + multimodal** — raw-candle localization/representation followed by causal fusion with numeric and temporal state. Vision must add incremental OOS value over numeric baselines.
4. **Unsupervised regime discovery** — GMM/HMM/anomaly/regime features may condition another model but are never treated as standalone alpha.
5. **Portfolio allocator** — selection and sizing are optimized/evaluated at portfolio level with overlap, one-position-per-symbol, risk budgets, rolling correlation, MTM drawdown and CVaR.
6. **Ensemble/stacking** — allowed only after the contributing tracks independently pass their own validation gates; weak models are not averaged simply to improve appearance.
7. **Reinforcement learning** — remains disconnected until a frozen state representation and portfolio allocator have passed fresh evidence. RL is initially limited to PAPER/testnet allocation/execution, not unrestricted live trading.

## Promotion gate

A candidate cannot become a Forward-PAPER candidate unless all frozen requirements pass simultaneously:

- at least 200 fresh terminal trades/events;
- profit factor >= 1.05 after frozen baseline costs;
- positive mean R;
- MTM maximum drawdown <= 5%;
- positive-symbol fraction >= 60%;
- paired/block-bootstrap incremental uplift lower bound > 0;
- PBO <= 0.50;
- Deflated-Sharpe probability >= 0.95;
- non-negative / economically acceptable stress result at 36 bps round-trip friction;
- fresh external-venue or future-time evidence;
- no post-test refit, threshold tuning, feature addition or allocator retuning.

Passing this gate can authorize only `FORWARD_PAPER_CANDIDATE`. It never authorizes LIVE execution by itself.

## Portfolio risk contract

Baseline research controls:

- base risk per accepted trade: 0.25%;
- maximum simultaneous portfolio open risk: 1.00%;
- maximum per-strategy open risk: 0.50%;
- maximum same-direction open risk: 0.75%;
- maximum concurrent positions: 5;
- hard MTM drawdown kill: 5%;
- CVaR confidence level: 95%;
- provisional maximum CVaR loss budget: 2%;
- correlated positions above 0.80 rolling pairwise correlation do not receive full independent risk budget.

The v0.24b realized-equity simulator is not sufficient for final promotion. v0.24c requires point-in-time MTM replay for open positions before a portfolio risk claim is accepted.

## Failure routing and anti-loop policy

A software failure may retry once only if the retry changes the recovery mechanism. After repeated failure, record the fingerprint and use a different route. A scientific failure never triggers automatic hyperparameter expansion on the same test.

Examples:

- API unavailable -> external fallback source or future-time plan.
- insufficient external history -> future-time accumulation.
- model PF < gate on valid untouched data -> research redesign with a new version and future fresh test.
- Vision does not add incremental value -> remove it from the promoted stack; do not force multimodality.
- sequence model is weaker than tabular -> retain tabular champion; sequence remains challenger.
- family meta-filter harms portfolio return -> reject the filter; do not promote event-level metrics alone.

## Current frozen hypotheses entering v0.24c

The most defensible research hypotheses from the prior evidence are:

- `H4_S6_BREAKOUT` as the primary strategy hypothesis;
- `H4_D1_OB_BOS_RISK` as a secondary challenger;
- portfolio-level allocation/risk as a first-class model component rather than an after-the-fact filter.

The current CoinEx terminal sample is already seen and cannot be reused as untouched promotion evidence.

## Required v0.24c deliverables

- immutable experiment manifest and freeze timestamp;
- external/future data provenance and checksums;
- ML track registry and exact model budgets;
- causal feature and sequence manifests;
- untouched predictions and complete abstention ledger;
- MTM portfolio ledger;
- rolling correlation and CVaR diagnostics;
- cost stress at 24/36/60 bps;
- symbol/strategy/regime breadth;
- bootstrap, CPCV/PBO and DSR/multiple-testing diagnostics;
- exact source commit and dependency lock/freeze;
- decision file with explicit `PASS/FAIL/DATA_INSUFFICIENT/BLOCKED` per plan;
- `forward_paper_authorized=false` and `live_execution_authorized=false` unless a later dedicated gate explicitly changes the PAPER state.

## Scientific principle

The system must be difficult to fool. A model is valuable only when it adds incremental economic value after costs and portfolio interactions on fresh evidence. Accuracy, AUC, attractive charts or a single profitable backtest are insufficient.
