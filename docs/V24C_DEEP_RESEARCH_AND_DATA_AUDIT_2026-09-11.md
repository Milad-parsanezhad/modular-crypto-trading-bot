# v0.24c Deep-Research + Data-Audit Rationale — 2026-09-11

## Why a deeper research pass was required

The v0.24/v0.24b evidence showed that model quality cannot be judged from classification metrics or event-level trade filtering alone. Family-specific ML sometimes improved per-trade expectancy, yet the portfolio-level result deteriorated after simultaneous signals competed for symbol, strategy, direction and total-risk budgets. Therefore v0.24c treats **validation methodology, non-stationarity and portfolio economics as part of the ML problem itself**.

This document records the design decisions made before reading v0.24c external outcomes.

## Literature-driven methodological decisions

The research review concentrated on recent financial-ML work and established backtest-overfitting methodology. The resulting design rules are:

1. **Single holdout / single walk-forward is not sufficient for model-search evidence.** Use purging/embargo, CSCV/PBO diagnostics and ultimately CPCV/multiple-testing-aware validation. A terminal external/future sample is consumed only once by a frozen candidate.
2. **Temporal models are challengers, not automatic upgrades.** LSTM, GRU, TCN, CNN-LSTM and Transformer models are evaluated on event-aligned strategy targets, rather than generic next-bar accuracy. Development fits the network; validation selects model/seed/threshold; a previously observed terminal period is not reused to promote a deep model.
3. **Non-stationarity is first-class.** Regime/context features and drift diagnostics may condition models, but an unsupervised cluster is not alpha by itself.
4. **Vision/multimodal value must be incremental.** A visual/localization representation is retained only if it adds OOS economic value beyond numeric/temporal baselines. Multimodality is not forced.
5. **Uncertainty/calibration is an abstention tool, not a rescue mechanism.** Calibration/conformal-style uncertainty can later reduce exposure when confidence is weak, but calibration parameters must be frozen before the terminal sample.
6. **Portfolio selection is part of model evaluation.** Overlap, correlation, transaction costs, MTM drawdown and CVaR are evaluated jointly with the signal filter/allocator.
7. **RL remains downstream.** Reinforcement learning will initially be limited to allocation/execution in PAPER/testnet after a frozen state representation and portfolio allocator survive fresh evidence. It cannot repair a weak predictive representation by repeatedly interacting with the same historical test.

## Internal artifact data audit

The v0.24b evidence bundle contains 4,979 strategy events across 12 CoinEx spot symbols. Descriptive segment analysis of the two most defensible candidates showed material regime instability:

- `H4_S6_BREAKOUT`: mean R moved from approximately +0.085R in development to -0.241R in validation and +0.156R in the already-seen terminal shadow segment.
- `H4_D1_OB_BOS_RISK`: mean R moved from approximately +0.065R in development to -0.031R in validation and -0.012R in the terminal shadow segment.

Rolling 60-day windows also changed sign multiple times. These are descriptive diagnostics, not promotion evidence. They justify keeping regime/drift analysis and fresh-time validation in the protocol.

## Frozen Plan A candidates

Only two v0.24b hypotheses are carried into Plan A:

- `H4_S6_BREAKOUT`: frozen `sgd_logistic`, seed 314, threshold `0.2992513650430247`.
- `H4_D1_OB_BOS_RISK`: frozen `logistic`, seed 314, threshold `0.5481968244713689`.

The models are reconstructed from the frozen historical CoinEx coverage and must reproduce the exact v0.24b development/validation/shadow event counts and the exact causal feature-schema hash before any external score is read. Failure to reproduce is a protocol block, not a reason to retune.

## External/OOD evidence design

Plan A uses Bybit public spot OHLCV and a fixed 12-symbol universe disjoint from the v0.24b 12-symbol training panel:

`ATOM, ETC, FIL, NEAR, UNI, APT, ARB, OP, SUI, INJ, AAVE, TON` (USDT pairs).

Symbols are filtered only for pre-defined data availability/quality; outcomes cannot be used to add or remove a market after the external read.

The external calendar overlaps the internal historical period. Therefore this is **cross-venue + cross-sectional OOD evidence**, not the strongest possible future-time evidence. Even a positive Plan A result remains subject to future-time Plan B before a strong real-world claim.

## New MTM risk semantics

v0.24b measured realized-equity drawdown. v0.24c adds a point-in-time open-book replay:

- risk-based notional sizing from entry-to-stop distance;
- one active position per symbol;
- maximum 1.00% total open risk;
- maximum 0.50% strategy open risk;
- maximum 0.75% same-direction open risk;
- maximum five concurrent positions;
- pairwise-correlation risk scaling above |rho| = 0.80;
- rolling historical 95% CVaR with a provisional 2% loss budget;
- close-to-close MTM drawdown;
- conservative intrabar adverse stress mark (bar low for long, bar high for short);
- full 24 bps liquidation friction reserved while a position is open;
- hard 5% MTM drawdown kill.

This is intentionally stricter than event-level compounding.

## Parallel temporal-ML track

A separate development/validation-only tournament now evaluates:

- LSTM;
- GRU;
- TCN;
- CNN-LSTM;
- Transformer;

with three deterministic seeds, 64-bar causal windows and expanding-only normalization. Sequence windows terminate at the strategy signal close. Outcome labels never enter the input tensor. The v0.24b terminal shadow is not scored in this stage.

A temporal champion is only a **CHALLENGER** until it is frozen and evaluated on fresh external/future evidence.

## Pre-registered failure routing

- Plan A technically blocked / insufficient history -> Plan B future-time validation.
- Plan A valid economic failure -> Plan D research redesign; do not test another rescue model on the same external sample.
- Temporal challenger weaker than tabular -> retain tabular model; do not force deep learning.
- MTM/CVaR failure despite event-level profit -> reject portfolio promotion.
- External success but common-calendar concern -> require future-time Plan B for stronger evidence.

`forward_paper_authorized = false`

`live_execution_authorized = false`
