# v0.5 Research Basis — Regime-Conditional Alpha & Robustness

## Research question
Do Funding, Basis, Order Flow and Open Interest provide incremental economic information **conditionally on pre-specified market regimes and horizons**, after realistic costs, calibration, abstention controls and multiple-testing correction?

## Why v0.5 exists
v0.4 found no stable pooled alpha. Basis and a coarse Order-Flow proxy improved aggregate point estimates but failed multiplicity-adjusted inference. Their gains were fold-dependent. This motivates a regime-conditional test, not a more complex model.

## Pre-specified regimes
All definitions are point-in-time and use lagged rolling quantiles; they are not fitted to test returns.
- **Bull:** trailing 42-bar trend above its historical 70th percentile, outside high-vol/crisis.
- **Bear:** trailing trend below its 30th percentile, outside high-vol/crisis.
- **High-volatility:** trailing volatility above its historical 80th percentile.
- **Crisis:** weak trend (<= historical 10th percentile) and volatility >= historical 90th percentile.
- **Sideways:** remaining valid observations.

The regime model is deliberately simple and deterministic. HMM/Markov-switching models are challengers only after this transparent baseline.

## Pre-specified horizons
4h, 12h and 24h only. Multiplicity is handled jointly; we do not search arbitrary horizons.

## Statistical safeguards
- Outer expanding walk-forward.
- Purge gap increases with horizon.
- Fit / probability calibration / threshold selection are time-ordered and inside training only.
- Platt probability calibration.
- Round-trip fee + slippage charged for selected events.
- Non-overlapping horizon observations.
- Moving-block bootstrap of paired incremental strategy returns.
- Benjamini-Hochberg adjustment across horizon × regime comparisons.
- Coverage-matched baseline diagnostic to expose selective-prediction/persistence illusions.
- Regime-transition audit.
- CPCV sanity panel, Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR) diagnostics.

## Scientific basis
1. Agakishiev et al., **Regime switching forecasting for cryptocurrencies**, Digital Finance (2025), DOI: `10.1007/s42521-024-00123-2`. Regime information can help in-sample but OOS improvement was not guaranteed; therefore regime awareness is a falsifiable hypothesis.
2. **Hybrid Regime-Switching Models for Cryptocurrency Prices**, Stats (2026), DOI: `10.3390/stats9040071`. Motivates leakage-free Markov/regime-specific ML challengers.
3. **Selective Prediction and the Persistence Illusion: A Diagnostic Decomposition of VIX Regime Classification**, Journal of Risk and Financial Management (2026), DOI: `10.3390/jrfm19080609`. Motivates risk-coverage, transition and coverage-matched diagnostics.
4. Kaya & Nguyen, **Conformal Prediction for Reliable Stock Selections**, PMLR 266 (2025). Motivates uncertainty-aware selection as a later challenger.
5. Arian et al., **Backtest overfitting in the machine learning era**, Knowledge-Based Systems (2024), DOI: `10.1016/j.knosys.2024.112477`. Motivates CPCV/PBO-style robustness.
6. Bailey & López de Prado, **The Deflated Sharpe Ratio**, Journal of Portfolio Management (2014), DOI: `10.3905/jpm.2014.40.5.094`.
7. Bailey et al., **The Probability of Backtest Overfitting**, Journal of Computational Finance (2017), DOI: `10.21314/JCF.2016.322`.
8. Anastasopoulos et al., **Order flow and cryptocurrency returns**, Journal of Financial Markets (2026), DOI: `10.1016/j.finmar.2026.101047`.

## Kill rule
No regime/feature interaction is promoted to the robot merely because one cell is profitable. It must survive adjusted inference, coverage-matched diagnostics, CPCV/PBO/DSR, several folds, realistic costs and a later untouched forward test.
