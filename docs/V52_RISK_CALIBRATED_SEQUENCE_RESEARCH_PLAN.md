# v0.52 Risk-Calibrated Sequence Research Track

Status: RESEARCH_SANDBOX_ONLY
Date: 2026-09-13

## Governance boundary

This track is intentionally isolated from the frozen v0.51 prospective protocol.
It must not alter the v0.51 predictor identity, thresholds, labels, cost assumptions,
asset/venue universe, prospective start, first-seen policy, or scientific decision.

Execution remains fail-closed:

- LIVE_EXECUTION = false
- PAPER_EXECUTION = false
- Kraken remains sealed
- no profitability claim is authorized

## Motivation from current literature

The 2025-2026 literature suggests three technically relevant directions for the next
post-v0.51 experimental generation:

1. Modern sequence models should be benchmarked on risk-adjusted trading outcomes,
   not forecasting error alone. A 2026 large-scale financial benchmark reports that
   temporal representation models can outperform simpler baselines, but rankings vary
   materially across Sharpe, downside risk, transaction-cost breakeven, seeds, and
   computational efficiency. Reference: Saly-Kaufmann et al., arXiv:2603.01820.

2. State-space / Mamba-style sequence models are promising for long temporal contexts,
   but financial use should include uncertainty and robust OOS testing rather than
   importing generic forecasting claims. References include CMDMamba (Frontiers in AI,
   2025, doi:10.3389/frai.2025.1599799), MambaTS (Pattern Recognition, 2026,
   doi:10.1016/j.patcog.2026.114536), and QuantFlow (arXiv:2607.02632).

3. Nonstationary tail risk should be calibrated separately from alpha prediction.
   Recent conformal-risk work studies time-decayed and regime-weighted calibration for
   one-sided VaR under drift, while 2025-2026 conformal backtesting work emphasizes
   conditional-coverage diagnostics in non-exchangeable financial time series.
   References: Schmitt, arXiv:2602.03903; Retzlaff et al., PMLR 266 (2025);
   Mathematics 2026, 14(15), 2847, doi:10.3390/math14152847.

Recent cost-aware DRL studies also reinforce that turnover, transaction costs,
regime sensitivity, and multi-seed stability must be explicit rather than hidden in a
single backtest. Relevant examples include Pacific-Basin Finance Journal 94 (2025)
102876 and Array 31 (2026) 100991.

## Proposed architecture

The experimental v0.52 comparison should be modular and ablation-first:

A. Feature / market-state layer
- causal OHLCV and existing point-in-time features
- existing Ichimoku / market-structure / microstructure features only when causally
  available
- regime descriptors separated from target construction
- no future-complete higher-timeframe values

B. Sequence representation benchmark
- strong linear / tree baseline
- LSTM baseline
- PatchTST-style baseline
- Mamba/state-space candidate only after the baseline harness is frozen
- identical train/validation/test chronology and transaction-cost assumptions

C. Decision layer
- supervised probability / expected-utility baseline
- offline sequence-policy candidate (Decision-Transformer family) only as an ablation
- PPO/SAC-style RL only in a separate experiment with multi-seed stability and a fixed
  environment; no direct promotion from in-sample reward

D. Risk layer
- existing deterministic RiskEngine remains the hard safety gate
- experimental conformal VaR wrapper calibrates tail-risk forecasts under drift
- effective-sample-size gate fails closed when regime localization is too sparse
- CVaR/ES remains a reportable risk metric; no automatic execution authorization

## New implementation in this branch

`research_bot/risk_calibration_v52.py` adds a model-agnostic experimental one-sided
VaR calibration layer using:

- nonconformity score = realized loss - predicted VaR
- exponential time-decay weights
- optional regime-similarity RBF weights
- weighted quantile correction
- effective-sample-size diagnostics
- fail-closed behavior when calibration history is insufficient

`tests/test_risk_calibration_v52.py` covers insufficient-history failure, positive
buffering when the base forecaster underestimates loss, regime-localization ESS failure,
and rejection of non-finite current forecasts.

This implementation is not a claim of reproducing any cited paper exactly; it is a
research adaptation whose empirical validity must be established by preregistered
ablation and OOS testing.

## Frozen evaluation requirements before any scientific promotion

At minimum:

- chronological train/validation/test split with no leakage
- purging/embargo when labels overlap
- minimum 20 independent seeds for stochastic RL candidates when computationally
  feasible; report the distribution, not only the best seed
- transaction fees, spread/slippage sensitivity, turnover, and cost breakeven
- cumulative return, Sharpe, Sortino, maximum drawdown, Calmar, profit factor
- VaR/ES exceedance diagnostics and conformal conditional-coverage checks
- paired/block-bootstrap uncertainty where appropriate
- symbol/venue breadth and regime breakdown
- comparison against buy-and-hold and simple deterministic strategy baselines
- explicit negative-result retention

Promotion rule: a new model may only replace the current research baseline if the gain
survives OOS, costs, seed variation, and risk calibration. A green CI run means code
integrity, not scientific success.

## Next implementation sequence

1. Keep v0.51 prospective evidence collection untouched.
2. Establish a frozen v0.52 benchmark dataset and split manifest.
3. Build common sequence-model interface and deterministic baseline harness.
4. Add PatchTST/LSTM first; add Mamba only under identical evaluation rules.
5. Add Decision-Transformer/offline-RL ablation after supervised baselines are frozen.
6. Integrate the conformal risk wrapper as a risk ablation, not as an alpha feature.
7. Run cost, regime, seed, and tail-risk audits.
8. Only then decide whether a v0.52 candidate merits a new prospective protocol.
