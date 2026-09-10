# v0.23 Machine-Learning Evidence Protocol

## Objective

v0.23 is designed to answer a narrower and more defensible question than "which model predicts price best?":

> After realistic trading costs, risk constraints, model-selection controls, and an untouched final test, does any machine-learning candidate add economically useful evidence that is strong enough to justify prospective PAPER evaluation?

No v0.23 result can directly authorize LIVE execution.

## Research basis

The design is informed by recent peer-reviewed findings that are especially relevant to this project:

- Cakici, Shahzad, Będowska-Sójka & Zaremba (2024), *International Review of Financial Analysis*, DOI `10.1016/j.irfa.2024.103244`: cross-sectional crypto predictability can be economically meaningful, but simple models can remain competitive and apparent alpha is often concentrated in small, illiquid, difficult-to-trade assets. This motivates simple baselines, liquidity/capacity controls, and no automatic preference for deeper networks.
- Li et al. (2026), *Pacific-Basin Finance Journal*, DOI `10.1016/j.pacfin.2025.103033`: in a high-dimensional cryptocurrency factor framework, tree models outperformed neural networks in their reported comparison. This motivates a broad tabular model zoo rather than assuming deep learning dominance.
- Giantsidi & Tarantola (2025), *International Review of Economics & Finance*, DOI `10.1016/j.iref.2025.104719`: recent financial forecasting literature spans MLP, LSTM/GRU, CNN, TCN, autoencoders, hybrid and multimodal systems; model choice must be task- and evidence-driven.
- 2026 systematic review in *Computers & Electrical Engineering*, DOI `10.1016/j.compeleceng.2026.111110`: a persistent gap exists between statistical forecast accuracy and practical profitability. v0.23 therefore makes economic validation a first-class gate.
- 2026 multimodal financial forecasting work in *Machine Learning with Applications*, DOI `10.1016/j.mlwa.2026.100840`: cross-attention and calibrated uncertainty are promising for heterogeneous financial modalities. v0.23 adopts calibration/uncertainty diagnostics now; more complex adaptive fusion remains a challenger, not a default assumption.
- Bailey & López de Prado (2014), *Journal of Portfolio Management*, DOI `10.3905/jpm.2014.40.5.094`: selection bias and multiple testing inflate backtest performance. v0.23 therefore freezes the champion before the final test and records the number of candidate trials. DSR/PBO/SPA remain required downstream.

## Scientific contract

1. Every feature at signal time `t` must be computable from data available at or before `close[t]`.
2. The primary economic target begins strictly after the signal: hypothetical fill at `open[t+1]`, exit at `open[t+2]`.
3. The positive classification target requires the future gross return to exceed the baseline round-trip cost hurdle (24 bps).
4. Course-only Wyckoff/ICT proxies are excluded from promotable v0.23 ML features because v0.22d produced no incremental evidence for them.
5. Split order is per symbol: development -> calibration -> validation -> final_test, with target-horizon purge and an embargo around boundaries.
6. Model fitting uses development only. Probability calibration uses calibration only. Threshold and champion selection use validation only. The final test is inspected only after the champion is frozen.
7. External-venue replication is frozen: no model refit, recalibration, or threshold retuning on the external venue.

## Candidate model families

### Tabular

- Logistic regression — mandatory simple baseline.
- Histogram Gradient Boosting.
- Random Forest.
- Extra Trees.
- MLP classifier.
- XGBoost when installed.
- LightGBM when installed.
- CatBoost when installed.

### Deep temporal challengers

- LSTM.
- GRU.
- Temporal Convolutional Network (TCN).
- Transformer encoder.

Each deep family is trained across multiple random seeds. Early stopping is confined to an internal tail of the development split, probability calibration uses the separate calibration split, and family selection uses validation only.

## Probability quality and uncertainty

Prediction quality is evaluated with ROC-AUC, balanced accuracy, log loss, Brier score, and expected calibration error (ECE). Platt scaling is fitted only on the calibration split.

A split-conformal binary diagnostic is also recorded. Because financial time series are dependent and non-stationary, classical exchangeability guarantees are **not** claimed. The conformal layer is an uncertainty diagnostic and future research hook, not a mathematical guarantee of live-trading safety.

## Capital management and economic simulation

Baseline friction is 12 bps one way (24 bps round trip), with stress tests at 18 and 30 bps one way (36 and 60 bps round trip).

Sizing is constrained by:

- risk budget: 0.25% per selected event;
- ATR-based sizing denominator using a 1.5 ATR risk distance;
- minimum stop-distance proxy: 0.3%;
- maximum single-asset weight: 35%;
- maximum simultaneous portfolio gross exposure: 70%;
- confidence scaling above the frozen probability threshold;
- causal hard drawdown kill switch at 5%.

The 1.5 ATR quantity is used for position sizing in this one-period historical experiment; the experiment does not falsely claim intrabar stop execution when only OHLC bars are available.

## Validation objective

Probability threshold selection occurs on validation only. Candidate thresholds must produce a minimum event count. The objective combines net compounded return, risk-adjusted return, and drawdown penalty after costs. This objective is a selection rule, not a proof of expected future profitability.

## Final and external gates

A candidate can be labelled `FORWARD_PAPER_ML_CANDIDATE` only if it simultaneously clears internal final-test requirements, external-venue requirements, and the 60 bps stress gate. Otherwise the formal state remains `NO_ML_ALPHA_PROMOTION`.

Even if all v0.23 gates pass:

- PAPER replacement remains false;
- LIVE authorization remains false;
- Vision/Multimodal-to-RL connection remains false until sufficient immutable prospective PAPER evidence is accumulated.

## Required later robustness

v0.23 is deliberately not the end of the statistical audit. Before any live-readiness claim, the surviving frozen candidate must undergo DSR, PBO/CPCV or an equivalent combinatorial overfit audit, SPA/Reality Check where appropriate, block-bootstrap inference, regime stability, capacity/slippage analysis, and prospective immutable PAPER execution with fill reconciliation.

## RL policy

Localization v0.22c and multimodal representation v0.22d have both passed their representation gates. This satisfies the user's prerequisite for *considering* a Vision/Multimodal RL state, but it does not yet make RL scientifically justified. RL is therefore kept disconnected until v0.23 economic evidence and prospective PAPER evidence establish a stronger state representation and reward foundation. If it is later admitted, RL should initially be tested for allocation/risk/execution decisions against simpler policy baselines rather than being assumed to create directional alpha.
