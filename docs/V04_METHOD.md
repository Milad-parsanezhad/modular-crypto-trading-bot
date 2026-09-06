# v0.4 Historical Archive + Alpha Ablation Lab

## Question
Does Funding, same-venue realized Basis, cross-venue taker Order Flow, or historical Open Interest add **incremental out-of-sample information** beyond a fixed market baseline?

## Anti-leakage rules
- No random shuffling.
- No backfill.
- External derivatives/archive observations are lagged.
- Binance daily OI is conservatively treated as available the next day.
- Every model variant inside a panel uses identical dates.
- Preprocessing is fitted inside each training fold.

## Validation
- Expanding walk-forward.
- Purge gap between train and test.
- Fixed model hyperparameters and fixed trade threshold; no test-set tuning.
- Explicit fee + slippage turnover cost.
- Predictive metrics: AUC, Brier, log loss.
- Economic metrics: net return, Sharpe, max drawdown, exposure, turnover.
- Paired moving-block bootstrap on strategy-return differences.
- Benjamini-Hochberg multiple-testing adjustment.

## Evidence gate
`candidate_incremental_alpha` requires positive Delta Sharpe, positive 95% block-bootstrap lower confidence bound for incremental mean return, and BH-adjusted p <= 0.10. Otherwise the family is either `no_incremental_evidence` or `inconclusive`.

## Source labels
CoinEx remains the primary execution/price venue. Binance Vision archives are cross-venue features only. Kline taker-buy imbalance is a coarse Order Flow proxy, not L2/L3 order-book microstructure.

No live trading is authorized by v0.4.
