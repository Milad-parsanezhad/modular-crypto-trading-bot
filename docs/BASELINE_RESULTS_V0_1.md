# Baseline Results v0.1 — Real Market Run

Run date: 2026-09-06

## Setup
- Exchange: CoinEx
- Symbol: BTC/USDT
- Timeframe: 4h
- Raw bars returned: 1,000
- Train rows: 699
- Holdout test rows: 300
- Fee assumption: 10 bps per turnover unit
- Slippage assumption: 2 bps per turnover unit
- ML: HistGradientBoostingClassifier
- Decision rule: long/flat with abstention, `P(up) > 0.56`

## Results

| Strategy | Total return | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|
| Buy & Hold | 23.45% | 4.52 | 8.21 | -6.26% |
| 24-bar Momentum | 11.46% | 2.80 | 4.05 | -7.64% |
| ML + Abstention baseline | -3.91% | -3.48 | -2.27 | -4.31% |

## Scientific interpretation

The first real-market ML baseline **fails**. This is a useful research result, not a project failure.

It establishes several important facts:

1. A tree-based classifier using only OHLCV-derived technical/liquidity/regime features does not automatically produce tradable alpha.
2. Probability-based abstention alone is insufficient in the current design.
3. The simple momentum baseline dominates the current ML strategy, so model complexity has not yet earned its place.
4. Buy & Hold dominates both active baselines in this short holdout window.
5. The current result supports the literature-derived decision to move toward stronger information sources: order flow, on-chain value/network activity, funding/carry, liquidity/capacity, event-time sampling and better validation.

## Important caveats

These figures are **not** final thesis results and should not be annualized or generalized as evidence of a persistent edge because:
- only 1,000 bars were returned in this first run,
- only one chronological holdout split was used,
- CPCV, purging, embargo, PBO and Deflated Sharpe are not yet implemented,
- true order-book slippage/market impact is not modeled,
- funding and borrow costs are absent,
- no on-chain or order-flow inputs are used,
- no forward paper-trading evidence exists yet.

## Decision

**Do not promote v0.1 ML to the final bot.**

Proceed to v0.2 Replication Lab and require every new component to beat:
1. Buy & Hold,
2. simple momentum,
3. this v0.1 ML baseline,
under realistic costs and robust out-of-sample validation.
