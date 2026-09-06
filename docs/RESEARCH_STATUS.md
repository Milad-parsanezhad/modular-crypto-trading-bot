# Research Status — 2026-09-06

## Conclusion reached so far
The research phase has reached a **design conclusion**, not a profitability conclusion. The strongest evidence supports testing a multi-alpha, regime-adaptive architecture rather than privileging Ichimoku, LSTM, Transformer or RL in advance.

### Core candidate engines
1. Order flow / microstructure.
2. On-chain value / network activity.
3. Carry / funding / basis.
4. Momentum / liquidity / cross-sectional interactions.
5. Multi-dimensional regime state.
6. Cost-aware abstention.
7. RL primarily for execution/allocation as a challenger.

## Current repository stage
**Stage 1 complete:** research framing + evidence map.

**Stage 2 started:** reproducible baseline implementation and real public market-data execution.

### v0.1 asks
- Can point-in-time market data be fetched reproducibly?
- Do simple baselines survive fees/slippage?
- Does a tree model add value beyond Buy & Hold and momentum?
- Does abstention improve net performance by reducing turnover?

### v0.1 cannot yet prove
- final profitability,
- robustness to CPCV/PBO/DSR,
- survival under true order-book impact,
- incremental on-chain/order-flow/carry alpha,
- survival in forward paper trading.

## Next gate (v0.2)
Replication Lab: CUSUM/event-time sampling, triple-barrier labels, purged/embargoed CV, CPCV, DSR/PBO, funding/carry, point-in-time on-chain, order flow/LOB and full feature-family ablations including Ichimoku.
