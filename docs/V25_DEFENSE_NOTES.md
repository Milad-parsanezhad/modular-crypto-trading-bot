# v0.25 Defense Notes — Signal Alpha vs Portfolio Alpha

## Thesis-level question

Why can an event-level ML filter show PF > 1 on two external venues while the portfolio still fails?

## Defense answer

Because a trading system is a constrained decision system, not an IID classifier. When multiple selected events arrive simultaneously, limited capital/risk budgets force an admission decision. The v0.24d filter answered **which events appear admissible**, but the portfolio engine still needed a rule for **which admissible events receive scarce risk first**. Deterministic strategy/symbol ordering is reproducible but not economically optimized.

v0.25 therefore decomposes the problem into:

1. **event filter** — frozen v0.24b binary models;
2. **portfolio ranker** — relative priority among simultaneous eligible events;
3. **risk engine** — correlation, CVaR, open-risk and drawdown constraints;
4. **future evidence** — no-retune prospective evaluation;
5. **RL allocator** — only a later challenger after simpler ranking baselines survive.

## Why this is scientifically stronger than immediately using RL

- the hypothesis is smaller and falsifiable;
- the incremental contribution of ranking can be ablated against frozen-score ordering;
- lower model complexity reduces the search surface;
- transaction costs and turnover can be measured before introducing an adaptive policy;
- a negative result is interpretable: either event scores are not rank-useful or portfolio constraints dominate them.

## Key bias control

The OKX/KuCoin v0.24d sample created this hypothesis and is therefore **spent**. It may diagnose the allocator but cannot validate it. The first promotable evidence for v0.25 must occur after the frozen future boundary.

## Claim boundary

A validation-selected ranker is a `CHALLENGER`, not alpha. A future-time pass is still not LIVE authorization. PAPER replacement additionally requires the project’s search-aware/statistical and prospective observation gates.
