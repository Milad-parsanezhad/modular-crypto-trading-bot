# v0.8 — Advanced Research-to-Code Integration

**Status:** research branch; `LIVE` execution remains disabled by default.

This phase converts the consolidated 2026 research dossier into enforceable software contracts.  The goal is not to add model complexity for its own sake; the goal is to make every future model obey the same evidence, cost, risk, leakage and execution rules.

## Research principle -> implementation

| Research principle | v0.8 implementation |
|---|---|
| Evidence Before Opinion | `research_bot.contracts.EvidenceStamp`, `SignalEvidence` |
| Point-in-time availability | `research_bot.point_in_time.point_in_time_asof_join` |
| Net Alpha = return - cost - risk - uncertainty | `research_bot.decision.DecisionEngine` |
| No-trade is a valid action | cost/confidence-aware abstention in `DecisionEngine` |
| Independent risk layer | `research_bot.risk.RiskEngine` |
| Drawdown/CVaR/liquidity/turnover gates | `RiskLimits`, `RiskSnapshot`, kill switch |
| Signal != execution | `ResearchTradingOrchestrator` -> `PaperExecutionEngine` |
| Realistic frictions | fee/slippage/funding diagnostics in `backtest.py`; fee/slippage in paper fills |
| Idempotency / partial fills | `PaperExecutionEngine` |
| Live off by default | `ExecutionPolicy` hard guard; no live-order implementation in v0.8 |
| Dynamic universe / eligibility | `research_bot.universe` |
| Coverage must be measured, never guessed | `CoverageReport`; missing market-cap denominator -> `DATA_UNAVAILABLE` |
| Ichimoku is candidate feature, not truth | `ichimoku_advanced.py`; leakage-safe state + candidate detector |
| Triangle under Kumo must be algorithmic | rolling geometry + prior resistance + TK/Kumo/volume/volatility confirmation |
| Chikou future leakage forbidden | Chikou excluded from modelling features |
| Reproducibility | dataframe SHA-256 + `ExperimentManifest` persistence |
| Tail risk matters | VaR/CVaR, profit-factor, exposure and cost diagnostics in `backtest.py` |

## What is deliberately NOT promoted to the core yet

The research dossier finds that complex directional RL, Transformers and LLM trading do not have a right to be core components merely because they are sophisticated.  They remain challengers until they demonstrate stable OOS improvement over simple baselines after fees, slippage, funding, multiple seeds and regime tests.

Likewise, on-chain, whale, fundamental, tokenomics and sentiment are not silently backfilled into historical samples.  Their integration must use `available_at` and point-in-time joins.  If historical availability is unknown, the correct status is `DATA_UNAVAILABLE` or `UNVERIFIED`, not an imputed feature.

## v0.8 decision path

```text
market/universe discovery
    -> eligibility + provenance
    -> point-in-time features
    -> model forecast (any baseline/challenger)
    -> expected cost + uncertainty
    -> cost-aware abstention / candidate
    -> independent risk gate + kill switch
    -> PAPER execution simulator
    -> immutable experiment artifacts
```

## Required next experiments

1. **Dynamic-universe smoke run** on live public market metadata; record discovery/rejection/coverage counts.
2. **Cross-sectional ranking v0.6 -> dynamic universe** with non-overlapping PnL and the same cost model.
3. **Ichimoku ablation:** base features vs base+Ichimoku vs triangle detector; same splits and costs.
4. **Cost sensitivity:** maker/taker, spread, slippage and (for perpetuals) funding scenarios.
5. **Risk ablation:** predictive signal with and without independent risk engine.
6. **Regime stability:** bull/bear/sideways/high-vol/stress reporting.
7. **Multiple testing:** WFV/CPCV + bootstrap + PBO/Deflated Sharpe where strategy multiplicity warrants it.
8. **Forward paper trading:** only after the unified backtest passes leakage and artifact audits.
9. **Testnet:** only after forward paper reconciliation, retry/idempotency and failure drills pass.
10. **LIVE-readiness audit:** separate approval gate; not part of v0.8.

## Acceptance gate for this phase

v0.8 can be called integrated only when:

- all unit tests pass;
- no future-availability join is possible;
- live mode fails closed;
- low net-alpha predictions abstain before execution;
- risk breaches block execution;
- duplicate order IDs are rejected;
- market coverage reports `DATA_UNAVAILABLE` when a valid denominator is missing;
- experiment artifacts record data/model/seed/commit metadata.

No profitability claim is created by passing these engineering tests.
