# Research-to-Code Traceability Matrix

Date: 2026-09-08

This file connects the consolidated thesis research dossier to executable repository modules and validation artifacts.  A phase is not marked complete merely because code exists; it must have tests/artifacts and, where applicable, real OOS evidence.

| Research phase | Research requirement | Current code/artifact | Status | Acceptance evidence still required |
|---|---|---|---|---|
| 1. Audit | Detect leakage, placeholders, unsafe execution | prior audits + v0.8 regression suite | ACTIVE | keep audit on each model/data source |
| 2. Dynamic Universe | Multi-exchange discovery, mapping, provenance, failure isolation | `market_discovery.py`, `universe.py`, live discovery workflow | IMPLEMENTED / VALIDATING | historical/data-quality eligibility + entity resolution |
| 3. Coverage | Measured coverage; never invent denominator | `CoverageReport` | IMPLEMENTED PARTIAL | trustworthy market-cap snapshot and top-100 denominator |
| 4. Fast Scanner | Scalable technical eligibility/opportunity scanner | `fast_scanner.py` | IMPLEMENTED / VALIDATING | real-universe throughput and stability artifact |
| 5. Cross-sectional Ranking | Multi-asset ranking and non-overlapping PnL | `cross_sectional_v06.py` | PROTOTYPE | connect dynamic eligible universe; rerun WFV/CPCV |
| 6. Ichimoku/Pattern | Feature family + algorithmic triangle under Kumo; no Chikou leakage | `ichimoku_advanced.py` | IMPLEMENTED | OOS ablation and false-breakout study |
| 7. Derivatives | Funding/basis/OI/order flow/liquidation, multi-provider | v0.3 + v0.5 research modules | PARTIAL | trustworthy historical OI/liquidations + provider fallback |
| 7. On-chain | Point-in-time network features | `point_in_time.py` contract | CONTRACT READY | real provider + historical `available_at` data |
| 7. Whale | Labelled transfer intent with uncertainty | research specification only | NOT IMPLEMENTED | wallet/entity mapping + point-in-time evidence |
| 8. Fundamental | Fees/revenue/TVL/users/dev/security point-in-time | point-in-time contract only | NOT IMPLEMENTED | provider connectors + definitions + lag audit |
| 8. Tokenomics | supply/vesting/unlocks/dilution | research specification only | NOT IMPLEMENTED | trustworthy unlock history and availability timestamps |
| 8. Sentiment/News | dedup/event-time/freshness/entity resolution | point-in-time contract only | NOT IMPLEMENTED | source pipeline and anti-leakage audit |
| 9. ML Baselines | logistic/tree models before complex challengers | HGB/logistic/cross-sectional baselines | ACTIVE | unified tournament, calibration/Brier by regime |
| 9. DL Challengers | LSTM/GRU/TCN/Transformer only if OOS justified | not promoted to core | GATED | beat simple baseline after cost across folds/regimes |
| 10. Regime | trend/vol/liquidity/funding/correlation stress | `regime.py` + prior ablations | PARTIAL | multidimensional regime validation |
| 11. RL | sequential decision only with seed/cost/risk comparison | safe orchestration contract; old pseudo-RL rejected | GATED | valid env/reward + multi-seed OOS superiority or execution-only role |
| 12. Risk | independent exposure/DD/CVaR/liquidity/kill switch | `risk.py` | IMPLEMENTED | portfolio-level ablation on real OOS runs |
| 12. Portfolio | correlation/cluster/cash/turnover/leverage constraints | partial research code | PARTIAL | unified multi-asset optimizer and capacity study |
| 13. Backtest | unified cost-aware WFV/CPCV/DSR/PBO/ablation | enhanced `backtest.py` + prior robustness modules | ACTIVE | dynamic-universe end-to-end artifact |
| 14. Paper Trading | real-time simulation, persistence, reconciliation | `execution.py`, `orchestrator.py`, research API | ENGINEERING READY | forward-market feed + restart/reconciliation tests |
| 15. Dashboard | market/scanner/risk/backtest/model/coverage/health | existing Railway `web` service outside this core | PARTIAL | connect stable v0.8 API/artifacts |
| 16. Testnet | exchange test environment, failure drills | execution mode contract | NOT STARTED | exchange adapter + testnet credentials + reconciliation |
| 16. Live readiness | security, secret handling, no withdrawal, kill switch | LIVE fails closed | BLOCKED BY DESIGN | all prior gates + explicit approval |

## Mandatory thesis comparisons

- Buy & Hold BTC
- Buy & Hold ETH
- Equal-weight crypto basket
- Simple momentum
- Ichimoku-only
- ML-only
- RL-only (only if a valid RL implementation reaches evaluation)
- Full system

## Mandatory ablations

- without Ichimoku
- without derivatives/whale/on-chain/fundamental/sentiment/tokenomics when those modules exist
- without regime gate
- without RL
- without dynamic risk
- cost sensitivity
- point-in-time/freshness audit

## Evidence labels

`NOT_TESTED`, `DATA_UNAVAILABLE`, `UNVERIFIED`, `HYPOTHESIS`, `DISCOVERY_CANDIDATE`, `VALIDATED_OOS`.

No module can transition to `VALIDATED_OOS` from a unit test, synthetic smoke run, a single backtest, or paper execution alone.
