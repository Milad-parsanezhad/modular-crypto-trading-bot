# Research-to-Code Traceability Matrix

Date: 2026-09-09

This file connects the consolidated thesis research dossier to executable repository modules and validation artifacts. A phase is not marked complete merely because code exists; it must have tests/artifacts and, where applicable, real OOS evidence.

| Research phase | Research requirement | Current code/artifact | Status | Acceptance evidence still required |
|---|---|---|---|---|
| 1. Audit | Detect leakage, placeholders, unsafe execution | prior audits + v0.8-v0.12 regression suites | ACTIVE | keep audit on each model/data source |
| 2. Dynamic Universe | Multi-exchange discovery, mapping, provenance, failure isolation | `market_discovery.py`, `universe.py`, v0.9/v0.10 live workflows | IMPLEMENTED / VALIDATING | historical membership/entity resolution |
| 3. Coverage | Measured coverage; never invent denominator | `CoverageReport` | IMPLEMENTED PARTIAL | trustworthy market-cap snapshot and top-100 denominator |
| 4. Fast Scanner | Scalable technical eligibility/opportunity scanner | `fast_scanner.py`, v0.9 live scanner | IMPLEMENTED / VALIDATING | larger-universe throughput/stability artifact |
| 5. Cross-sectional Ranking | Multi-asset ranking and non-overlapping PnL | `cross_sectional_v06.py`, v0.10/v0.11 tournaments | ACTIVE | broader time/venue replication and capacity study |
| 6. Ichimoku/Pattern | Feature family + algorithmic triangle under Kumo; no Chikou leakage | `ichimoku_advanced.py`, v0.10-v0.12 ablations | ACTIVE | dedicated triangle/false-breakout OOS study |
| 7. Derivatives | Funding/basis/OI/order flow/liquidation, multi-provider | v0.3/v0.4/v0.6 + `derivatives_ablation_v12.py` | **EXTERNAL HOLDOUT TESTED — NO INCREMENTAL EDGE** | independent source/window; true basis; liquidation/finer microstructure |
| 7. On-chain | Point-in-time network features | `point_in_time.py` contract | CONTRACT READY | real provider + historical `available_at` data |
| 7. Whale | Labelled transfer intent with uncertainty | research specification only | NOT IMPLEMENTED | wallet/entity mapping + point-in-time evidence |
| 8. Fundamental | Fees/revenue/TVL/users/dev/security point-in-time | point-in-time contract only | NOT IMPLEMENTED | provider connectors + definitions + lag audit |
| 8. Tokenomics | supply/vesting/unlocks/dilution | research specification only | NOT IMPLEMENTED | trustworthy unlock history and availability timestamps |
| 8. Sentiment/News | dedup/event-time/freshness/entity resolution | point-in-time contract only | NOT IMPLEMENTED | source pipeline and anti-leakage audit |
| 9. ML Baselines | logistic/tree models before complex challengers | v0.10-v0.12 Logistic/HGB/RF comparisons | **OOS TESTED — NOT PROMOTED** | new information source/window; calibration/decision objective work |
| 9. DL Challengers | LSTM/GRU/TCN/Transformer only if OOS justified | not promoted to core | GATED | beat simple baseline after cost across folds/regimes |
| 10. Regime | trend/vol/liquidity/funding/correlation stress | `regime.py`, v0.11/v0.12 diagnostics | ACTIVE DIAGNOSTIC | pre-registered external regime-gate validation; no post-hoc tuning on spent holdouts |
| 11. RL | sequential decision only with seed/cost/risk comparison | safe orchestration contract; old pseudo-RL rejected | GATED | valid env/reward + multi-seed OOS superiority or execution-only role |
| 12. Risk | independent exposure/DD/CVaR/liquidity/kill switch | `risk.py` | IMPLEMENTED | portfolio-level ablation on promoted candidate only |
| 12. Portfolio | correlation/cluster/cash/turnover/leverage constraints | partial research code | PARTIAL | unified multi-asset optimizer and capacity study |
| 13. Backtest | unified cost-aware WFV/holdout/bootstrap/FDR/ablation | `backtest.py`, `robustness_v11.py`, `derivatives_ablation_v12.py` | ACTIVE / STRONGER EVIDENCE | CPCV/PBO/DSR on any future candidate before promotion |
| 14. Paper Trading | real-time simulation, persistence, reconciliation | `execution.py`, `orchestrator.py`, research API | ENGINEERING READY / RESEARCH GATED | forward candidate must first pass OOS evidence gate |
| 15. Dashboard | market/scanner/risk/backtest/model/coverage/health | existing Railway `web` service outside this core | PARTIAL | connect stable research API/artifacts |
| 16. Testnet | exchange test environment, failure drills | execution mode contract | NOT STARTED | exchange adapter + testnet credentials + reconciliation |
| 16. Live readiness | security, secret handling, no withdrawal, kill switch | LIVE fails closed | BLOCKED BY DESIGN | all prior gates + explicit approval |

## Evidence milestones

### v0.10
- Live eligible-universe purged OOS tournament.
- No learned model promoted.
- Ichimoku baseline strong in aggregate but regime/fold unstable.

### v0.11
- Multi-seed robustness, moving-block bootstrap, FDR and regime diagnostics.
- No model promoted.
- Strong evidence that v0.10 aggregate performance was not a stable learned-model edge.

### v0.12
- Pre-registered external USD-M derivatives holdout.
- 12/12 frozen symbols usable; 1,094 synchronized 8h timestamps.
- Price / Ichimoku / derivatives / full ablation with Logistic and HGB.
- Untouched holdout: 2026-06-01 16:00 UTC to 2026-08-31 16:00 UTC.
- Best full variant (Logistic) net return -5.85%, Sharpe -0.298.
- Price-only Logistic net return -5.19%, Sharpe -0.192.
- Full vs price-only bootstrap/FDR did not show incremental edge.
- Decision: `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.
- Paper/testnet/live remain closed.

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

`NOT_TESTED`, `DATA_UNAVAILABLE`, `UNVERIFIED`, `HYPOTHESIS`, `DISCOVERY_CANDIDATE`, `VALIDATED_OOS`, `EXTERNAL_HOLDOUT_NEGATIVE_RESULT`.

No module can transition to `VALIDATED_OOS` from a unit test, synthetic smoke run, a single backtest, or paper execution alone. Negative external-holdout results are retained as first-class scientific evidence and are not tuned away on the spent holdout.
