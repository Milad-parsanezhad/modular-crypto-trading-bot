# Research-to-Code Traceability Matrix

Date: 2026-09-09

This file connects the consolidated thesis research dossier to executable repository modules and validation artifacts. Engineering completion and empirical validation are reported separately.

| Research phase | Current code/artifact | Status | Acceptance evidence still required |
|---|---|---|---|
| Audit | prior audits + v0.8-v1 regression suites | ACTIVE | keep audit on each source |
| Dynamic Universe | `market_discovery.py`, `universe.py` | IMPLEMENTED / VALIDATING | historical membership/entity resolution |
| Coverage | `CoverageReport` | IMPLEMENTED PARTIAL | trustworthy market-cap denominator |
| Fast Scanner | `fast_scanner.py` | IMPLEMENTED / VALIDATING | larger-universe throughput artifact |
| Cross-sectional Ranking | `cross_sectional_v06.py`, v0.10/v0.11 | ACTIVE | broader venue/capacity replication |
| Ichimoku/Pattern | `ichimoku_advanced.py`, v0.14 shadow | ACTIVE / FORWARD OBSERVATION | pre-registered false-breakout study |
| Derivatives | v0.12 external holdout | EXTERNAL HOLDOUT TESTED — NO INCREMENTAL EDGE | new source/window |
| Microstructure | `microstructure_v13.py`, daily CI | FORWARD COLLECTION ACTIVE | sufficient post-2026-09-01 window; liquidation DATA_UNAVAILABLE |
| On-chain | `point_in_time.py` contract | CONTRACT READY | real provider/history |
| Whale | specification | NOT IMPLEMENTED | wallet/entity evidence |
| Fundamental | point-in-time contract | NOT IMPLEMENTED | provider/lag audit |
| Tokenomics | specification | NOT IMPLEMENTED | unlock history |
| Sentiment/News | point-in-time contract | NOT IMPLEMENTED | source pipeline |
| ML Baselines | v0.10-v0.12 | OOS TESTED — NOT PROMOTED | new information/window |
| DL Challengers | gated | GATED | beat simple baseline after cost |
| Regime | `regime.py`, v0.11/v0.12 | ACTIVE DIAGNOSTIC | external pre-registered gate |
| RL | safe contract | GATED | valid env + multi-seed OOS superiority |
| Risk | `risk.py`, v0.14 | IMPLEMENTED / FORWARD OBSERVATION | elapsed paper evidence |
| Portfolio | v0.14 account/positions | PARTIAL | optimizer/capacity study |
| Backtest | `backtest.py`, v0.11/v0.12 | STRONG RESEARCH INFRASTRUCTURE | full trial registry + valid SPA/Reality Check/DSR/PBO on a future candidate |
| Search-aware statistical audit | `evidence_ledger_v17.py`, `V17_SEARCH_AWARE_STATISTICAL_AUDIT.md` | PROTOCOL IMPLEMENTED | complete trial registry and return paths from next search cycle |
| Paper Trading | `execution.py`, `forward_paper_v14.py`, `persistence.py`, API | ENGINEERING COMPLETE / FORWARD COLLECTION ACTIVE | minimum elapsed forward window |
| Forward evidence | v0.15 scheduled artifacts | ACTIVE / SAMPLE INSUFFICIENT | >=168 h, >=100 observations, >=10 fills |
| Formal forward review | v0.16 evaluation + defense artifact | IMPLEMENTED / SAMPLE INSUFFICIENT | pre-registered gate + sufficient independent returns |
| Thesis synthesis | `evidence_ledger_v17.py`, `V17_THESIS_EVIDENCE_SYNTHESIS.md`, `CHAPTER4_RESULTS_MASTER_DRAFT.md` | IMPLEMENTED | final forward evidence refresh before thesis freeze |
| Dashboard | `/dashboard`, `/paper/*` | IMPLEMENTED | refinement only |
| Testnet | execution-mode contract | READINESS GATED | provider/testnet credentials + promoted evidence-backed strategy |
| Live readiness | LIVE fails closed | BLOCKED BY EVIDENCE CONTRACT | search-aware survivor + forward replication + testnet + explicit approval |

## Evidence milestones

- v0.10: purged live-universe OOS tournament; no learned model promoted.
- v0.11: multi-seed robustness/bootstrap/FDR; no learned model promoted.
- v0.12: external derivatives holdout; decision `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.
- v0.13: post-v0.12 BTC/ETH true-basis/finer-flow forward collection begins 2026-09-01.
- v0.14/v1.0-rc1: CoinEx BTC/ETH completed-4h paper observer, live depth, independent risk, PostgreSQL persistence, dashboard; LIVE disabled.
- v0.15: immutable scheduled prospective evidence with pre-registered minimums of 168 h, 100 observations and 10 simulated fills.
- v0.16: formal forward aggregation, snapshot-independence filtering, Chapter 4/defense artifact generation; current state `INSUFFICIENT_FORWARD_SAMPLE`.
- v0.17: cross-stage thesis evidence synthesis plus search-aware statistical audit protocol; current claims remain engineering-operational but alpha/profitability/LIVE-readiness unproven.

## Evidence labels

`NOT_TESTED`, `DATA_UNAVAILABLE`, `UNVERIFIED`, `HYPOTHESIS`, `DISCOVERY_CANDIDATE`, `VALIDATED_OOS`, `EXTERNAL_HOLDOUT_NEGATIVE_RESULT`, `FORWARD_PAPER_HYPOTHESIS`, `SEARCH_AWARE_SURVIVOR`, `FORWARD_REPLICATED`.

No module can transition to `VALIDATED_OOS` from a unit test, synthetic smoke run, a single backtest, or paper execution alone. No candidate can transition to `SEARCH_AWARE_SURVIVOR` without a complete trial registry and appropriate correction for search/multiple testing. Negative results are retained as first-class scientific evidence.
