# Research-to-Code Traceability Matrix

Date: 2026-09-10

This file connects the consolidated thesis research dossier to executable repository modules and validation artifacts. Engineering completion and empirical validation are reported separately.

| Research phase | Current code/artifact | Status | Acceptance evidence still required |
|---|---|---|---|
| Audit | prior audits + v0.8-v1 regression suites + `audit_v19.py` | ACTIVE / V0.19 VERIFIED | keep audit on each source and preserve original artifacts |
| Dynamic Universe | `market_discovery.py`, `universe.py` | IMPLEMENTED / VALIDATING | historical membership/entity resolution |
| Coverage | `CoverageReport` | IMPLEMENTED PARTIAL | trustworthy market-cap denominator |
| Fast Scanner | `fast_scanner.py` | IMPLEMENTED / VALIDATING | larger-universe throughput artifact |
| Cross-sectional Ranking | `cross_sectional_v06.py`, v0.10/v0.11, corrected v0.19 audit | ACTIVE | broader venue/capacity replication |
| Ichimoku/Pattern | `ichimoku_advanced.py`, v0.14 shadow, v0.18/v0.19 audits | ACTIVE / REGIME HYPOTHESIS NOT REPLICATED | new untouched source/window before any regime gate |
| Derivatives | v0.12 external holdout | EXTERNAL HOLDOUT TESTED — NO INCREMENTAL EDGE | higher-quality independent flow source |
| Microstructure | `microstructure_v13.py`, `forward_microstructure_v19.py`, `microstructure_quality_v19.py` | V0.19 MEASUREMENT INFRASTRUCTURE VERIFIED / FORWARD SAMPLE IMMATURE | >=168 h, >=336 opportunities, >=80% BTC+ETH authorized coverage |
| Multi-venue trade-flow PIT contract | fixed 60s window, unique-venue coverage, staleness/clock-skew gates | IMPLEMENTED / SMOKE VERIFIED | 7-day prospective quality artifact |
| Cost-aware abstention | `decision.py`, `cost_regime_v18.py`, corrected `audit_v19.py` | ECONOMIC RISK REDUCTION OBSERVED / STATISTICAL GATE FAILED | new untouched replication with complete trial registry |
| On-chain | `point_in_time.py` contract | CONTRACT READY | real provider/history |
| Whale | specification | NOT IMPLEMENTED | wallet/entity evidence |
| Fundamental | point-in-time contract | NOT IMPLEMENTED | provider/lag audit |
| Tokenomics | specification | NOT IMPLEMENTED | unlock history |
| Sentiment/News | point-in-time contract | NOT IMPLEMENTED | source pipeline |
| ML Baselines | v0.10-v0.12 | OOS TESTED — NOT PROMOTED | new information/window |
| DL Challengers | gated | GATED | beat simple baseline after cost |
| Regime | `regime.py`, v0.11/v0.12/v0.18/v0.19 | RETROSPECTIVE CONSTRUCTION AUDIT — NOT PROMOTED | fresh pre-registered replication |
| RL | safe contract | GATED | valid env + multi-seed OOS superiority |
| Risk | `risk.py`, v0.14 | IMPLEMENTED / FORWARD OBSERVATION | elapsed paper evidence |
| Portfolio | v0.14 account/positions | PARTIAL | optimizer/capacity study |
| Backtest | `backtest.py`, v0.11/v0.12 | STRONG RESEARCH INFRASTRUCTURE | full trial registry + valid SPA/Reality Check/DSR/PBO on a future candidate |
| Search-aware statistical audit | `evidence_ledger_v17.py`, `trial_registry_v17.py`, `V17_SEARCH_AWARE_STATISTICAL_AUDIT.md` | PROTOCOL IMPLEMENTED | complete return-path registry from next large search cycle |
| Paper Trading | `execution.py`, `forward_paper_v14.py`, `persistence.py`, API | ENGINEERING COMPLETE / FORWARD COLLECTION ACTIVE | minimum elapsed forward window |
| Forward evidence | v0.15 scheduled artifacts | ACTIVE / SAMPLE INSUFFICIENT | >=168 h, >=100 observations, >=10 fills |
| Formal forward review | v0.16 evaluation + defense artifact | IMPLEMENTED / SAMPLE INSUFFICIENT | pre-registered gate + sufficient independent returns |
| Thesis synthesis | v0.17 synthesis + Chapter 4 master draft + v0.18/v0.19 audit record | IMPLEMENTED / UPDATING | final forward evidence refresh before thesis freeze |
| Dashboard | `/dashboard`, `/paper/*` | IMPLEMENTED | refinement only |
| Testnet | execution-mode contract | READINESS GATED | provider/testnet credentials + promoted evidence-backed strategy |
| Live readiness | LIVE fails closed | BLOCKED BY EVIDENCE CONTRACT | search-aware survivor + forward replication + testnet + explicit approval |

## Evidence milestones

- v0.10: purged live-universe OOS tournament; no learned model promoted.
- v0.11: multi-seed robustness/bootstrap/FDR; no learned model promoted; Ichimoku regime dependence became a hypothesis.
- v0.12: external derivatives holdout; decision `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`.
- v0.13: post-v0.12 BTC/ETH true-basis/finer-flow forward collection begins 2026-09-01.
- v0.14/v1.0-rc1: CoinEx BTC/ETH completed-4h paper observer, live depth, independent risk, PostgreSQL persistence, dashboard; LIVE disabled.
- v0.15: immutable scheduled prospective evidence with pre-registered minimums of 168 h, 100 observations and 10 simulated fills.
- v0.16: formal forward aggregation, snapshot-independence filtering, Chapter 4/defense artifact generation; current state `INSUFFICIENT_FORWARD_SAMPLE`.
- v0.17: cross-stage thesis evidence synthesis plus search-aware statistical audit protocol; alpha/profitability/LIVE-readiness remain unproven.
- v0.18 Experiment A: original cost-aware experiment reduced turnover/drawdown but paired bootstrap crossed zero; decision `NO_COST_AWARE_CONVERSION_EVIDENCE`.
- v0.18 Experiment B: originally described external Ichimoku/regime test was later found not to reproduce the v0.11 cross-sectional geometry exactly. It remains immutable historical evidence, but v0.19 corrects its interpretation.
- v0.19 retrospective audit A: refreshed reconstruction with OOF uncertainty and terminal costs observed naive -9.1269% versus cost-aware +0.3350%, but the bootstrap CI `[-0.0002727,+0.0005571]` crossed zero. Label: `REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`.
- v0.19 retrospective audit B: restored v0.11 cross-sectional top-quartile Ichimoku and market-level regime construction on 10 OKX symbols; plain -43.1233% versus conditioned -34.9873%, but Sharpe worsened (-0.671 to -0.956) and the bootstrap CI crossed zero. Label: `RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`.
- v0.19 prospective infrastructure: fixed 60-second PIT trade-flow window, exact quote-notional depth, reported-side semantics, unique-venue counting, configured-universe materialization, provider staleness and cross-venue clock-skew gates. Verified CI run `34474370382`; artifact `10150943098`; digest `sha256:d8b70ebc58c4d2ed59e0590c8c071ef6eb9dab0895d3308392b69ffc250f794a`. Smoke snapshot had 6 observations, 0 provider failures and 2/2 symbols authorized, but does not satisfy the 7-day Phase-Q maturity gate.

## Evidence labels

`NOT_TESTED`, `DATA_UNAVAILABLE`, `UNVERIFIED`, `HYPOTHESIS`, `DISCOVERY_CANDIDATE`, `VALIDATED_OOS`, `EXTERNAL_HOLDOUT_NEGATIVE_RESULT`, `FORWARD_PAPER_HYPOTHESIS`, `SEARCH_AWARE_SURVIVOR`, `FORWARD_REPLICATED`, `V19_MEASUREMENT_INFRASTRUCTURE_VERIFIED_FORWARD_SAMPLE_IMMATURE`.

No module can transition to `VALIDATED_OOS` from a unit test, synthetic smoke run, a single backtest, or paper execution alone. No candidate can transition to `SEARCH_AWARE_SURVIVOR` without a complete trial registry and appropriate correction for search/multiple testing. Negative and inconclusive results are retained as first-class scientific evidence.
