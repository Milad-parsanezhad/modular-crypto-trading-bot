# Modular Crypto Trading Bot — Evidence-Driven Research Platform

Academic, modular cryptocurrency trading research platform for building and validating a multi-market intelligent trading system.

> **Current branch:** `research-v08-advanced-integration`  
> **Safety:** `LIVE` execution is disabled by default.  
> **Scientific status:** this repository does **not** claim a validated profitable strategy. Negative OOS results are preserved.

## Core research contract

**Evidence Before Opinion.** A trading claim is not accepted unless it has point-in-time data, provenance, reproducible artifacts, realistic costs and out-of-sample validation.

The platform now follows this path:

**Dynamic Universe → Eligibility/Coverage → Point-in-Time Features → Alpha/Ranking → Regime → Model Tournament → Cost/Uncertainty-Aware Abstention → Independent Risk → Cost-Aware Execution → Robust Validation → Paper Trading → Testnet → Live-Readiness Audit**

Ichimoku is a candidate feature family and must earn its place through ablation. RL/Transformers/LLMs are challengers, not default winners.

## What exists today

### Existing research engine (v0.1–v0.6)
- public OHLCV and derivatives ingestion,
- BTC/USDT 4h real-market baselines,
- leakage-safe technical/liquidity features,
- Ichimoku candidate features,
- CUSUM + triple-barrier replication,
- purged/walk-forward validation,
- funding/basis/order-flow research,
- regime-conditional ablations,
- cross-sectional research prototype,
- CPCV/PBO/Deflated-Sharpe diagnostics,
- GitHub Actions and Colab notebooks.

### v0.8 advanced research integration
- `research_bot/contracts.py` — evidence/status/decision/execution contracts,
- `research_bot/point_in_time.py` — availability-time as-of joins for on-chain/news/fundamental data,
- `research_bot/universe.py` — dynamic eligibility, rejection reasons, deterministic dedup and measured coverage,
- `research_bot/ichimoku_advanced.py` — leakage-safe Ichimoku state and algorithmic triangle-under-Kumo candidate detector,
- `research_bot/decision.py` — Net-Alpha decision engine with cost/risk/uncertainty-aware abstention,
- `research_bot/risk.py` — independent drawdown/exposure/liquidity/turnover/CVaR risk gate and kill switch,
- `research_bot/execution.py` — guarded PAPER/BACKTEST/TESTNET simulator with fees, slippage, partial fills and idempotency,
- `research_bot/orchestrator.py` — forecast → decision → risk → execution integration,
- `research_bot/reproducibility.py` — dataset fingerprinting and experiment manifests,
- richer backtest diagnostics including VaR/CVaR, profit factor, exposure, turnover and explicit/funding costs.

See `docs/V08_ADVANCED_RESEARCH_INTEGRATION.md` for the research-to-code mapping.

## Scientific findings already recorded

The project has intentionally retained negative findings: a simple directional ML baseline did not outperform its benchmark in the initial real-market window, and CUSUM + triple-barrier labelling did not create alpha by itself. These findings changed the project toward cross-sectional ranking, derivatives/order-flow evidence, stronger validation and explicit execution/risk gates.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

## v0.8 engineering smoke test

```bash
python scripts/run_research_core_v08.py --output-dir artifacts/v08
```

Without `--input-csv`, this command uses deterministic **synthetic smoke data only to test engineering contracts**. Synthetic results are never valid thesis performance evidence.

With a real OHLCV CSV:

```bash
python scripts/run_research_core_v08.py \
  --input-csv path/to/real_ohlcv.csv \
  --fee-bps 10 \
  --slippage-bps 2 \
  --output-dir artifacts/v08-real
```

Even a real single CSV run is not considered validated until it passes the full WFV/CPCV, regime, multiple-testing, cost and untouched-test protocol.

## Required validation ladder

1. Hypothesis registration
2. Timestamp/lineage audit
3. Leakage audit
4. Purged WFV/CPCV
5. Simple baselines
6. Train/validation-only tuning
7. Multiple seeds for stochastic models
8. Fees/spread/slippage/funding
9. Liquidity/capacity checks
10. Regime stability
11. Bootstrap confidence intervals
12. PBO / Deflated Sharpe when strategy multiplicity exists
13. Ablation
14. Untouched final test
15. Forward paper trading
16. Testnet
17. Live-readiness audit

## Execution safety

Predictive models never call an exchange directly. Signals must pass through cost/uncertainty abstention, an independent risk engine and an execution adapter. v0.8 contains no live-order implementation; the paper engine fails closed when `LIVE` is requested without explicit readiness.

## Immediate next work

- run dynamic-universe discovery with real public market metadata and persist coverage/rejection artifacts,
- connect v0.6 cross-sectional ranking to the dynamic universe,
- run Ichimoku and risk ablations under identical OOS/cost assumptions,
- add point-in-time derivatives/on-chain/whale/fundamental connectors only where historical availability is trustworthy,
- stabilize forward paper execution and reconciliation,
- build the research/dashboard layer after backend artifacts are stable.

## v0.17 frozen Ichimoku Strategy Lab

The repository now includes a causal 4H comparison of B0, S1-S6, IRCP, C1 and C2 plus the IRGC-S event/meta-label candidate. Run:

```bash
pytest -q tests/test_strategy_lab_v17.py tests/test_binance_spot_archive_v17.py tests/test_forward_candidate_v17.py
PYTHONPATH=. python scripts/run_strategy_lab_v17.py --archive-cache data/cache --bars 20000 --output-dir artifacts/v17-strategy-lab
PYTHONPATH=. python scripts/run_forward_candidate_v17.py --archive-cache data/cache --symbol BTCUSDT
```

See `docs/V17_ICHIMOKU_STRATEGY_LAB_PROTOCOL.md` and the dated v0.17 results report. The S6 adapter emits observation-only paper candidates with a 0.25% risk budget, 35% per-asset cap, 70% portfolio-gross cap and 5% drawdown kill switch. Results remain research-only and cannot enable live execution.

## v0.20 causal ICT/M1 extraction lab

The uploaded ICT/TTrades/M1 educational rules are now represented by a
closed-bar state machine: confirmed swing -> liquidity sweep -> close-confirmed
MSB -> origin return -> structural stop/3R target. Three origin definitions are
pre-registered and compared under identical costs. Run:

```bash
pytest -q tests/test_ict_m1_v20.py
PYTHONPATH=. python scripts/run_ict_m1_lab_v20.py \
  --archive-cache ../data/cache --symbol BTCUSDT --timeframe 4h \
  --ichimoku-gate trend --output-dir artifacts/v20-ict-m1-btc
```

The 2020-2025 BTC replication rejected all three simple origin variants after
costs. Small positive ETH cells remain insufficient and are not promotion
evidence. See `docs/V20_ICT_M1_STRATEGY_PROTOCOL_AND_RESULTS.md`. Live execution
remains disabled.

[Run the frozen v0.20 replication in Google Colab](https://colab.research.google.com/github/parsa314/modular-crypto-trading-bot/blob/research-v20-ict-m1/notebooks/Research_Bot_v0_20_ICT_M1_Colab.ipynb).
