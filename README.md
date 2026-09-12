# Modular Crypto Trading Bot — Evidence-Driven Thesis Research System

Private academic cryptocurrency trading-research repository for a causal, reproducible and fail-closed thesis bot.

> **Current research line:** `v0.41` (`research/v41-event-competing-risk`)  
> **Mother strategy:** `v0.39`  
> **Execution state:** `RESEARCH_ONLY`  
> **Kraken holdout:** `SEALED`  
> **PAPER:** disabled  
> **LIVE:** disabled  
> **Scientific status:** no validated profitable strategy is claimed unless every frozen development/holdout gate passes.

## Core principle

**Evidence Before Opinion.**

A strategy claim is admissible only when it is based on causal point-in-time inputs, reproducible code/artifacts, realistic costs, purged out-of-sample evaluation, uncertainty accounting and explicit promotion gates. Technical CI success never equals alpha evidence.

## Current end-to-end architecture

```text
Market data
  -> causal feature engineering / robust normalization
  -> independent mother-strategy engines
       ICT
       SMC
       Ichimoku
       Al Brooks-inspired price action
       higher-timeframe context
  -> broad mother-event pool
  -> event-family / side / regime representation
  -> statistical learning layer
  -> expected post-cost R + uncertainty + duration/path information
  -> independent financial risk governor
  -> portfolio admission / drawdown firewall
  -> frozen OOS characterization
  -> evidence artifacts / thesis results
```

The strategy engines are intentionally independent. The system does **not** require a brittle hard conjunction such as “6 of 7 conditions.” Their outputs become causal features/events and must earn predictive value out of sample.

## Research lineage that matters now

### v0.39 — Robust mother strategy and financial system

Reconstructed and integrated:

- ICT: liquidity sweep, MSS/CHoCH/BOS, displacement and premium/discount context;
- SMC: structure, FVG, order-block mitigation and supply/demand context;
- Ichimoku: Kumo, Tenkan/Kijun, Kijun slope, breakout and pullback context;
- Al Brooks-inspired engine: Always-In, trend/range, breakout/failed breakout, H1/H2, L1/L2, wedge, micro-double, signal-bar, follow-through and measured-move context;
- causal higher-timeframe context;
- robust rolling normalization;
- model complexity ladder and fixed-seed policy;
- independent risk governor and experiment loop guard.

v0.39 repaired the v0.39R zero-event problem but did **not** produce a development-qualified alpha model.

### v0.40 — Two-stage hurdle experiment

Separated:

1. probability an event has positive post-cost R;
2. positive payoff magnitude;
3. loss magnitude;
4. holding duration;
5. regime-aware conformal uncertainty.

Result: the current pooled mother-event representation did not show useful OOS Stage-1 discrimination. This result is preserved as negative scientific evidence in `docs/V40_RESULTS_2026-09-12.md`.

### v0.41 — Event-specific competing-risk experiment

Current hypothesis:

- freeze semantic event families;
- model LONG and SHORT separately;
- estimate discrete-time cause-specific TARGET and STOP hazards;
- treat TIME as right-censoring to the frozen horizon;
- reconstruct cumulative incidence for target/stop/timeout;
- use family-aware conformal expected-R bounds;
- admit capital only when the conservative lower expected-R is positive and target probability exceeds stop probability.

This experiment does **not** relax v0.40 thresholds after seeing outcomes and does not touch Kraken.

## Financial governance

Frozen core limits inherited by the current research line:

- base stop-risk per trade: **0.25% equity**;
- max stop-risk per trade: **0.50%**;
- aggregate open stop-risk cap: **2.00%**;
- same-direction open stop-risk cap: **1.50%**;
- max nominal asset weight: **35%**;
- max portfolio gross: **70%**;
- drawdown risk reduction around **2.0%** and **3.5%**;
- hard drawdown firewall: **5.0%**;
- base round-trip cost: **24 bps**;
- stress round-trip cost: **36 bps**;
- no martingale;
- no averaging down;
- no revenge-risk increase;
- no best-seed cherry-picking;
- no post-result threshold rescue under the same experiment ID.

The predictive model never directly owns capital. Capital is admitted only after the independent financial governor accepts the prediction, uncertainty and portfolio state.

## Install

```bash
python -m pip install -e ".[dev]"
```

Run the fail-closed health check:

```bash
modular-crypto-bot doctor
```

Expected safety state includes:

```text
RESEARCH_ONLY
Kraken = SEALED
PAPER = false
LIVE = false
```

## Unified CLI

```bash
modular-crypto-bot status
modular-crypto-bot manifest --version v41
modular-crypto-bot doctor
```

Build the causal mother features from an OHLCV CSV:

```bash
modular-crypto-bot features \
  --input path/to/ohlcv.csv \
  --output results/features.csv
```

Run a frozen characterization:

```bash
modular-crypto-bot characterize \
  --version v41 \
  --output-dir results/v41_characterization
```

The package can also be invoked with:

```bash
python -m research_bot
```

There is deliberately **no** CLI command for LIVE orders.

## Development universe and validation

Consumed development venues:

- CoinEx
- OKX
- KuCoin

Reserved external holdout:

- Kraken — **SEALED until a preregistered development winner passes every frozen gate**

Current primary research universe:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- DOGE/USDT
- 4-hour decision timeframe

The current protocol uses purged chronological folds, embargo, independent venue accounting, realistic cost stress, moving-block bootstrap confidence intervals, asset breadth, quarter persistence, seed stability and max-drawdown gates.

## Key files

Mother strategy and finance:

- `research_bot/mother_strategy_v39.py`
- `research_bot/brooks_engine_v39.py`
- `research_bot/financial_system_v39.py`

v0.40:

- `research_bot/two_stage_hurdle_v40.py`
- `docs/V40_TWO_STAGE_HURDLE_PREREGISTRATION.md`
- `docs/V40_RESULTS_2026-09-12.md`

v0.41:

- `research_bot/event_competing_risk_v41.py`
- `research_bot/event_competing_risk_vectorized_v41.py`
- `scripts/run_v41_competing_risk_characterization.py`
- `scripts/run_v41_competing_risk_characterization_fast.py`
- `docs/V41_EVENT_COMPETING_RISK_PREREGISTRATION.md`
- `research_bot/cli.py`

Tests and CI:

- `tests/test_event_competing_risk_v41.py`
- `tests/test_v41_vectorized_equivalence.py`
- `tests/test_cli_v41.py`
- `.github/workflows/v41-competing-risk-ci.yml`
- `.github/workflows/v41-competing-risk-characterization.yml`

## Scientific promotion policy

A green CI run proves implementation integrity only. Promotion requires the preregistered economic/robustness gates to pass across every development venue. Only then may the sealed external holdout be evaluated. PAPER and LIVE remain fail-closed until a later explicit promotion protocol authorizes them.

This repository intentionally preserves failed hypotheses and negative results because they are part of the thesis evidence, not files to be hidden or rewritten.