# v0.19 — Source-Derived Multi-Timeframe Strategy Tournament

Date: 2026-09-10  
Status: **research protocol; LIVE execution remains prohibited**

## Objective

Convert the newly supplied trading books/notes plus the project's existing causal Ichimoku research into a frozen, testable strategy registry. The goal is not to declare a book setup profitable. The goal is to translate source concepts into explicit crypto research proxies, run them under one execution/cost/risk contract, and select at most one candidate for **forward PAPER observation** after a validation-only ranking and untouched-test gate.

## Source-to-code map

- **M1Trades System**: higher-timeframe context (Weekly/Daily/H4/H1) and lower-timeframe refinement (M5/M1); Sweep -> MSB -> Origin RTO; HTF supply/demand/origins; London/New York context; 3R target and defensive stop management.
- **ORDER BLOCKS**: a valid order block requires a preceding BOS; HTF OBs are refined on lower timeframes; confirmed entry waits for a reaction in the HTF OB, then a lower-timeframe BOS and lower-timeframe OB.
- **Supply & Demand**: DBD/DBR/RBR/RBD zones, proximal/distal construction, and explicit multi-timeframe mappings such as 4H/1H analysis -> 15m entry.
- **Unicorn Model**: liquidity sweep (or SMT), breaker block, FVG overlap, and a clear draw on liquidity.
- **TTrades**: New-York-time kill zones, OTE/premium-discount, MSS/CISD, FVG/IFVG, internal/external liquidity, Silver Bullet, AMD/Judas concepts, OSOK/fractal logic and SMT.
- **ICT Mentorship Handbook**: Big Picture -> Intermediate -> Short-Term decomposition, PD arrays, monthly/weekly/daily/H4/H1 sequence, swing/day-trading models, time-and-price and session conditioning.
- **Advanced Technical Analysis**: institutional-candle/supply-demand context, correlation divergence, open-interest context, Fibonacci retracement logic, and psychological whole/quarter price levels.
- **Project v0.17/v0.18 evidence**: causal Ichimoku, S6 breakout, triangle-under-Kumo, cost-aware/no-trade philosophy, and fail-closed promotion.

Several texts were written for FX or index futures. Their crypto versions are explicitly labelled **research adaptations**, not verbatim source rules. Fixed pip rules are never copied into crypto. They are translated into ATR/volatility-normalized constraints. Open-interest/COT concepts are not backfilled into spot OHLCV; they remain a separate derivatives-data experiment.

## Frozen registry — 30 candidates

Five candidates are registered for each primary timeframe.

| Timeframe | Candidates |
|---|---|
| 1m | `M1_ATM_ORIGIN_RTO`, `M1_CONFIRMED_ORDER_BLOCK`, `M1_UNICORN`, `M1_FVG_RETRACE`, `M1_FRACTAL_CISD` |
| 5m | `M5_ATM_ORIGIN_RTO`, `M5_UNICORN`, `M5_OTE_PD_ARRAY`, `M5_JUDAS_AMD`, `M5_EXTERNAL_INTERNAL_LIQ` |
| 15m | `M15_SUPPLY_DEMAND_MTF`, `M15_CONFIRMED_ORDER_BLOCK`, `M15_SILVER_BULLET`, `M15_OTE_PD_ARRAY`, `M15_ROUND_NUMBER_SWEEP` |
| 1h | `H1_SUPPLY_DEMAND_MTF`, `H1_OB_BOS_RETEST`, `H1_FIB_INSTITUTIONAL_RETRACE`, `H1_CORRELATION_DIVERGENCE`, `H1_ICHIMOKU_PULLBACK` |
| 4h | `H4_S6_BREAKOUT`, `H4_KUMO_TRIANGLE`, `H4_OB_BOS_RETEST`, `H4_SUPPLY_DEMAND`, `H4_CORRELATION_DIVERGENCE` |
| 1d | `D1_ICT_SWING_PD`, `D1_OB_BOS_RETEST`, `D1_SUPPLY_DEMAND`, `D1_FIB_INSTITUTIONAL`, `D1_CORRELATION_DIVERGENCE` |

This registry is frozen before the public-data tournament. Sparse strategies are **not** loosened merely to reach the requested sample count. A strategy that cannot produce enough legitimate observations fails the sample gate rather than being made artificially more active.

## Causal execution contract

- Decisions use only completed bars.
- Entry is at `open[t+1]` after a signal at `close[t]`.
- Swing points are one-bar-confirmed pivots; no centered/future pivot is used.
- Rolling breakout levels exclude the current candle.
- Ichimoku Senkou information is represented at the decision timestamp; plotting displacement is never reversed into future information.
- Correlated-asset data are joined backward/as-of only.
- Fee = **10 bps one way** and slippage = **2 bps one way** (24 bps base round trip).
- Account risk budget = **0.25% per trade**; default target = **3R**.
- Stops are ATR-normalized by timeframe, never copied FX pip constants.
- One trade per symbol/candidate at a time; no martingale or averaging down.
- If stop and target are touched in the same candle, **stop is assumed first**.
- Promotion maximum-drawdown gate = **5%**.
- LIVE execution remains disabled.

## Data and 1000+ trade contract

Default CoinEx public spot OHLCV history requested per symbol:

| Timeframe | Bars/symbol |
|---|---:|
| 1m | 30,000 |
| 5m | 24,000 |
| 15m | 16,000 |
| 1h | 12,000 |
| 4h | 8,000 |
| 1d | 3,000 |

Default universe: 12 liquid crypto assets. Missing histories are recorded in provenance, never imputed.

The requested **1000+ trade** standard is a frozen pre-test minimum: a candidate requires at least **1,000 closed development+validation trades** before its final test can qualify. It also needs at least **200 untouched-test trades**. Higher-timeframe or sparse candidates that cannot honestly reach that depth are not artificially loosened.

## Selection protocol

Chronological split: 60% development, 20% validation, 20% untouched test.

Pre-test eligibility requires: >=1000 pre-test trades, positive validation expectancy in R, validation profit factor >=1.05, >=60% positive assets, and validation MDD <=5%. Candidates are ranked **only on validation**. The top eligible candidate then faces the untouched test once.

`FORWARD_PAPER_CANDIDATE` additionally requires >=200 test trades, positive test expectancy, test PF >=1.05, >=60% positive assets, test MDD <=5%, and a positive lower bound for a moving-block-bootstrap 95% CI of mean account return.

Passing does **not** replace production automatically and does not authorize LIVE. Thirty candidates also create multiple-testing risk; any survivor must next pass the project's search-aware PBO/DSR/SPA/Reality-Check controls and fresh forward-paper replication.

## Reproduction

```bash
python -m pip install -e '.[dev]'
pytest -q tests/test_multitimeframe_strategies_v19.py
PYTHONPATH=. python scripts/run_v19_multitimeframe_tournament.py \
  --min-pretest-trades 1000 \
  --min-test-trades 200 \
  --output-dir artifacts/v19-multitimeframe
```

Artifacts: `strategy_registry.csv`, `strategy_summary.csv`, `trade_ledger.csv`, and `decision.json`.
