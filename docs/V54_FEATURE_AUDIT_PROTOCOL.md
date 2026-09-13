# v0.54 — Feature Family Audit and Walk-Forward OOS Protocol

Status: **PREREGISTERED RESEARCH GATE / NO EXECUTION AUTHORIZATION**

v0.54 answers one question only: which feature families from v0.53 add reproducible out-of-sample value after realistic trading friction?

## Frozen scope

Universe: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT on CoinEx spot public OHLCV.

Decision timeframe: 1h.
Context timeframe: 4h.
Target: next 1h close-to-close return, measured only after the current 1h bar is closed and available.

Canonical decision clock: `decision_at = decision bar available_at`.
Raw `timestamp` remains bar-open provenance and must never be used as the decision clock for same-bar close features.

## Feature families

1. `SMC_ICT`: causal structure, sweep, displacement, FVG/OB/breaker lifecycle, premium/discount and related observable proxies.
2. `BROOKS`: deterministic Al Brooks proxies.
3. `ICHIMOKU`: causal Tenkan/Kijun plus visible-now and projected-cloud variables without future shifts.
4. `HTF`: 4h features joined only when `4h_available_at <= decision_at`.
5. `BASE`: remaining numeric causal context, where present.

No practitioner label is treated as institutional-intent truth or validated alpha.

## Model and validation

The primary audit uses a fixed Ridge regression pipeline with training-only median imputation and standardization. Complexity is deliberately limited: a feature family must demonstrate value before deep/RL challengers are allowed to exploit it.

Walk-forward:
- expanding training window;
- minimum training rows: 1200 in the real runner;
- test window: 240 rows;
- step: 240 rows;
- horizon: 1 bar;
- purge: at least 1 bar, never below target horizon.

No random shuffling.

## Ablations

For every family:
- `ONLY_<family>`;
- `DROP_<family>`;
- full `ALL` model.

Family incremental value at the base cost is defined as:

`Sharpe(ALL) - Sharpe(DROP_family)`.

This is reported alongside the family-only Sharpe and is not by itself a scientific proof of alpha.

## Cost stress

Every fold and aggregate is evaluated at round-trip assumptions:
- 0 bps;
- 24 bps base;
- 36 bps stress.

A result that only survives at zero cost is not promotable.

## Cross-symbol promotion-candidate rule

A family is only marked `promotion_candidate=true` when all are true:
1. results are available for at least 3 symbols;
2. median `Sharpe(ALL)-Sharpe(DROP_family)` across symbols is positive;
3. the increment is positive on at least 3 symbols.

This label means **candidate for the next research stage only**. It does not authorize paper or live execution.

## Required integrity checks

- `decision_at >= bar_close_at` for every decision row;
- every `*_available_at <= decision_at`;
- no duplicate decision timestamps within a symbol;
- chronological expanding folds;
- purge covers label horizon;
- no non-finite predictions;
- dataset fingerprints recorded;
- failed/blocked symbols retained in the report rather than silently omitted.

## Safety state

- `PAPER_EXECUTION=false`
- `LIVE_EXECUTION=false`
- Kraken sealed
- no order submission path in v0.54
- v0.51 prospective evidence unchanged
- v0.52/v0.53 results are not retroactively rewritten

## Interpretation boundary

A green CI run establishes implementation integrity only. A positive walk-forward result is still exploratory OOS research unless it passes the later frozen model tournament and forward paper validation. Negative findings are retained.
