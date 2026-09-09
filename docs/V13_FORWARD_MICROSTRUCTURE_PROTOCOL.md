# v0.13 Forward Microstructure Collection Protocol

Date frozen: 2026-09-09

## Purpose

v0.12 spent its final holdout through 2026-08-31 and found no sufficient incremental derivatives edge. v0.13 therefore does not tune the v0.12 holdout. It starts a new post-v0.12 evidence window on 2026-09-01.

## Frozen sources and features

Public Binance Vision daily archives, no credentials:

- synchronized completed 1h spot and USD-M perpetual klines;
- true spot-perpetual basis reconstructed directly from synchronized closes;
- separate spot/futures taker-buy quote imbalance at 1h;
- order-flow spread between futures and spot;
- daily USD-M open-interest metrics with conservative +1 UTC day availability shift.

Liquidation history is explicitly `DATA_UNAVAILABLE` in this implementation. It is not proxied or invented.

## Symbols

Initial forward cohort:

- BTCUSDT
- ETHUSDT

The cohort is intentionally small because this phase is evidence collection, not another large hyperparameter search.

## Evidence contract

v0.13 produces descriptive coverage and diagnostic Spearman correlations only.

It does **not**:

- promote a model;
- generate a BUY/SELL authorization;
- retune v0.12;
- use future observations before their availability time;
- claim L2/L3 microstructure from kline taker-flow fields.

The workflow runs daily so the post-v0.12 window accumulates without rewriting history.

## Acceptance

The collector is engineering-complete when:

1. tests pass;
2. checksums/schema parsing work;
3. BTC/ETH forward rows are persisted as workflow artifacts;
4. missing sources are labelled explicitly;
5. no execution authorization appears in artifacts.

A statistical alpha claim requires a later pre-registered minimum observation window and remains a future empirical gate.
