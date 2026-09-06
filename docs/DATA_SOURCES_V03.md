# v0.3 Data & Alpha Lab — Source Contract

This stage is about **data integrity and stronger information sources**, not strategy profitability.

## Official CoinEx public sources

- Spot OHLCV: `GET /v2/spot/kline` with explicit `start_time` / `end_time` windows.
- Futures OHLCV: `GET /v2/futures/kline`.
- Public funding history: `GET /v2/futures/funding-rate-history`.
- Public basis history: `GET /v2/futures/basis-history`.
- Public market trades: `GET /v2/spot/deals` and `GET /v2/futures/deals`; the `side` field is taker side.
- Current futures open interest: `GET /v2/futures/market` (`open_interest_volume`).

## Open-interest limitation

CoinEx currently exposes market-level open interest publicly as a current snapshot. The research code therefore:
1. saves CoinEx OI as a timestamped **snapshot**;
2. does **not** backfill that snapshot into history;
3. optionally ingests recent Binance USD-M OI as a clearly labeled cross-venue auxiliary feature;
4. keeps long-history OI archival backfill as a separate data-engineering task.

## Point-in-time rules

- No backward filling of temporal features.
- No future-shifted Ichimoku/Chikou features.
- No cross-venue series may be relabeled as same-venue data.
- Every output includes coverage timestamps and a gap audit where applicable.
- Recent trade depth is measured in trade IDs/rows and is not claimed to be long time coverage.

## v0.3 outputs

- long-history CoinEx spot and futures klines;
- funding history;
- basis history;
- CoinEx OI snapshot;
- recent spot/futures public trades;
- 4h order-flow imbalance features;
- aligned spot/perpetual realized-basis proxy;
- optional recent Binance OI series;
- machine-readable JSON audit report and CSV artifacts.

No private keys and no live orders are used.
