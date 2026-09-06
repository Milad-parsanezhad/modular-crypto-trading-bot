# v0.3 Data & Alpha Lab — Deep-History Result (2026-09-06)

## Run identity
- GitHub Actions run: `34049950143`
- Source commit: `0ba5f071906be726f2143ad8c344e4fd930e89c5`
- Symbol: BTC/USDT
- Primary venue: CoinEx public API
- Requested market bars: 15,000 × 4h
- No private keys; no live orders.

## Historical market data

| Series | Rows | Start | End | Gaps |
|---|---:|---|---|---:|
| CoinEx Spot 4h | 6,573 | 2023-09-07 08:00 UTC | 2026-09-06 16:00 UTC | 0 |
| CoinEx Perpetual/Futures 4h | 12,291 | 2021-01-27 08:00 UTC | 2026-09-06 16:00 UTC | 0 |
| Aligned spot-perpetual | 6,573 | 2023-09-07 | 2026-09-06 | 0 within overlap |

The 1,000-bar engineering ceiling is solved. The remaining pre-2023 Spot limitation is source availability at the requested 4h granularity; CoinEx's official historical-data page states that pre-May-2023 candlestick archives support 1-minute data, so the correct next implementation is archive download + point-in-time resampling.

## Funding
- Rows: **6,144**
- Coverage: **2021-01-27 16:00 UTC → 2026-09-06 16:00 UTC**
- Mean actual funding rate: **0.0000992501**
- Median: **0.0**
- Min: **-0.00375**
- Max: **+0.00374999**
- Non-zero share: **72.705%**

Funding is now a real long-history feature family and can enter carry/regime experiments.

## Basis
### Independently computed spot-perpetual basis proxy
`(futures_close - spot_close) / spot_close`
- n = **6,573**
- Mean: **0.00023488** (~2.35 bps)
- Median: **0.00023031**
- Std: **0.00043216**
- Min: **-0.00206685**
- Max: **+0.00571359**
- Non-zero share: **99.48%**

### CoinEx basis-history endpoint
- Rows: **8,761** over one year
- All returned `basis_rate` values for BTCUSDT were **0.0** in this run.

Decision: do not use this endpoint as an informative BTC feature until its semantics/data are independently verified. Keep the independently computed spot-perpetual basis proxy.

## Open Interest
### CoinEx
Current public snapshot collected successfully:
- Open-interest volume: **912.5745 BTC**
- Contract: linear
- Maker fee rate: 0.0003
- Taker fee rate: 0.0005

CoinEx historical OI is not available from the same public market endpoint. Snapshots should be archived prospectively.

### Cross-venue historical OI
- Binance REST attempt from GitHub Actions: HTTP 451 (regional restriction).
- Bybit REST attempt from GitHub Actions: HTTP 403 (runner/network restriction).

These failures are recorded, not silently replaced. Next OI backfill path: public exchange archives (e.g. Binance Vision metrics) and/or Colab-accessible public endpoints, always labeled by venue.

## Order Flow
CoinEx public recent deals were ingested with taker side.

### Spot recent sample
- Trades: **1,000**
- Coverage: 2026-09-06 17:01:47 → 17:53:59 UTC
- Buy trade share: **47.7%**
- Buy notional: **192,394.37 USDT**
- Sell notional: **221,957.18 USDT**
- Quote imbalance: **-7.13%**

### Futures recent sample
- Trades: **1,000**
- Coverage: 2026-09-06 16:35:18 → 17:53:55 UTC
- Buy trade share: **58.8%**
- Buy notional: **647,749.21 USDT**
- Sell notional: **376,302.11 USDT**
- Quote imbalance: **+26.51%**

This is proof that taker-side order-flow ingestion works, but REST recent deals do not provide enough clock-time depth for historical research. CoinEx officially provides downloadable historical transaction archives; those archives are the next same-venue backfill target.

## Reliability engineering learned in this stage
The first 15k-bar run was interrupted by a transient CoinEx TLS connection reset. Retrying with exponential backoff solved the run without reducing the requested history. A production-quality data layer must tolerate temporary network failures and preserve auditability.

## Scientific conclusion
v0.3 successfully moved the project from weak OHLCV-only modeling to a real multi-source data layer:
1. long-history CoinEx futures prices are available from 2021;
2. long-history funding is available from 2021;
3. same-venue spot/futures basis can be calculated reliably from 2023-09 at 4h via API;
4. current OI and recent taker-side order flow are available;
5. historical OI and full historical order flow require archive/backfill infrastructure rather than pretending recent REST snapshots are historical features.

No alpha/profitability claim is made at v0.3. The next gate is **v0.4 Historical Archive + Alpha Ablation**: pre-2023 minute data resampling, historical transaction/order-flow archives, historical OI archives, then isolated OOS tests for Funding, Basis, OI and Order Flow before any multi-alpha combination.
