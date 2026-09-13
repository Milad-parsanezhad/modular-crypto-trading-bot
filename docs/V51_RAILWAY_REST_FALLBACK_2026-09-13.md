# v0.51 — Railway Direct-REST Collector Fallback — 2026-09-13

Status: **ENGINEERING TRANSPORT FALLBACK FROZEN BEFORE PROSPECTIVE START**

Parent preregistration: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`  
Predictor-identity amendment: `4838c0d98c408d358929b0767676a5ac024bd8dd`  
Evidence-integrity addendum: `da5dcf5cd3568a92c75871016f000917c9380955`  
Prospective start: `2026-09-13T12:00:00Z`

## Why this fallback exists

On 2026-09-13 GitHub Actions experienced degraded performance and this repository's
hosted jobs repeatedly failed before any step with `runner_id=0`. A minimal independent
runner smoke job reproduced the same behavior. Railway also rejected provisioning a
new service because the current plan resource limit is reached.

An existing Railway service, `v6_2_verification_read_only`, has no active deployment,
already has a read-only diagnostic purpose and an existing `SHARED_DATABASE_URL`
reference to the project's Postgres. To avoid modifying any active trading/fundamental
service or provisioning a new resource, v0.51 may use this otherwise idle service as a
cron-only market-evidence collector.

This document freezes only the **data transport/runtime fallback**. It does not alter
A0/A1, predictor identity, Expected-R admission, costs, exits, Financial Governor,
risk gates, assets, venues, support thresholds or the 60-minute first-seen rule.

## Public REST sources

No exchange credentials or trading permissions are required.

- CoinEx spot market candlesticks: `GET https://api.coinex.com/v2/spot/kline`
  with `market=<ASSET>USDT`, `period=4hour`.
- OKX spot candlesticks: `GET https://www.okx.com/api/v5/market/candles`
  with `instId=<ASSET>-USDT`, `bar=4H`; only rows with `confirm=1` are accepted.
- KuCoin spot klines: `GET https://api.kucoin.com/api/v1/market/candles`
  with `symbol=<ASSET>-USDT`, `type=4hour`.

The frozen universe remains BTC, ETH, SOL, XRP and DOGE on CoinEx, OKX and KuCoin.
Kraken is forbidden.

## Semantic contract

The direct-REST implementation must produce the same normalized raw fields as the
Python collector:

- `first_seen_at`
- `venue`, `symbol`
- `bar_open_time`, `bar_close_time`
- `capture_lag_minutes`
- `prospective_eligible_v51`
- OHLCV
- source identifier

A bar is eligible only when it is fully closed, lies inside the 150-day window and is
first seen no later than 60 minutes after close. Older API backfill is retained as
context with `prospective_eligible_v51=false`.

## Persistent evidence store

The fallback writes only to a dedicated Postgres schema named `v51_research` using the
existing `SHARED_DATABASE_URL` reference. It must not update or delete tables in
`public` or any other application schema.

Tables are namespaced for v0.51 raw bars, collection runs and run manifests. Existing
raw-bar identities are immutable: if an exchange later returns changed OHLCV for the
same `venue × symbol × bar_open_time`, the collector records a revision error and does
not overwrite the first-seen evidence.

## Cron and runtime safety

Nominal cron: `10 */4 * * *` UTC.

Runtime state:

`RESEARCH_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`

The process contains no authenticated exchange endpoint, API key, order method,
position method or execution engine. It exits after one collection pass.

## Canonicality

The JavaScript/Bun source used in Railway must be committed in this repository before
it is encoded into the Railway function start command. The Railway start payload must
be derived byte-for-byte from that committed source. Deployment success is engineering
evidence only; it is not a v0.51 economic result.
