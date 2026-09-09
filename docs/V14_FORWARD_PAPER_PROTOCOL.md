# v0.14 Forward Paper Protocol

Date frozen: 2026-09-09

## Objective

Collect live-market, forward-only execution and risk evidence after no learned model passed the v0.10-v0.12 promotion gates.

## Venue and universe

Public CoinEx market data:

- BTC/USDT
- ETH/USDT
- 4-hour completed spot bars
- live public top-of-book depth

No private API key is required.

## Frozen shadow strategy

`ICHIMOKU_SHADOW_V14`

This is a transparent rule-based paper hypothesis, **not validated alpha**.

Entry score uses only current/past information:

- price above current Kumo;
- Tenkan above Kijun;
- bullish cloud;
- price above Kijun;
- positive six-bar momentum;
- the pre-existing triangle-under-Kumo detector can elevate the score to 1.0.

Exit:

- price below Kumo; or
- bearish Tenkan/Kijun with price below Kijun.

The strategy acts at most once per completed symbol/bar/version key.

## Risk and execution

Before any paper fill:

- independent drawdown gate;
- gross-exposure gate;
- asset-weight gate;
- turnover gate;
- observed spread gate;
- slippage gate;
- historical CVaR when enough paper equity observations exist.

Sizing uses a fixed risk fraction with ATR-derived stop distance and notional caps.

The paper execution simulator applies fee, slippage, spread-linked extra slippage, visible-depth partial-fill approximation, deterministic client-order IDs and duplicate-bar protection.

## Persistence

When `DATABASE_URL` is configured, PostgreSQL stores observations, positions, fills, account state and equity history. This allows restart/redeployment without replaying the same completed bar.

## Evidence labels

All v0.14 records remain `FORWARD_PAPER_HYPOTHESIS` until a minimum forward observation window is accumulated and evaluated. Paper performance is not a live-trading authorization.

## Live constraint

`LIVE` is fail-closed in code and is not enabled by this phase.
