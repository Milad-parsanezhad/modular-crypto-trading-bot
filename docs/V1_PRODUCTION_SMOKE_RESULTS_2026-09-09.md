# v1.0.0-rc1 Railway Production Smoke — Results

Date: 2026-09-09
GitHub Actions run: `34360057064`
Railway service: `thesis-trading-bot-v08`
Railway deployment: `a74994ff-b1f0-4cde-b2ec-6316c781a2cd`
Deployed repository commit: `1ddeaab39f2b66b8b3c1d0f4c5c8f5cbea8a3853`
Public service domain: `https://thesis-trading-bot-v08-production.up.railway.app`
Status: **SUCCESS — PAPER ONLY — LIVE DISABLED**

## Purpose

This smoke run verifies that the thesis release candidate is not merely unit-tested: the deployed Railway service can be reached from an independent GitHub Actions runner, initializes PostgreSQL persistence, runs the forward-paper pipeline, and remains fail-closed for live execution.

It is an engineering/runtime verification artifact. It is not profitability evidence.

## Production contract checks

The workflow successfully verified:

- `/health` returned `status=ok`;
- service version was `1.0.0-rc1`;
- `forward_paper_enabled=true`;
- `live_execution=false`;
- `/paper/status` returned `RUNNING`;
- `paper_execution_enabled=true`;
- persistence backend was `postgres`;
- `/research/status` retained the research decision `NO_INCREMENTAL_DERIVATIVES_EVIDENCE`;
- forward strategy remained labelled `HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA`.

## Observed forward-paper state

At the beginning of the independent smoke run:

- PostgreSQL observations: **2**;
- paper fills: **1**;
- equity points: **2**.

The persisted account state reported by the paper cycle was:

- cash: **7997.388327318343**;
- marked paper equity: **9997.388327318342**;
- peak paper equity: **10000.0**.

A paper-only ETH/USDT position existed:

- quantity: **0.7997280924485675 ETH**;
- average paper fill price: **2501.6140867262347 USDT**.

These are simulated paper-account values and must not be described as real exchange holdings or realized investment performance.

## Safe run-once verification

The workflow sent one request to `/paper/run-once`.

The service returned:

`V14_FORWARD_PAPER_RUNNING_NOT_LIVE`

with:

- `paper_execution_enabled=true`;
- `live_execution=false`;
- BTC/USDT: `ALREADY_OBSERVED_BAR`, action `NO_TRADE`;
- ETH/USDT: `ALREADY_OBSERVED_BAR`, action `HOLD`.

Both referred to the already-recorded completed 4h bar at `2026-09-09T08:00:00+00:00`. This independently demonstrates restart/retry deduplication: the smoke request did not create another order for a previously processed bar.

## Persistence verification after cycle

A second production read confirmed:

- observations: **2**;
- fills: **1**;
- equity points: **2**;
- `last_cycle_error=null`;
- PostgreSQL backend still active;
- LIVE still false.

## Scientific interpretation

The production paper service is now operational and is collecting prospective evidence. The single paper fill and current paper equity do not validate alpha. A statistically meaningful forward window must accumulate before any performance claim, strategy promotion, testnet gate, or live-readiness review.

## Execution status

- Backtest: implemented.
- External holdout: completed through v0.12; no learned-model promotion.
- Forward microstructure: operational collection in v0.13.
- Forward paper: operational in v0.14.
- PostgreSQL persistence: production verified.
- Railway API/dashboard: production verified by health/API smoke test.
- Testnet: not yet authorized.
- Real-money live: **DISABLED / FAIL-CLOSED**.
