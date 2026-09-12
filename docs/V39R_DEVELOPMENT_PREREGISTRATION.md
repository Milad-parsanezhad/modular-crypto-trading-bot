# v0.39R Development Characterization — Preregistration

Status: **frozen before empirical execution**  
Purpose: characterize the reconstructed canonical mother strategy on already-consumed development venues.  
This is not v0.39 learning-process optimization and cannot authorize PAPER/LIVE execution.

## Frozen universe

Venues (development only):

- CoinEx
- OKX
- KuCoin

Kraken is explicitly excluded and remains sealed.

Symbols, required where listed on each venue:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT
- DOGE/USDT

Timeframe: `4h`

Fixed sample window:

- start: `2025-01-01T00:00:00Z`
- end: `2026-09-10T23:59:59Z`

The end date is fixed before running this characterization. No later candles may be appended to this preregistered run.

## Frozen canonical decision rule

Use `research_bot.canonical_strategy_v39r.CanonicalConfig` unchanged:

- `min_score = 6`
- `require_event = true`
- `displacement_atr = 0.80`
- `mitigation_atr = 0.50`
- `brooks_body_fraction = 0.55`
- `brooks_close_location = 0.65`
- `stop_atr = 1.50`

No threshold may be changed after observing empirical results from this run.

## Frozen trade realization

- Decision is made on closed bar `t`.
- Entry is `open[t+1]`.
- One active trade per symbol at a time; new signals while a trade is open are ignored.
- Stop distance = `1.5 × ATR[t]`.
- Profit target = `3.0R`.
- Maximum holding period = `30` bars.
- If stop and target are both touched in the same bar, stop is assumed first (conservative ordering).
- If neither barrier is reached, exit at the close of the final holding bar.

## Frozen costs

Base round-trip cost: `24 bps`.

Stress round-trip cost: `36 bps`.

Costs are converted to R using entry price and initial stop distance.

## Frozen metrics and development gates

Each venue is evaluated independently. A venue passes only if all gates pass:

- selected/trade count `>= 200`
- base-cost profit factor `>= 1.05`
- base-cost mean net expectancy `> 0R`
- positive-asset breadth `>= 0.60`
- moving-block bootstrap 95% CI lower bound for mean net R `> 0`
- positive-quarter fraction `>= 0.60`
- 36 bps stress profit factor `>= 1.00`

Bootstrap settings:

- resamples: `500`
- moving block length: `20` trades
- deterministic seed: `314`

## Frozen overall decision

`V39R_DEVELOPMENT_PASS` requires **all three venues** to pass every frozen gate.

Any of the following yields `V39R_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE`:

- insufficient signal/trade count,
- failure of any frozen profitability/robustness gate,
- missing required market coverage that prevents a valid venue evaluation,
- data-fetch/integrity failure.

## Scientific interpretation

This run answers only:

> Does the reconstructed canonical architecture, with parameters fixed before looking at these results, produce sufficiently broad and temporally robust development evidence under the inherited frozen gates?

It does not answer whether live trading is appropriate, and it does not justify touching the sealed Kraken holdout.
