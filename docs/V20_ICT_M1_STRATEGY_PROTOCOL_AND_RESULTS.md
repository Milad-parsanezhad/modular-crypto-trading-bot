# v0.20 ICT/M1 Causal Strategy Extraction Lab

Date: 2026-09-10

Status: `HYPOTHESIS_NOT_PROMOTED`

Execution: `PAPER/BACKTEST ONLY`

Live execution: `false`

## Objective

Convert the uploaded ICT, TTrades and M1Trades teaching material into explicit,
causal rules and test whether the extracted entry sequence has economic value
after fees and slippage.  The educational sources motivate hypotheses; they do
not constitute scientific evidence of profitability.

## Frozen strategy contract

The implemented sequence is:

1. compute point-in-time Ichimoku state;
2. confirm a swing only after the following candle has closed;
3. require a liquidity sweep to wick beyond a previously confirmed swing and
   close back inside it;
4. require a later directional candle to break and close beyond the opposite
   confirmed swing;
5. require the confirmation body to be at least `0.50 * ATR(14)`;
6. activate a pending origin entry only from the next candle;
7. compare three pre-registered origin definitions at the 50% level:
   `sweep_origin`, `opposing_candle`, and `fvg`;
8. use the sweep extreme plus `0.05 * ATR` as structural invalidation;
9. target `3R`, arm break-even after `2R`, and expire pending/held trades after
   fixed horizons;
10. charge 10 bps fee and 2 bps slippage per side assumption as implemented by
    the trade simulator, with a 0.25% account risk budget per trade;
11. resolve an unknown same-candle TP/SL path conservatively as stop-first.

The Ichimoku gate is trend-only and causal.  No forward-displaced Senkou or
future Chikou value is read.

## Engineering defects found and fixed

### UTC reporting boundary

The first real archival run failed because period starts were timezone-naive
while Binance timestamps were timezone-aware UTC.  All development, validation
and final-test boundaries are now explicit UTC, with a regression test.

### Test environment

The initial command used a Python environment without pytest.  The repository
contract remains `pip install -e ".[dev]"`; Colab and CI install the declared dev
dependencies before running tests.

### `horizon` versus `horizon_bars`

The current repository was audited for the previously reported schema mismatch.
All active research engines use `horizon_bars`; an existing regression test
rejects the stale `horizon` contract.  No speculative compatibility alias was
added.

## Leakage controls

- Swing pivots are not back-painted onto their pivot candle.
- A confirmed swing becomes sweep-eligible only on the following candle.
- A sweep and its structure confirmation cannot be the same candle.
- A wick through structure is not an MSB; a close is required.
- Origin orders become eligible only after MSB confirmation.
- Future outcome data exist only in the trade audit table.
- Mutating future candles leaves historical features and setups unchanged.
- Invalid OHLCV fails closed.

## Real-data experiment

Source: official Binance Vision monthly spot archives already cached locally.

Timeframe: 4h.

Range: 2020-01-01 through 2025-12-31.

Rows per asset: 13,151.

### Full period

| Asset | Origin variant | Trades | Win rate | Mean net R | Profit factor | Total return | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | Sweep origin | 153 | 24.18% | -0.1525 | 0.811 | -5.79% | -7.67% |
| BTCUSDT | Opposing candle | 135 | 29.63% | -0.1279 | 0.813 | -4.31% | -5.56% |
| BTCUSDT | FVG | 90 | 26.67% | -0.1912 | 0.718 | -4.26% | -7.13% |
| ETHUSDT | Sweep origin | 133 | 27.07% | 0.0318 | 1.044 | 0.94% | -3.39% |
| ETHUSDT | Opposing candle | 121 | 26.45% | -0.1280 | 0.813 | -3.88% | -5.99% |
| ETHUSDT | FVG | 86 | 25.58% | -0.1165 | 0.827 | -2.53% | -5.89% |

### Untouched 2025 final test

| Asset | Origin variant | Trades | Profit factor | Total return | Decision |
|---|---|---:|---:|---:|---|
| BTCUSDT | Sweep origin | 25 | 0.992 | -0.06% | Reject |
| BTCUSDT | Opposing candle | 23 | 0.409 | -2.58% | Reject |
| BTCUSDT | FVG | 18 | 0.630 | -1.08% | Reject |
| ETHUSDT | Sweep origin | 24 | 1.198 | 0.89% | Insufficient sample |
| ETHUSDT | Opposing candle | 17 | 0.862 | -0.43% | Reject |
| ETHUSDT | FVG | 14 | 1.252 | 0.54% | Insufficient sample |

The positive ETH cells are too small and too sparse to establish an edge.  The
BTC opposing-candle variant was positive in 2024 validation but failed sharply
in 2025, which is direct evidence of temporal instability.

## Scientific conclusion

The extracted single-timeframe 4h ICT/M1 rule is **not a validated profitable
strategy**.  It must not be promoted to live, used as a thesis performance
claim, or optimized on the 2025 holdout.  The result is still useful: it rejects
the idea that a simple sweep/MSB/origin translation is sufficient and narrows
the remaining research to information that is absent from this baseline.

## Evidence boundary and targeted literature check

A targeted search did not locate a peer-reviewed validation of the branded
`ICT`, `M1Trades`, `smart-money concepts`, or `fair-value-gap` sequence as a
complete profitable rule. This is not proof that no such paper exists; it is a
reason to treat the uploaded teaching material as a hypothesis source rather
than performance evidence.

There is narrower scientific support for mechanisms adjacent to the teaching
language:

- Osler reports clustering of stop-loss orders and price cascades in currency
  markets. That supports testing liquidity-trigger mechanisms, but does not
  validate the ICT entry sequence or its profitability.
- Cont, Kukanov and Stoikov find that short-horizon price changes are strongly
  related to order-flow imbalance and market depth. An OHLC `liquidity sweep`
  is only a coarse proxy for those order-book variables.
- Donier and Bonart document market impact in Bitcoin metaorders. This supports
  using crypto microstructure data in the next experiment, not inferring hidden
  institutional intent from candle geometry.
- The Deflated Sharpe Ratio and Probability of Backtest Overfitting literature
  requires disclosure and correction for strategy search. The three origin
  variants in this lab are therefore recorded as three trials, and no winner is
  promoted from the small positive ETH cells.

Primary references:

1. Osler, C. L. (2005), *Stop-loss orders and price cascades in currency
   markets*, Journal of International Money and Finance 24(2), 219-241,
   https://doi.org/10.1016/j.jimonfin.2004.12.002
2. Cont, R., Kukanov, A., and Stoikov, S. (2014), *The Price Impact of Order
   Book Events*, Journal of Financial Econometrics 12(1), 47-88,
   https://doi.org/10.1093/jjfinec/nbt003
3. Donier, J. and Bonart, J. (2015), *A Million Metaorder Analysis of Market
   Impact on the Bitcoin*, https://arxiv.org/abs/1412.4503
4. Bailey, D. H. and Lopez de Prado, M. (2014), *The Deflated Sharpe Ratio*,
   Journal of Portfolio Management 40(5), 94-107,
   https://doi.org/10.3905/jpm.2014.40.5.094
5. Bailey, D. H., Borwein, J., Lopez de Prado, M., and Zhu, Q. J. (2017), *The
   Probability of Backtest Overfitting*, Journal of Computational Finance
   20(4), 39-69, https://doi.org/10.21314/JCF.2016.322

## Required next research

1. Preserve 2025 as spent holdout and register a new future/forward evaluation
   before changing parameters.
2. Implement the source-faithful multi-timeframe sequence (4h/1h context and
   15m/5m/1m entry) with availability-time as-of joins and automated tests that
   forbid using an unfinished higher-timeframe candle.
3. Test session and news gates adapted to 24/7 crypto rather than copying Forex
   kill zones.
4. Add funding, open interest, spread, order-book imbalance and depth only when
   point-in-time history is reliable; do not label candle geometry as observed
   liquidity.
5. Add bootstrap confidence intervals, multiple-testing correction and
   cross-asset replication before promotion.
6. Use ML only as a pre-registered meta-label filter over primary events; do not
   search architectures until the rule baseline and labels are stable.

## Reproduction

```bash
pip install -e ".[dev]"
pytest -q tests/test_ict_m1_v20.py

PYTHONPATH=. python scripts/run_ict_m1_lab_v20.py \
  --archive-cache ../data/cache \
  --symbol BTCUSDT \
  --timeframe 4h \
  --bars 20000 \
  --ichimoku-gate trend \
  --output-dir artifacts/v20-ict-m1-btc
```

The Colab notebook performs the same run and records the data fingerprint.
