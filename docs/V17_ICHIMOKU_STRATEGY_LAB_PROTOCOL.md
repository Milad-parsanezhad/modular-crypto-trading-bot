# v0.17 Ichimoku Strategy Lab — Frozen 4H Protocol

## Scope

The primary research universe is BTC/USDT and ETH/USDT at 4-hour resolution. The lab compares Buy & Hold, six simple rules, IRCP, trend-filtered mean reversion, volatility breakout and the IRGC-S CUSUM/meta-label candidate. All decisions use closed bars and execute at the next open.

## Frozen candidates

| ID | Rule family |
|---|---|
| B0 | Buy & Hold |
| S1 | Tenkan/Kijun cross |
| S2 | Kumo breakout |
| S3 | Ichimoku bull regime |
| S4 | Confirmed pullback |
| S5 | Kijun/Kumo structural rejection |
| S6 | 20-bar breakout with EMA200 trend |
| IRCP | Regime-confirmed Ichimoku pullback |
| C1 | Long-trend/short-horizon mean reversion |
| C2 | Volatility squeeze breakout |
| IRGC-S | Regime + CUSUM + primary signal + purged logistic meta-label |

## Causal contract

- Senkou values are computed from information available at the decision timestamp. Plotting displacement is never joined backward.
- Chikou is represented only by the causal comparison of current close with close 26 bars ago.
- Rolling breakout levels exclude the current candle.
- Meta labels can use future barriers only as targets; features are captured at event time.
- Training events must finish before the OOS boundary.
- If stop and profit barriers occur within the same candle, the stop is assumed first.

## Cost and risk contract

- Fee: 10 bps one way.
- Slippage: 2 bps one way.
- Base round trip: 24 bps.
- Stress round trips: 24, 36 and 60 bps.
- Research risk budget: 0.25% per trade.
- Maximum weight per asset: 35%; maximum two-asset gross budget: 70%.
- Drawdown kill switch: 5%. The research run remains flat afterward; 42 bars is the minimum review/cooling period before any externally authorized reset.
- Long/flat, no martingale, no averaging down and no live execution.

## Promotion rule

IRGC-S is not promoted merely for positive raw return. It needs at least 30 selected OOS events, positive net return, Sharpe greater than the unfiltered primary signal and positive performance in at least two OOS folds. Passing this small-sample gate permits forward paper evaluation only; it does not permit live trading.

## Reproduction

```bash
pytest -q tests/test_strategy_lab_v17.py
PYTHONPATH=. python scripts/run_strategy_lab_v17.py --archive-cache data/cache --bars 9000 --output-dir artifacts/v17-strategy-lab
```
