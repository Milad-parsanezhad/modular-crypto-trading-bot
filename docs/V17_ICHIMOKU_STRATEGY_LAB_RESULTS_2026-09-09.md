# v0.17 Ichimoku Strategy Lab — Results (2026-09-09)

## Decision

**S6 (20-bar breakout with a positive EMA200 slope) advances only to forward paper observation. IRGC-S is rejected in its current form. Live execution remains disabled.**

This is a research decision, not a profitability claim. S6 was selected after comparison with multiple candidates, so its result requires fresh forward evidence and multiple-testing correction before any testnet review.

## Data and frozen execution assumptions

- BTC/USDT and ETH/USDT spot, 4-hour bars.
- 13,151 bars per asset, 2020-01-01 00:00 through 2025-12-31 20:00 UTC.
- Official monthly Binance Vision archives already stored locally.
- Development: 2020–2023; validation: 2024; untouched final test: 2025.
- Signal at close `t`; simulated fill at open `t+1`; return from open `t+1` to open `t+2`.
- Base friction: 10 bps fee plus 2 bps slippage per side, or 24 bps round trip.
- No future-displaced Ichimoku feature and no backward filling after indicator construction.
- Active-rule sizing: 0.25% risk budget, 35% asset cap, 70% portfolio-gross cap.
- Hard research kill switch: 5% drawdown. Reset is external and audited; the backtest does not auto-reset.

## Untouched final test (2025)

| Asset | Candidate | Net return | Sharpe | Sortino | Maximum drawdown | Explicit cost |
|---|---:|---:|---:|---:|---:|---:|
| BTC/USDT | S6 | 3.53% | 0.831 | 0.627 | -3.42% | 1.37% |
| ETH/USDT | S6 | 5.06% | 1.207 | 0.855 | -3.24% | 0.71% |

For comparison, buy-and-hold returned -6.82% on BTC and -11.55% on ETH in the same calendar window, with drawdowns of -34.37% and -61.80%. This comparison does not correct for the fact that S6 was selected from a candidate set.

## IRGC-S outcome

IRGC-S combined an Ichimoku regime gate, volatility-normalized CUSUM events, IRCP/C2 primary signals, triple-barrier labels and an expanding out-of-sample Logistic meta-model.

| Metric | Result |
|---|---:|
| All candidate events | 355 |
| Out-of-sample events | 139 |
| Selected events | 49 |
| Positive OOS folds | 1 |
| Selected net return | -0.46% |
| Win rate | 46.94% |
| Profit factor | 0.943 |
| Maximum drawdown | -3.59% |

The unfiltered primary events lost 6.72%. Meta-labeling reduced that loss but did not create positive alpha, and only one fold was positive. The correct decision is `REJECT_OR_CONTINUE_RESEARCH`, not promotion.

## Scientific interpretation

The result is consistent with published evidence that Ichimoku performance is market- and regime-dependent rather than universal ([Deng et al.](https://doi.org/10.1002/ijfe.2067); [Che-Ngoc et al.](https://doi.org/10.1007/s10614-022-10319-6); [Lutey and Rayome](https://doi.org/10.17549/gbfr.2022.27.5.17)). S6 is also consistent with the broader time-series-momentum literature ([Moskowitz, Ooi and Pedersen](https://doi.org/10.1016/j.jfineco.2011.11.003); [Lempérière et al.](https://arxiv.org/abs/1404.3274)), but those studies do not validate this particular crypto implementation.

Meta-labeling is treated as a selective overlay, not an automatic source of alpha ([Joubert](https://doi.org/10.3905/jfds.2022.1.098); [Meyer, Barziy and Joubert](https://doi.org/10.3905/jfds.2023.1.119)). Model selection remains exposed to backtest overfitting and inflated Sharpe estimates ([Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253); [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). Competition standings are useful only for generating hypotheses because leverage, winner selection and incomplete strategy disclosure prevent causal inference from a winner table ([official historical standings](https://www.worldcupchampionships.com/world-cup-trading-championship-historical-standings)).

## Remaining gates

1. Record S6 and IRCP challenger decisions prospectively on a second venue.
2. Accumulate the pre-registered minimum forward sample; preserve rejects, outages and zero-trade periods.
3. Reconcile simulated fills with spread, latency, partial-fill and cancellation evidence.
4. Run block bootstrap, Deflated Sharpe and multiple-testing/PBO diagnostics.
5. Repeat under 36 and 60 bps round-trip stress and perform capacity analysis.
6. Permit testnet review only if the independent risk and evidence gates pass.
7. Keep live execution disabled pending a separate live-readiness audit and explicit human authorization.

## Reproduction

```bash
python -m pip install -e '.[dev]'
pytest -q tests/test_strategy_lab_v17.py tests/test_binance_spot_archive_v17.py tests/test_forward_candidate_v17.py
PYTHONPATH=. python scripts/run_strategy_lab_v17.py \
  --archive-cache data/cache --bars 20000 \
  --output-dir artifacts/v17-strategy-lab
PYTHONPATH=. python scripts/run_forward_candidate_v17.py \
  --archive-cache data/cache --symbol BTCUSDT
```

The repository does not distribute the large source archives. The committed result tables and JSON manifest contain the frozen configuration, provenance label and decision contract.
