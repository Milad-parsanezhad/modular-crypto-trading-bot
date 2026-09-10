# v0.18 Cost-Aware Alpha Conversion & External Regime Replication — Official Results

Date: 2026-09-09

Status: **Research evidence only — no PAPER replacement or LIVE authorization**

Validated GitHub Actions run: `34404326283`

Artifact: `v18-cost-aware-regime-results`

Artifact ID: `10124733622`

Artifact SHA-256: `655cf19fee5e7d181111b579e9b273f0fef307ce6939e0170f68e55b18d17c54`

## Experiment A — Frozen forecast: naive vs cost-aware conversion

Source: CoinEx public spot 4h; BTC/USDT and ETH/USDT.

Decision: **`NO_COST_AWARE_CONVERSION_EVIDENCE`**

| Metric | Naive sign trading | Cost-aware abstention |
|---|---:|---:|
| Net return | -7.48% | +0.05% |
| Sharpe | -1.109 | 0.085 |
| Max drawdown | -13.57% | -1.32% |
| Turnover sum | 114.0 | 10.0 |
| Explicit cost sum | 0.1368 | 0.0120 |
| Exposure | 51.39% | 1.48% |

The cost-aware rule materially reduced turnover, explicit cost, exposure and drawdown, and moved aggregate net performance from negative to approximately flat/slightly positive. However, the paired moving-block bootstrap on cost-aware minus naive mean net return used 540 matched periods and produced observed mean difference `0.0001303` with 95% CI `[-0.0003031, +0.0005508]`. Because the lower bound is not above zero, the pre-registered superiority gate did not pass.

Per asset, BTC improved from -19.28% naive to -0.21% cost-aware, while ETH moved from +5.35% naive to +0.29% cost-aware. This means the aggregate improvement is driven primarily by avoiding a large BTC loss rather than by creating a robust positive alpha stream across both assets.

Scientific interpretation: cost-aware abstention shows a promising *risk/turnover conversion effect* in this frozen experiment, but there is insufficient statistical evidence to claim a positive incremental trading edge.

## Experiment B — Frozen Ichimoku/regime external replication

Source: OKX public spot 4h via CCXT; BTC/USDT and ETH/USDT.

Frozen favorable regimes inherited from v0.11: `HIGH_VOL`, `TREND_DOWN`.

Decision: **`NO_EXTERNAL_REGIME_REPLICATION_EVIDENCE`**

| Metric | Plain Ichimoku | Regime-conditioned Ichimoku |
|---|---:|---:|
| Net return | -9.07% | -2.79% |
| Sharpe | -0.289 | -0.175 |
| Max drawdown | -30.96% | -17.99% |
| Turnover sum | 108.0 | 38.0 |
| Explicit cost sum | 0.1296 | 0.0456 |
| Exposure | 35.15% | 10.58% |

The frozen regime filter improved aggregate return, drawdown, turnover and exposure relative to plain Ichimoku, but did not turn the portfolio positive. The paired moving-block bootstrap used 1,748 matched periods; observed mean difference was `0.0000253` with 95% CI `[-0.0002141, +0.0002566]`. The interval includes zero, so external replication is not established.

Per asset, BTC improved substantially from -23.61% plain to -0.73% conditioned, whereas ETH worsened from +7.14% plain to -5.30% conditioned. This cross-asset divergence is a key falsification signal: the favorable v0.11 regime interaction did not replicate consistently across BTC and ETH on the external venue.

## Combined conclusion

- `cost_aware_supported = false`
- `external_regime_supported = false`
- `paper_strategy_promotion_authorized = false`
- `live_execution_authorized = false`

The two hypotheses showed economically interesting reductions in turnover/drawdown, but neither cleared the pre-registered statistical evidence gate. Therefore v0.18 is retained as an **inconclusive/negative replication result**, not tuned further on the same evaluation sample.

## Thesis interpretation

v0.18 adds an important distinction to the thesis: a policy can improve operational characteristics such as turnover, drawdown and exposure without demonstrating statistically reliable alpha. Similarly, an externally tested regime gate can improve aggregate portfolio behavior while failing cross-asset consistency and dependence-aware confidence requirements. These findings support continued use of fail-closed promotion and search-aware validation rather than post-hoc optimization.
