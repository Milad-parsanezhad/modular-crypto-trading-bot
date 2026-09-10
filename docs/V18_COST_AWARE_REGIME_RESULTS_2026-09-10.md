# v0.18 Cost-Aware Alpha Conversion & External Regime Replication — Official Results

Date: 2026-09-10

Status: **Verified research evidence only — no PAPER strategy replacement and no LIVE authorization**

GitHub Actions run: `34404326283`

Artifact: `v18-cost-aware-regime-results`

Artifact ID: `10124733622`

Artifact digest: `sha256:655cf19fee5e7d181111b579e9b273f0fef307ce6939e0170f68e55b18d17c54`

CI result: **SUCCESS**

## 1. Experiment A — Cost-aware alpha conversion

### Research question

Does a frozen forecasting rule produce better economic results when its directional forecast is passed through a cost + uncertainty hurdle rather than executed naively?

### Data and configuration

- Source: CoinEx public spot data.
- Symbols: `BTC/USDT`, `ETH/USDT`.
- Timeframe: 4h.
- Per-symbol split: 70% training / 30% untouched holdout.
- Holdout observations: 540 per symbol.
- One-way cost: 12 bps.
- Round-trip hurdle: 24 bps.
- Uncertainty penalty: training-residual MAD × 0.25.
- Forecast model: frozen Ridge regression within the v0.18 protocol.
- Paired moving-block bootstrap: 500 resamples, block length 12 bars.

### Aggregate portfolio results

| Metric | Naive sign trading | Cost-aware abstention |
|---|---:|---:|
| Net return | **-7.48%** | **+0.05%** |
| Annualized return | -27.06% | +0.20% |
| Sharpe | -1.109 | +0.085 |
| Sortino | -1.442 | +0.035 |
| Max drawdown | -13.57% | **-1.32%** |
| Calmar | -1.995 | +0.149 |
| Turnover sum | 114.0 | **10.0** |
| Explicit cost sum | 0.1368 | **0.0120** |
| Exposure | 51.39% | **1.48%** |
| Trade transitions | 191 | **18** |

The cost-aware policy therefore reduced turnover by about 91%, reduced explicit modeled cost by about 91%, and reduced maximum drawdown substantially. However, statistical support for a positive incremental edge was not established.

### Paired bootstrap inference

Cost-aware minus naive mean net return:

- common periods: `540`;
- observed mean difference: `+0.0001303` per 4h period;
- 95% moving-block bootstrap CI: `[-0.0003031, +0.0005508]`.

Because the lower bound is below zero, the pre-specified superiority gate is not satisfied.

### Per-symbol nuance

**BTC/USDT**

- naive net return: `-19.28%`;
- cost-aware net return: `-0.21%`;
- naive max drawdown: `-20.09%`;
- cost-aware max drawdown: `-0.48%`;
- turnover: `121 -> 4`.

**ETH/USDT**

- naive net return: `+5.35%`;
- cost-aware net return: `+0.29%`;
- naive max drawdown: `-16.98%`;
- cost-aware max drawdown: `-2.63%`;
- turnover: `107 -> 16`.

This is economically interesting but not sufficient for a claim of improved expected return. The cost-aware policy mostly changed **when not to trade**, sharply lowering exposure and turnover.

### Formal decision

`NO_COST_AWARE_CONVERSION_EVIDENCE`

Interpretation: the frozen cost-aware rule improved several economic-risk diagnostics in this sample, but the incremental mean edge did not survive the pre-specified dependence-aware confidence gate. It remains a research hypothesis rather than a promoted strategy.

---

## 2. Experiment B — External regime-conditioned Ichimoku replication

### Research question

Does the v0.11 observation that Ichimoku performed better in `HIGH_VOL` and `TREND_DOWN` regimes replicate on an external venue when the regime definition is frozen and not retuned?

### Data and configuration

- External source: OKX public spot data through CCXT.
- Symbols: `BTC/USDT`, `ETH/USDT`.
- Timeframe: 4h.
- Rows per symbol after feature preparation: 1,748.
- Frozen favorable regimes: `HIGH_VOL`, `TREND_DOWN`.
- Cost: 12 bps one-way.
- Paired moving-block bootstrap: 500 resamples, block length 12 bars.
- No regime threshold was tuned on this external sample.

### Aggregate portfolio results

| Metric | Plain Ichimoku | Regime-conditioned Ichimoku |
|---|---:|---:|
| Net return | **-9.07%** | **-2.79%** |
| Annualized return | -11.23% | -3.48% |
| Sharpe | -0.289 | -0.175 |
| Sortino | -0.297 | -0.105 |
| Max drawdown | -30.96% | **-17.99%** |
| Turnover sum | 108.0 | **38.0** |
| Explicit cost sum | 0.1296 | **0.0456** |
| Exposure | 35.15% | **10.58%** |
| Trade transitions | 182 | **65** |

Regime conditioning reduced loss, drawdown, turnover, cost and exposure relative to plain Ichimoku. But the conditioned strategy itself remained negative and the paired confidence interval crossed zero.

### Paired bootstrap inference

Regime-conditioned minus plain Ichimoku mean net return:

- common periods: `1,748`;
- observed mean difference: `+0.0000253` per 4h period;
- 95% moving-block bootstrap CI: `[-0.0002141, +0.0002566]`.

The external replication therefore does not establish a statistically supported positive regime interaction.

### Per-symbol nuance

**BTC/USDT**

- plain Ichimoku: `-23.61%`;
- regime-conditioned: `-0.73%`;
- max drawdown: `-39.74% -> -15.46%`;
- turnover: `126 -> 34`.

**ETH/USDT**

- plain Ichimoku: `+7.14%`;
- regime-conditioned: `-5.30%`;
- max drawdown: `-26.45% -> -22.12%`;
- turnover: `90 -> 42`.

This cross-asset disagreement is important. The regime filter materially improved BTC but worsened ETH return, which argues against claiming a generalizable regime-conditioned Ichimoku edge.

### Formal decision

`NO_EXTERNAL_REGIME_REPLICATION_EVIDENCE`

Interpretation: the frozen v0.11 regime hypothesis improved several aggregate risk/cost diagnostics on the external venue, but it did not replicate as a positive, statistically supported cross-asset trading edge.

---

## 3. Combined v0.18 conclusion

The combined decision is fail-closed:

- `cost_aware_supported = false`;
- `external_regime_supported = false`;
- `paper_strategy_promotion_authorized = false`;
- `live_execution_authorized = false`.

The most useful scientific insight is not that both ideas are useless. It is that both showed **economic attenuation** — much lower turnover, cost and drawdown — without clearing the statistical promotion gate. That distinction matters: an execution/risk filter can be operationally useful even when it has not established positive alpha.

The correct thesis statement is therefore:

> In v0.18, cost-aware abstention and frozen regime conditioning both reduced trading intensity and downside exposure in the tested samples, but neither produced statistically supported incremental alpha under the pre-specified bootstrap gate. Consequently, both remain unpromoted research hypotheses and the real-money execution gate remains closed.

## 4. Next justified research step

Do not tune either v0.18 result on the same samples. The next defensible step is to use the search-aware v0.17 trial registry and move to genuinely new evidence for any further test. Priority should be given to:

1. prospective forward replication of cost-aware abstention using an unchanged rule;
2. higher-quality global/multi-venue order-flow features;
3. only if a candidate survives those gates, run SPA/Reality Check, DSR/PSR and CPCV/PBO where statistically appropriate;
4. keep RL and deeper architectures gated until a simpler signal demonstrates robust incremental value.
