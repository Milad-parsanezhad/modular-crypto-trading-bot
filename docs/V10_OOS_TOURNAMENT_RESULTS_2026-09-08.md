# v0.10 Live Eligible-Universe Purged OOS Alpha Tournament — 2026-09-08

## Status

**Research result only — not a BUY/SELL signal and not authorization for paper/live execution.**

GitHub Actions run `34219431117` completed successfully: deterministic tests, live public data collection, eligibility auditing, purged out-of-sample tournament, evidence validation, and artifact upload all passed.

## Live data scope

- Discovered listings: **15,801**
- Unique discovered assets: **3,798**
- Deep-history candidates requested: **20**
- Final eligible assets after history/spread/data-quality gates: **17**
- Provider/history failures: **3** (`DGAI/USDT`, `PONS/USDT`, `CP/USDT`)
- Eligible symbols: `ASTER/USDT`, `BTC/USDT`, `DOGE/USDT`, `DOT/USDT`, `ETH/USDT`, `HYPE/USDT`, `NEAR/USDT`, `PUMP/USDT`, `SOL/USDT`, `SUI/USDT`, `TAO/USDT`, `UNI/USDT`, `WLD/USDT`, `XAUT/USDT`, `XMR/USDT`, `XRP/USDT`, `ZEC/USDT`.

The v0.9 pagination limitation was corrected before this experiment: exchange-specific short OHLCV pages (for example ~300 or ~499 rows) no longer cause a false `INSUFFICIENT_HISTORY` conclusion while timestamps continue to advance.

## OOS design

- Timeframe: **4h**
- Forecast horizon: **1 bar**
- Purged expanding OOS folds: **3**
- Panel rows: **25,483**
- Synchronized timestamps: **1,619**
- Panel coverage: **2025-12-12 12:00 UTC → 2026-09-08 04:00 UTC**
- Portfolio: long-only top quartile, coverage matched
- One-way transaction-cost assumption: **12 bps**
- Positive-label hurdle: **12 bps**
- Learned challengers: Logistic Regression, HistGradientBoosting, Random Forest
- Baselines: Momentum, Ichimoku, Equal-Weight Market

## Aggregate OOS results

| Variant | Net return | Sharpe | Sortino | Max DD | Explicit cost sum | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Ichimoku baseline | +32.04% | 1.578 | 2.422 | -35.80% | 0.2404 | — |
| Equal-weight market | +24.90% | 1.512 | 2.212 | -22.62% | 0.00219 | — |
| Logistic | +8.97% | 0.717 | 1.170 | -29.99% | 0.55176 | 0.5262 |
| Momentum baseline | +0.97% | 0.397 | 0.603 | -44.63% | 0.3068 | — |
| Random Forest | -37.52% | -1.774 | -2.752 | -53.49% | 0.81768 | 0.5383 |
| HistGradientBoosting | -39.94% | -1.973 | -2.986 | -59.03% | 0.9324 | 0.5251 |

These figures are specific to this pilot configuration and period. They are not a profitability claim.

## Fold stability

The strongest aggregate learned model was Logistic Regression, but it was not stable across folds:

- Fold 1: net **-13.64%**, Sharpe **-1.10**
- Fold 2: net **-9.04%**, Sharpe **-1.10**
- Fold 3: net **+38.72%**, Sharpe **5.03**

The Ichimoku baseline also varied materially by regime/fold:

- Fold 1: net **+6.89%**, Sharpe **1.13**
- Fold 2: net **-20.79%**, Sharpe **-4.19**
- Fold 3: net **+55.94%**, Sharpe **6.79**

The equal-weight market baseline was negative in the first two folds and strongly positive in the third. This supports the research conclusion that a favorable aggregate result can be dominated by a specific market regime and must not be treated as a stable trading edge without robustness tests.

## Promotion decision

**`NO_MODEL_PROMOTED`**

Provisional best learned model: **Logistic Regression**.

Promotion failed for two explicit reasons:

1. `NO_CLEAR_SHARPE_EDGE_OVER_BASELINES`
2. `INSUFFICIENT_FOLD_STABILITY`

Therefore the model-selection gate remains closed. The bot must not route this tournament winner to paper execution.

## Scientific interpretation

1. In this pilot, more complex tree models did not outperform the simple baselines after costs.
2. The learned models had AUC values close to 0.5; direction classification quality is weak even where portfolio P&L is temporarily positive.
3. Transaction costs materially changed conclusions: Logistic gross return was much higher than net return because turnover consumed a large fraction of the gross edge.
4. Ichimoku was useful enough to survive as a baseline/challenger, but its fold instability prevents promotion to a standalone decision rule.
5. The next justified phase is **v0.11 robustness and inference**, not automatic paper deployment: multi-seed runs, moving-block bootstrap, multiple-testing correction, coverage-matched diagnostics, and regime-conditional ablation.

## Evidence contract

No v0.10 result is allowed to become `CONFIRMED_ENTRY`, `BUY_SIGNAL`, or live order authorization. Paper execution remains gated behind OOS superiority, robustness, uncertainty/cost tests, and the independent risk engine.
