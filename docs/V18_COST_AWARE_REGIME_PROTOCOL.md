# v0.18 Cost-Aware Alpha Conversion & External Regime Replication

Date: 2026-09-10

Status: **Pre-specified research experiment — no execution promotion**

## Objective

v0.18 tests two hypotheses that emerged from the project’s own evidence and recent literature without reusing v0.12 for tuning.

### Experiment A — Cost-aware alpha conversion

Question: does a fixed forecast become economically better when trades are entered only if predicted return exceeds a training-only hurdle representing expected round-trip cost plus an uncertainty penalty?

Frozen design:
- CoinEx public spot data;
- BTC/USDT and ETH/USDT;
- 4h bars;
- point-in-time project feature set;
- Ridge regression with fixed alpha `5.0`;
- chronological 70/30 train/holdout split with one-bar purge;
- no threshold tuning on holdout;
- naive position: predicted next return > 0;
- cost-aware position: predicted next return > `2 * one_way_cost + 0.25 * train_residual_MAD`;
- one-way cost assumption: 12 bps;
- paired moving-block bootstrap, 500 resamples, 12-bar blocks.

Support requires all of:
1. at least 120 holdout periods;
2. positive cost-aware net holdout return;
3. lower turnover than naive trading;
4. 95% moving-block bootstrap lower bound for cost-aware minus naive mean net return > 0.

Otherwise decision is `NO_COST_AWARE_CONVERSION_EVIDENCE`.

### Experiment B — External replication of the v0.11 Ichimoku/regime hypothesis

Question: does the previously observed concentration of Ichimoku performance in HIGH_VOL / TREND_DOWN replicate on a separate public venue/source?

Frozen design:
- primary external venue: OKX spot via CCXT; KuCoin fallback if the primary source is unavailable;
- BTC/USDT and ETH/USDT;
- 4h bars;
- same point-in-time regime code as v0.11;
- frozen base Ichimoku score: `ichi_tenkan_kijun + ichi_price_kijun - ichi_cloud_width`;
- plain signal: score > 0;
- conditioned signal: score > 0 AND (`HIGH_VOL` OR `TREND_DOWN`);
- no regime threshold tuning on the external sample;
- one-way cost assumption: 12 bps;
- paired moving-block bootstrap, 500 resamples, 12-bar blocks.

Support requires all of:
1. at least 120 external periods;
2. positive regime-conditioned net return;
3. 95% lower bootstrap bound for conditioned minus plain mean net return > 0.

Otherwise decision is `NO_EXTERNAL_REGIME_REPLICATION_EVIDENCE`.

## Why this design is stricter

The design intentionally separates signal estimation from execution conversion. Recent BTC walk-forward research reports that naive sign-based trading can fail after 10-bp costs while cost-aware filters can materially reduce turnover and improve economic results in selected configurations. That motivates Experiment A, but the project must still reproduce the effect with its own data and protocol.

Recent Journal of Financial Markets evidence also finds predictive value in cross-sectional world order flow. This motivates future higher-quality flow research, but does not retroactively rescue the negative v0.12 derivatives result. v0.18 therefore focuses first on execution conversion and an externally frozen regime hypothesis rather than adding more adaptive features.

## Claim policy

Possible outputs are research labels only. Even if either experiment is supported:
- no real-money LIVE authorization;
- no automatic PAPER strategy replacement;
- no claim of universal profitability;
- a supported result must enter v0.17 search-aware audit and prospective replication before further promotion.
