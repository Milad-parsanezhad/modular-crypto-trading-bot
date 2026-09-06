# Frozen initial real-market run — 2026-09-06

Configured before inspecting model test returns. This is an internal timestamped protocol, not an externally preregistered study.

- Instrument: Binance spot BTC/USDT, UTC 4h candle-open timestamps.
- Requested range: 2023-01-01 inclusive to 2026-08-01 exclusive. August 2026 monthly archive was unavailable (official checksum endpoint returned HTTP 404 on retrieval); no fabricated or forward-filled extension.
- Source: official Binance monthly archive, HTTPS checksum manifest verified against every ZIP. 2025+ microsecond timestamps and earlier milliseconds decoded explicitly.
- Cross-check: aggregate six 4h candles to each UTC day; compare all OHLC and BTC volume against CryptoDataDownload's Binance daily CSV. Tolerances: 0.01 USDT for prices, 0.00001 BTC for daily volume. This is a second delivery of the same exchange's prices, not independent exchange validation. A discrepancy beyond tolerance blocks the experiment until resolved.
- Frozen configuration: `configs/market_btc_4h.toml`. Initial train 4,800 bars, validation 720, test 720, three expanding folds, two-bar purges. Unused tail is excluded from this run.
- Seeds 7, 42, 314; all six model families; with/without Ichimoku; LSTM/XGBoost agreement and four baselines. Report all conditions, including flat/no-trade behavior. Do not select a deployment winner from test returns.
- Maximum supervised epochs 20, validation early stopping, three validation-only probability thresholds. PPO 20,000 requested timesteps, bounded seeded episodes sampling only the training segment (up to 720 bars per episode). This is a baseline compute budget, not a convergence claim.
- Risk: 0.1% fee per side, 5 bps adverse slippage per side, 95% allocation cap, 2% stop-based risk budget, ATR stop/take at 2/4, 5% equity drawdown circuit breaker, 3% daily loss breaker. These thresholds trigger responses; gap and execution losses can exceed them.
- Cash is reset at each fold. Compounded fold returns are a normalized, periodically reset walk-forward comparison, not a continuously operated live account. Seed-range error bars are not confidence intervals; repeated seeds share market observations.
- Carry forward all limitations in `docs/protocol.md`: development walk-forward, no final untouched holdout result, no multiple-comparison-adjusted superiority claim, no real orders or demonstrated live profitability.

Original synthetic evidence remains historical evidence for the original commit. Real-run manifests fingerprint the updated source, including archive ingestion and randomized PPO training starts.

## Pre-training data-quality amendment

The second delivery lacked 27 days in 2026, while the official 4h archives were complete. All OHLC values in the 1,281 shared days matched exactly. Before model training or test-return inspection, the remaining 27 days were assigned an additional check against checksum-verified official **daily** Binance archives. This is cross-resolution consistency, not an independent second provider for those dates. No missing days are synthesized, no rows are removed from the market dataset, and the configuration is unchanged. `scripts/audit_market_data.py` preserves both types of evidence and blocks on discrepancies.

Interrupted runs can resume from per-model atomic checkpoints, with configuration, data, source and environment equality required. This changes operational resilience, not thresholds or model selection.
