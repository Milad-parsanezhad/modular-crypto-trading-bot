# v0.22b ICT Vision Ablation Results — 2026-09-10

Workflow run: `34483282539`
Artifact: `v22b-ict-vision-ablation` (`10154695549`)
Artifact ZIP SHA-256: `d2ff1e571f2d1a7d9166047ceed940a6cf90d079d746a335344428236eafd0b3`
Artifact size: 11,385,144 bytes; 27 evidence files.

## Integrity

The v0.22b ablation test suite passed **5/5 tests**. These tests cover raw-channel isolation, class-imbalance weighting, support-aware F1, and fail-closed promotion logic.

## Frozen experiment

- Public CoinEx spot OHLCV.
- BTC/USDT, ETH/USDT, SOL/USDT.
- 4h timeframe.
- 64-bar causal lookback; 64x96 raster.
- Up to 600 samples per asset.
- Chronological split per asset, then concatenation.
- Development: 1,080 samples.
- Validation: 360 samples.
- Test: 360 samples.
- Architectures: small CNN, small ViT.
- Channel modes: raw candles; raw+causal Ichimoku; full structure-augmented.
- Five training epochs with development-only positive-class weighting.

## Model results

| Architecture | Channels | Val supported F1 | Val outcome AUC | Test supported F1 | Test outcome AUC | Test balanced acc. |
|---|---|---:|---:|---:|---:|---:|
| small CNN | raw candles | 0.13381 | 0.53795 | 0.11682 | 0.54000 | 0.51755 |
| small ViT | raw candles | 0.12059 | 0.56355 | 0.10401 | 0.51607 | 0.50000 |
| small CNN | raw + Ichimoku | 0.11924 | 0.54593 | 0.11442 | 0.58282 | 0.56682 |
| small ViT | raw + Ichimoku | 0.10834 | 0.52652 | 0.09486 | 0.54768 | 0.50000 |
| small CNN | structure augmented | **0.16267** | **0.58021** | **0.13630** | **0.58686** | **0.56043** |
| small ViT | structure augmented | 0.09628 | 0.48211 | 0.08833 | 0.51073 | 0.50000 |

Validation had 17 weak labels with adequate support; test had all 20 labels with adequate support.

## Decision

`AUGMENTED_REPRESENTATION_CANDIDATE_ONLY`

The validation-selected raw detector was `small_cnn:raw_candles`, but it failed the pre-registered raw-ICT-recognition gate because validation supported macro-F1 (0.13381) was below 0.25 and test supported macro-F1 (0.11682) was below 0.20.

The validation-selected structure-augmented model was `small_cnn:structure_augmented`. Its validation outcome AUC was 0.58021 versus 0.53795 for the frozen raw CNN, a validation gain of **+0.04226 AUC**. It also cleared the pre-registered test confirmation gate with test outcome AUC **0.58686**. Therefore engineered ICT/Ichimoku/liquidity structure is allowed to continue as a downstream representation challenger.

## What this does and does not show

This run provides preliminary evidence that the deterministic structure-augmented representation contains information that a small CNN can exploit better than raw candlestick/volume pixels under this particular 4h BTC/ETH/SOL experiment.

It does **not** show that a neural network independently learned ICT from raw candles. The raw detector explicitly failed that gate. It also does not demonstrate trading alpha: the outcome head is a representation diagnostic and has not been passed through the full transaction-cost, position-sizing, economic backtest, external-venue replication or forward-PAPER protocol.

## Next research action

The next vision stage should focus on **spatial supervision/localization** of ICT structures from raw candles (not only global multi-label classification), then compare raw candle, causal Ichimoku, structure channels, GAF/GADF and temporal/LSTM embeddings in a multimodal ablation. Reinforcement learning remains downstream of those representation tests.

No PAPER replacement or LIVE execution is authorized by this result.
