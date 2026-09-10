# v0.22 ICT Vision Baseline Results — 2026-09-10

Dedicated branch: `research/v22-ict-vision-lab`
Workflow run: `34482490299`
Artifact: `v22-ict-vision-smoke` (`10154281341`)
Artifact SHA-256: `f9db29f3cf867b2b8843cbe3a80aba2cd113c385aeb74a75fd37322cff865336`

## Integrity gate

The deterministic renderer/label integrity suite passed **6/6 tests**. The tests covered renderer determinism, prefix invariance/no-future pixels, weak-label prefix invariance, strict separation of future outcome from image channels, and the forward contracts for the custom CNN and small ViT.

## Public-data smoke design

- Data source: CoinEx public spot OHLCV.
- Assets: BTC/USDT, ETH/USDT, SOL/USDT.
- Timeframe: 4h.
- Requested history: 2,200 bars per asset.
- Causal image lookback: 64 bars.
- Raster: 8 channels, 64x96.
- Maximum samples per asset: 220.
- Training: two epochs for this smoke run.
- Architectures: custom small CNN and small ViT.
- Selection: validation weak-structure macro-F1 + 0.25 * validation outcome AUC; test was not used to rank architectures.

## Results

| Model | Validation weak macro-F1 | Validation outcome AUC | Validation balanced accuracy | Test weak macro-F1 | Test outcome AUC | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| small CNN | 0.07919 | 0.49491 | 0.5000 | 0.08509 | 0.48084 | 0.5000 |
| small ViT | 0.07097 | 0.47222 | 0.5000 | 0.06488 | 0.44010 | 0.5000 |

The original smoke runner mechanically named `small_cnn` a `VISION_REPRESENTATION_CANDIDATE` because it had the highest validation objective among successful models. **That mechanical label is not accepted as a scientific promotion decision.** Both models are close to random on the future-outcome head and the structural macro-F1 is too low for a useful ICT recognizer.

## Scientific decision

`NO_VISION_ENCODER_PROMOTED_BASELINE`

Reasons:

1. Outcome AUC did not exceed random discrimination on validation or test.
2. Weak-structure macro-F1 was below 0.10.
3. Several rare ICT labels collapsed to all-negative predictions, indicating severe label imbalance.
4. The structural raster itself contains engineered BOS/FVG/OB context, so high weak-label accuracy on that representation would not by itself prove that computer vision learned ICT structure from raw candles.
5. The smoke dataset and two-epoch budget are intentionally too small to support a promotion claim.

## Required correction before the next run

The next experiment is pre-registered as v0.22b and must add:

- class-balanced/asymmetric loss for rare ICT labels;
- raw-candlestick-only versus structure-augmented channel ablation;
- label-support-aware macro metrics;
- larger chronological sample;
- explicit minimum acceptance gates rather than automatic selection of the least-bad model;
- a distinction between **raw visual detection of ICT structure** and **using deterministic engineered ICT channels as downstream features**.

Until v0.22b passes those gates, the vision branch is research-only and is not exposed to the trading policy or reinforcement-learning state.
