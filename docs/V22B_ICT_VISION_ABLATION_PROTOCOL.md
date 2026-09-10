# v0.22b ICT Vision Ablation Protocol

Frozen before inspecting v0.22b model results: 2026-09-10.

## Why v0.22b exists

The first v0.22 smoke run passed all causality/integrity tests but failed scientifically as a useful vision encoder: weak ICT macro-F1 was below 0.10 and future-outcome AUC was below 0.50. The next run therefore changes the learning protocol, not the historical labels after seeing test outcomes.

## Pre-registered corrections

1. Rare ICT labels are trained with development-only damped positive-class weighting: `sqrt(negative/positive)`, clipped to a maximum of 25.
2. ICT structure receives 70% of the multitask loss and the noisy next-bar outcome head receives 30%.
3. Three channel modes are tested separately:
   - `raw_candles`: bull body, bear body, wick, volume only;
   - `raw_plus_ichimoku`: raw candles plus causal Ichimoku geometry;
   - `structure_augmented`: all eight deterministic channels including engineered liquidity/BOS/FVG/OB/PD context.
4. A weak-label contributes to the supported macro-F1 only when validation/test contain at least 8 positives and 8 negatives.
5. Model selection is validation-only. Test can only confirm or reject the frozen validation winner.

## Critical scientific distinction

A model that recognizes a weak ICT label from `structure_augmented` pixels has access to deterministic engineered structure marks, so that score is not evidence that the network independently discovered ICT structure from raw candles. The `raw_candles` arm is the actual visual-detection experiment. The `structure_augmented` arm instead asks whether deterministic ICT engineering provides a useful downstream representation.

## Frozen acceptance gates

### Raw ICT detector

- at least 8 supported labels on validation and test;
- validation supported macro-F1 >= 0.25;
- test supported macro-F1 >= 0.20.

### Structure-augmented representation

Relative to the validation-selected raw detector:

- validation outcome AUC >= 0.52;
- validation AUC gain >= 0.015;
- test outcome AUC >= 0.50.

No vision outcome can directly authorize a strategy, PAPER replacement, or live execution. A passing encoder is only eligible to enter the later multimodal ablation with tabular, unsupervised and temporal/LSTM representations.

## Current benchmark scope

The first v0.22b run uses public CoinEx spot 4h history for BTC/USDT, ETH/USDT and SOL/USDT, a 64-bar causal window, 64x96 rasters, up to 600 chronological samples per asset, five epochs, small CNN and small ViT architectures, and all three channel modes.
