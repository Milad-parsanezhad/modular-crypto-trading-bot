# v0.22 Multi-Channel Candlestick + ICT Vision Implementation

Frozen: 2026-09-10

Status: research/backtest/PAPER representation learning only. This stage does not authorize live execution.

## Purpose

This stage implements the computer-vision arm defined in `V22_V23_DEEP_MULTIMODAL_RL_RESEARCH_PROTOCOL.md`. The goal is not to teach a neural network to read a decorated TradingView screenshot. The goal is to construct a deterministic, causal visual tensor from exactly the same historical bars that the rule/sequence models see, and then test whether a vision encoder can reproduce formalized ICT/SMC structure and add downstream predictive information.

## Multi-channel renderer

`research_bot/vision_ict_v22.py` renders an axes-free tensor with eight channels:

1. bullish candle bodies;
2. bearish candle bodies;
3. candle wicks;
4. volume profile strip;
5. causal Ichimoku geometry (Tenkan, Kijun, unshifted causal cloud state);
6. liquidity/structure geometry (confirmed swing references, BOS, sweeps);
7. FVG/order-block/mitigation geometry;
8. premium/discount and coarse Kumo regime.

No ticker name, timestamp glyph, axis scale, annotation text, chart watermark, legend, RGB color convention or future-shifted Ichimoku display is drawn. This prevents the network from learning presentation artifacts instead of market geometry.

Every sample stores both a SHA-256 of the exact source bars and a SHA-256 of the rendered tensor. Re-rendering the same causal window must reproduce the same tensor hash.

## Weak ICT structural labels

The implementation creates twenty rule-derived labels known by the close of `signal_time`:

- bullish / bearish candle;
- confirmed previous-bar swing high / swing low;
- BOS up / BOS down;
- CHoCH-up / CHoCH-down proxy based on the prior confirmed BOS state;
- downside / upside liquidity sweep;
- bullish / bearish FVG;
- bullish / bearish order-block candidate;
- bullish / bearish mitigation proxy;
- discount / premium location inside the causal 20-bar dealing range;
- bullish / bearish Kumo state.

These are explicitly named **weak structural labels**. They encode our formal algorithmic interpretation of the uploaded ICT/TTrades/Order-Block material. They do not establish that ICT terminology is a scientifically verified causal law of price formation.

A future direction label is constructed separately for supervised outcome learning. It is never rendered into an image channel.

## Causality rules

- Image window for event `t` contains bars `<= t` only.
- Swing labels that require a right-hand confirmation are attached only after that confirmation bar has closed; therefore the label is named `confirmed_*_prevbar`.
- Ichimoku cloud values are causal rolling values, not visually shifted future cloud values.
- Premium/discount uses only the prior/current causal dealing-range calculation.
- Structural and outcome targets never enter the image channels.
- Chronological development/validation/test partitions are preserved per symbol before concatenation.

## Vision models

The initial benchmark supports:

- custom multi-channel CNN;
- small Vision Transformer;
- ResNet18 with an adapted eight-channel first convolution;
- EfficientNet-B0 with an adapted eight-channel stem.

All models use two supervised heads plus a reusable embedding:

- multi-label weak ICT structure head;
- future-outcome direction head;
- latent vision embedding for the later multimodal fusion/RL state.

The loss is a weighted combination of multi-label BCE for ICT structure and class-balanced BCE for the outcome head. Model fitting uses development data; early stopping and model ranking use validation only; test results are reported after model selection without refitting.

## Evaluation

The vision stage is primarily a representation test, not a trading-strategy promotion test. Its metrics include:

- macro-F1 across ICT weak labels;
- per-label F1;
- outcome ROC-AUC;
- outcome balanced accuracy;
- training/validation loss curves;
- frozen test predictions;
- reusable test embeddings.

The validation ranking objective is:

`weak_macro_F1 + 0.25 * outcome_AUC`

This intentionally prioritizes structural recognition. A selected vision encoder is only a `VISION_REPRESENTATION_CANDIDATE` for later ablation/fusion. It cannot replace a strategy and cannot authorize live capital.

## Saved evidence

The runner `scripts/run_v22_ict_vision_lab.py` persists:

- `vision_manifest.json`;
- `data_provenance.json`;
- `weak_label_prevalence.csv`;
- a small compressed dataset audit sample;
- one learning-curve CSV per trained architecture;
- one serialized PyTorch state per successful architecture;
- `vision_leaderboard.csv`;
- `test_predictions.csv`;
- compressed test embeddings for each model;
- `decision.json`.

GitHub Actions workflow `.github/workflows/v22-ict-vision.yml` separately runs deterministic/no-future tests and a public-data 4h smoke benchmark on BTC/ETH/SOL.

## Next research step

After the renderer/labels pass integrity and a vision encoder is reproducible, the next step is **multimodal ablation**, not immediate RL. We will test whether adding the frozen vision embedding to tabular + unsupervised + LSTM/temporal representations improves validation economics and survives the untouched test. Only then should the fused state be exposed to PPO/SAC/DQN/Decision-Transformer/Dreamer-style experiments.
