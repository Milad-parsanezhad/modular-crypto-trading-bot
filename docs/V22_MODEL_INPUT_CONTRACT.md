# v0.22 Model Input Contract

The computer-vision encoder may receive only tensors produced by the deterministic renderer and declared channel subsets. The following information is prohibited from model pixels: future candles, outcome labels, entry/exit outcome, realized PnL/R, symbol text, timestamps rendered as glyphs, axis prices, chart-platform watermarks, prediction annotations, test-segment statistics, or labels generated with right-side information that was unavailable at the signal close.

Raw-candle detector input is exactly: bullish body, bearish body, wick, volume.

Raw+Ichimoku input adds only causal Tenkan/Kijun/unshifted cloud geometry.

Structure-augmented input adds deterministic weak-rule features for liquidity/BOS/sweeps/FVG/OB/PD regime, and must never be described as independent visual discovery of those same labels.

Downstream fusion must consume frozen embeddings produced after model selection; the fusion learner cannot refit the image encoder on the untouched test period.
