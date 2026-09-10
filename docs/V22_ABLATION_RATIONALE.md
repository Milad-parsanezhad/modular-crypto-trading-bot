# Raw-vs-Engineered Vision Ablation Rationale

The first vision implementation deliberately renders both candle geometry and deterministic ICT structure channels. That representation is useful for downstream multimodal learning, but it creates a circularity risk if weak-label recognition on engineered channels is interpreted as proof that a neural network learned ICT from candles.

v0.22b removes that ambiguity by treating `raw_candles` as the actual structure-detection task and `structure_augmented` as a separate feature-engineering task. Any claimed visual ICT detector must therefore clear its minimum F1 gates using candle/wick/volume pixels without BOS/FVG/OB annotations in the input.
