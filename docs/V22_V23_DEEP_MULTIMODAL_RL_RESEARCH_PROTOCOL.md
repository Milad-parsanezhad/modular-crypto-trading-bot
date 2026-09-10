# v0.22–v0.23 Deep Multimodal + Reinforcement Learning Research Protocol

Frozen: 2026-09-10

Status: research / BACKTEST / PAPER only. Live capital is not authorized by this protocol.

## Research objective

Extend the v0.20 strategy laboratory and v0.21 leakage-safe meta-labeling system into a multimodal research stack that learns from four complementary views of market history:

1. causal tabular/event features;
2. sequential temporal tensors for recurrent/state-space/attention models;
3. candlestick/market-structure visual tensors for computer vision;
4. sequential decisions and portfolio state for reinforcement learning.

The system is not permitted to declare a model superior because it has the highest in-sample return. Every stage must produce immutable artifacts, dataset fingerprints, frozen validation decisions and a single-use chronological test.

## Stage A — supervised tabular benchmark

v0.21 remains the first benchmark. Major linear, kernel, tree, boosting, nearest-neighbor and MLP families are compared. Development fits parameters, validation selects model/threshold, and test is opened only after the champion is frozen.

Primary targets are post-cost profitable-event classification and realized R-multiple regression. Raw predictive metrics are secondary to post-cost economic metrics.

## Stage B — unsupervised and self-supervised representation learning

Unsupervised models are not allowed to directly claim profitable signals. Their output is a representation/regime/anomaly feature evaluated downstream.

Classical unsupervised arm:

- PCA / Incremental PCA
- K-Means
- Gaussian Mixture Model
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

Neural self-supervised arm:

- masked time-series autoencoder;
- LSTM sequence autoencoder;
- temporal contrastive representation learner;
- optional masked-latent functional prediction experiment.

Evaluation includes cluster stability across seeds, silhouette only as a descriptive statistic, regime persistence, anomaly enrichment in future realized-tail events, and incremental OOS value when embeddings are added to the supervised meta-label model.

## Stage C — historical sequence learning

The sequence dataset is point-in-time. For each event at time t, the tensor contains only bars <= t. No centered scaling, future normalization, revised higher-timeframe candle or future market statistic is permitted.

Core channels include normalized OHLC geometry, log returns, range/body/wick ratios, volume surprise, ATR, EMA distances/slopes, causal Ichimoku geometry, liquidity/BOS/sweep/FVG/order-block state, completed higher-timeframe context and optional point-in-time on-chain/derivatives features when available.

Frozen sequence model benchmark:

- vanilla RNN baseline;
- GRU;
- LSTM;
- bidirectional LSTM only for encoding the *past window* (never future bars beyond event time);
- CNN-1D + LSTM;
- Temporal Convolutional Network;
- Transformer encoder;
- PatchTST-style patch encoder;
- xLSTM research arm;
- state-space / Mamba-style research arm when the dependency can be reproduced reliably;
- hybrid variable-selection + LSTM arm.

LSTM is a required first-class benchmark rather than an assumed winner. Sequence length, hidden dimension and dropout are selected on validation only.

## Stage D — computer vision for candlestick and ICT structure recognition

Computer vision is used as a second representation of exactly the same causal window, not as a screenshot oracle.

Each image ends at signal_time. The image generator must be deterministic and store its rendering configuration and source-bar hash. Two image families are compared:

- raw normalized candlestick raster;
- multi-channel structural raster with separate channels for candle body/wicks, volume, causal Ichimoku cloud geometry and ICT/SMC structural annotations.

Multi-task labels include:

- bullish/bearish candle morphology;
- swing high/low;
- BOS up/down;
- CHoCH proxy where formally defined;
- liquidity sweep up/down;
- bullish/bearish FVG;
- order-block candidate;
- breaker/mitigation proxy where formally defined;
- premium/discount location;
- causal Kumo state;
- downstream post-cost event outcome.

Rule-derived ICT labels are explicitly marked `weak_structural_labels`; they teach the vision encoder to reproduce formalized structure, not to prove that ICT terminology is a causal market truth. The downstream outcome head is evaluated separately.

Vision benchmark:

- small custom CNN;
- ResNet18;
- EfficientNet-B0;
- Vision Transformer / small ViT;
- optional CNN + temporal fusion head.

Interpretability artifacts include Grad-CAM-like saliency for the CNN arm and attention/patch attribution where technically valid. Visual models are rejected if they exploit axes, labels, timestamps, color conventions or any rendering artifact correlated with the target.

## Stage E — multimodal fusion

The fusion model receives only frozen upstream representations:

`tabular_meta + unsupervised_regime + temporal_embedding + vision_embedding + account/risk_state`

Fusion candidates include calibrated logistic stacking, gradient boosting, gated MLP and attention/gating fusion. A component is retained only if ablation testing improves validation economics and then survives the untouched test.

## Stage F — reinforcement learning environment

The RL agent is trained by trial-and-error only inside historical/simulated/PAPER environments. It is prohibited from exploratory learning with live capital.

State contains causal market representations, current position, cash/exposure, recent turnover, drawdown, volatility/risk state, meta-label probability, temporal embedding and ICT-vision probabilities.

Initial action spaces:

- discrete: short / flat / long or reduce / hold / increase;
- continuous: target exposure in [-1, 1] where the venue and research contract allow it.

Every environment step applies transaction fees, slippage, turnover, position limits, liquidity constraints and the hard risk engine before reward calculation.

Base reward is risk-adjusted incremental log equity:

`reward_t = log(E_t / E_{t-1}) - λ_turnover*turnover - λ_dd*Δdrawdown_penalty - λ_tail*tail_risk - λ_inventory*inventory_risk`

No reward may use future information unavailable at the step.

## Stage G — RL algorithm tournament

Discrete-action arm:

- tabular Q-learning sanity baseline;
- DQN;
- Double DQN;
- recurrent/LSTM-DQN research arm.

Policy-gradient / actor-critic arm:

- A2C;
- PPO;
- recurrent PPO when reproducible;
- DDPG;
- TD3;
- SAC;
- recurrent/LSTM-SAC research arm.

Offline / sequence-RL arm:

- Behavior Cloning baseline;
- Conservative Q-Learning research arm;
- Decision Transformer;
- Q-learning Decision Transformer research arm if reproducible.

Model-based arm:

- DreamerV3-style world-model experiment after the simpler agents and environment pass integrity tests.

The final policy is selected by multi-seed validation statistics, not one lucky seed. The minimum research requirement is five seeds for promotion candidates; ten seeds are preferred for the final thesis comparison when compute permits.

## Stage H — anti-overfitting and promotion gates

Every model family must pass:

- deterministic seed/reproducibility logging;
- no-future-data integrity tests;
- chronological development/validation/test separation;
- walk-forward / expanding-window evaluation;
- explicit fees/slippage and breakeven-cost analysis;
- multiple random seeds for stochastic deep/RL models;
- bootstrap/block-bootstrap confidence intervals;
- multiplicity-aware model-selection reporting;
- per-asset/per-regime performance breadth;
- external-venue replication where data semantics permit;
- stress tests for higher cost, latency, missing data and parameter perturbation.

A historical pass authorizes only a fresh forward PAPER candidate. Live execution remains a separate engineering, operational and risk-readiness decision.

## Evidence saved at every stage

Every run must save a manifest with Git SHA, package versions, seed, UTC timestamp, input source identifiers, exact train/validation/test bounds, feature schema, target schema, fee/slippage assumptions, dataset SHA-256, model configuration and output hashes.

Each stage saves its leaderboard, per-seed metrics, predictions/actions, equity curve, trade ledger, confusion/calibration metrics where relevant, risk metrics, learning curves, ablations and a machine-readable `decision.json`.

No stage overwrites prior evidence; versioned artifact directories are mandatory.

## Final research architecture

`raw point-in-time data -> causal feature store -> strategy events -> weak ICT structure labels + future outcome labels -> tabular ML -> unsupervised/SSL representations -> LSTM/sequence encoders -> candlestick/ICT vision encoder -> multimodal fusion -> RL policy -> independent risk engine -> PAPER execution -> immutable evidence`

The independent risk engine remains outside the learned policy so an RL model cannot learn to disable the hard capital-protection rules.
