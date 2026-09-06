# Evidence → Implementation map

| Hypothesis | v0.1 | v0.2 target | Kill criterion |
|---|---:|---:|---|
| Tree models are strong baselines | ✅ | model tournament | no OOS value vs simple heuristic |
| Prediction ≠ tradability | ✅ fee/slippage | richer execution model | gross-only edge |
| Abstention can improve net alpha | ✅ | calibrated uncertainty | no utility improvement |
| Regime matters | ✅ simple regime | multi-dimensional regime | no OOS increment |
| Ichimoku may add information | ✅ features | formal ablation | remove if non-incremental |
| Event-time sampling may help | — | CUSUM/range/volume bars | benefit vanishes under robust CV |
| On-chain value contains information | — | point-in-time MVRV/network | no incremental OOS alpha |
| Carry/funding is independent | — | funding/basis engine | no net edge after frictions |
| Order flow is a major signal | — | trades/LOB engine | no capacity-adjusted edge |
| RL is strongest for execution/control | — | shielded PPO/SAC | seed/multiplicity instability |

## Non-negotiable validation
Point-in-time audit; realistic costs; purging/embargo; CPCV/walk-forward; multiple-testing correction; multi-seed stochastic models; ablation; untouched final test; forward paper trading.
