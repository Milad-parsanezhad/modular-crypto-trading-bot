# Roadmap and honest status

| Component | Status | Evidence / next gate |
|---|---|---|
| Causal features and cash/equity accounting | Implemented and tested | Tests cover future perturbations, fills and drawdown |
| RF/XGBoost/LSTM/CNN/Transformer/PPO | Runnable | Two-fold synthetic smoke across both feature conditions |
| Walk-forward, ablations, economic metrics | Implemented | Manifests, regression tests and synthetic outputs |
| LSTM/XGBoost agreement | Backtest implemented | No packaged ensemble paper artifact yet |
| Public exchange ingestion | Implemented, live access unverified | Network blocked the CoinEx download here |
| Forward paper checkpoint and polling | Implemented; deterministic fixture tested | Still needs real-data forward observation |
| Q1 and thesis evidence registry | Reviewed with access labels | Some metadata/abstract-only sources; exact JCR certification pending |
| Real BTC/ETH empirical results | Pending | Obtain actual data and freeze protocol |
| Formal multiple-comparison inference | Pending | Supervisor-approved final inferential plan |
| Joint portfolio, market depth and live orders | Not implemented | Separate specification and operational validation |
| Triangle-under-cloud rule | Not implemented | Resolve exact causal pattern and crossover rules |
| DT, DreamerV3, on-chain data, CVaR optimization | Not implemented | Demonstrate need through baselines first |

Future development should record a problem, a bounded change, tests, experiment provenance and a reviewable commit/PR. Existing fork PR #1 is not silently merged by this change.
