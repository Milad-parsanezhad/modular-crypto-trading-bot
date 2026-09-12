# Research Screening for v0.44 and Later Stages — 2026-09-12

Purpose: convert recent theses/papers into falsifiable engineering decisions for the thesis bot. This is not a bibliography dump. Sources are screened by leakage control, chronological validation, transaction-cost realism, benchmark quality, and relevance to the current bottleneck.

## Current empirical diagnosis before v0.44

- v0.42 falsified broad pooled cross-asset learning under the frozen protocol: 4/5 OOS folds were negative despite adequate sample on OKX/KuCoin.
- v0.43 therefore tests same-asset cross-venue learning, true training-data perturbation stability, data-quality governance and naive Brier skill.
- v0.44 is preregistered but empirically locked until v0.43 freezes its result.
- Kraken remains SEALED; PAPER/LIVE remain false.

## A. High-confidence methodological evidence — adopt in the thesis core

### Timotin (2026), MSc, Instituto Politécnico de Santarém
A Leakage-Aware Benchmark of Hourly Bitcoin Return Forecasting: Naive, Statistical, and Machine Learning Models under Walk-Forward Validation.
https://repositorio.ipsantarem.pt/items/5c80738b-ce21-4c90-8adc-1d137accadc8

Key value:
- hourly BTC, 2017–2025;
- many chronological walk-forward segments with purge;
- naive/statistical/ML/RNN comparison;
- apparent sophistication did not reliably beat a zero-return benchmark.

Decision:
- ADOPT naive baseline and strict chronological/purged comparison;
- REJECT model complexity as a promotion argument.

### Rezanka (2026), MSc, University of Applied Sciences Upper Austria
Evaluation und Backtesting von Machine-Learning-Modellen in algorithmischen Handelsstrategien.
https://pure.fh-ooe.at/en/studentTheses/evaluation-und-backtesting-von-machine-learning-modellen-in-algor/

Key value:
- BTC/ETH/XRP OHLCV;
- Logistic Regression, Random Forest, XGBoost;
- rolling walk-forward evaluation;
- ML integrated into rule-based strategies.

Decision:
- ADOPT the architecture in which ICT/SMC/Ichimoku/Brooks provide causal trading hypotheses while ML estimates conditional utility/filtering;
- REJECT replacing the financial hypothesis with a black-box predictor merely because train loss is lower.

### Pindza (2026), Frontiers in Blockchain
Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets.
https://doi.org/10.3389/fbloc.2026.1811716

Key value:
- >3m minute observations across six cryptocurrencies and spot/perpetual venues;
- purged walk-forward and realistic costs;
- flexible boosted models severely overfit under leakage controls;
- same-asset transfer across venues was stronger than cross-asset transfer;
- no standard retail-fee strategy survived.

Decision:
- DIRECTLY ADOPTED as motivation for v0.43 asset-specific cross-venue learning;
- ADOPT naive benchmarking and cost-first interpretation;
- DEFER minute microstructure/LOB features to a separate execution experiment.

### Bysik & Ślepaczuk (2026)
Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting.
https://arxiv.org/abs/2606.00060

Key value:
- about 70,000 hourly BTC observations, 2018–2026;
- XGBoost, LSTM, iTransformer under 27-fold walk-forward;
- naive sign trading failed after 10 bps costs;
- cost-aware trade admission strongly reduced turnover and restored selected configurations;
- XGBoost was descriptively strong but not statistically dominant over neural alternatives.

Decision:
- ADOPT prediction-to-trade separation, cost-aware admission, and strong classical baseline;
- supports current Expected-Net-R + uncertainty + Financial Governor architecture;
- REJECT automatic promotion of Transformer/LSTM.

### Kim (2026), VALID framework
Beyond Accuracy: A Validation Framework for Machine Learning in Cryptocurrency Trading.
https://doi.org/10.2139/ssrn.6508779

Key value:
- 340 strategy variants across assets/timeframes;
- documents directional bias, statistical/economic disconnect and transaction-cost omission;
- AUC/permutation success can coexist with CPCV/economic failure;
- multiple-testing corrections and PBO/Deflated-Sharpe reveal severe false-discovery risk.

Decision:
- ADOPT explicit statistical + economic gates and research-trial accounting;
- ADD PBO/Deflated-Sharpe as secondary multiple-testing diagnostics in a later validation layer;
- do not use them to rescue a candidate that fails frozen venue/economic gates.

## B. Evidence supporting v0.44 information-driven sampling

### Grądzki, Wójcik & Lessmann (2025), Financial Innovation
Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning.
https://doi.org/10.1186/s40854-025-00866-w

Key value:
- BTC/ETH tick data 2018–2023;
- information-driven sampling and Triple-Barrier labeling;
- CUSUM + Triple Barrier promising in tested settings after costs;
- Transformer models did not universally dominate.

Decision:
- ADOPT only as an isolated sampling hypothesis in v0.44;
- DO NOT claim to recreate tick/dollar/volume bars from 4h OHLCV;
- frozen v0.44 therefore tests only a causal volatility-adaptive CUSUM activity clock.

### Zhao (2026), PhD, University of Essex
Novel Trading Algorithms augmented by Intrinsic Time and Machine Learning.
https://repository.essex.ac.uk/43797/

Key value:
- event-driven Directional Change / intrinsic-time framework;
- trend-following and counter-trend strategy development with ML.

Decision:
- SUPPORTS the general idea that event time may be preferable to fixed clock time;
- DEFER Directional Change itself to a separate experiment after v0.44 so CUSUM and DC effects are not confounded.

## C. Robustness / data-snooping evidence — adopt as diagnostics

### López de Miguel (2026), Universidad Politécnica de Madrid
Generación y validación de estrategias de trading algorítmico.
https://oa.upm.es/95069/

Key value:
- robustness funnel;
- Monte Carlo perturbations;
- parameter-stability tests and systematic parameter permutations;
- Walk-Forward Matrix;
- explicit focus on overfitting/data-snooping.

Decision:
- ADOPT robustness funnel thinking;
- v0.44 primary CUSUM multiplier remains frozen at 1.0;
- a small 0.8/1.0/1.2 neighborhood may be reported only as diagnostic parameter stability after the frozen decision, never as a winner search.

### Application of artificial intelligence in cryptocurrency trading (2026), Prague University of Economics and Business
https://vskp.vse.cz/english/99320

Key value:
- LSTM/GRU crypto forecasting/trading analysis;
- explicitly cautions that reported outperformance depends on ex-post selection of best models and neglect of transaction costs.

Decision:
- ADOPT as a warning against winner-picking and headline forecast performance;
- reinforces the existing frozen candidate order and explicit cost model.

## D. RL / neural evidence — informative but not current core

### Jain (2026), PhD, UCL
Microstructural Financial Modelling: Point Processes and Reinforcement Learning.
https://discovery.ucl.ac.uk/id/eprint/10221263/

Key value:
- low-SNR electronic trading;
- LOB/point-process modelling, stochastic control, RL;
- discusses poor sample efficiency, local minima and limited generalization.

Decision:
- DEFER to a future LOB/execution-layer program;
- current 4h alpha problem should not be complicated with RL until predictive/economic stability is demonstrated.

### van Oosterhout (2025), TU Delft
Feature Engineering in Reinforcement Learning for Algorithmic Trading.
https://repository.tudelft.nl/record/uuid:b50b1185-38d6-4385-a8a5-40ddc67e5567

Key value:
- DQN on EUR/USD;
- more features/history could worsen OOS behavior;
- individual indicators sometimes outperformed combinations due to noise/overfit.

Decision:
- ADOPT incremental-utility/ablation discipline;
- REJECT automatic feature accumulation.

### Lee (2026), Seoul National University of Science and Technology capstone
The Effect of Reward Function Design in RL-Based BTC Grid Trading.
https://github.com/cosmicpotato2047/capstone-rl-trading

Key value:
- PPO reward variants, multiple seeds, CPCV/OOS;
- winner reversal across validation environments.

Decision:
- ADOPT as evidence that RL reward/model winner is evaluation-protocol dependent;
- DEFER PPO until alpha is stable; if RL is later tested, reward variants must be preregistered and multiple-testing controlled.

### Kwon & Sim (2025), Journal of Financial Engineering
AI-Based Cryptocurrency Trading Strategies: Application of Overfitting Minimization and Reinforcement Learning.
DOI: 10.35527/kfedoi.2025.24.3.001

Key value:
- PPO + CPCV;
- demonstrates the importance of robust time-series validation.

Decision:
- VALIDATION IDEA useful;
- no current RL promotion.

## E. Explicitly rejected shortcuts

The following are prohibited as immediate responses to a failed v0.43/v0.44:

1. Lowering the Expected-Net-R / uncertainty / venue qualification thresholds after seeing results.
2. Removing losing assets or event families using test expectancy/PF.
3. Choosing the best CUSUM multiplier after seeing OOS outcomes.
4. Switching to PatchTST/CMamba/LSTM/Transformer merely because HistGB fails.
5. Adding PPO/DDPG/SAC before a stable alpha layer is established.
6. Reconstructing tick-level dollar/volume bars from 4h OHLCV.
7. Treating classification accuracy/AUC as trading qualification.
8. Using Kraken to choose among development hypotheses.

## F. Frozen research sequence

1. Finish corrected v0.43 asset-specific cross-venue characterization.
2. Freeze v0.43 result and artifact lineage.
3. If scientifically warranted, unlock v0.44 CUSUM sampling ablation without changing model/risk gates.
4. Add multiple-testing diagnostics (trial count, PBO/DSR-style evidence) as secondary reporting.
5. Only after stable alpha: separate on-chain ablation.
6. Only after stable alpha: separate LOB/execution research.
7. RL remains an execution/allocation overlay candidate, not an alpha rescue mechanism.
