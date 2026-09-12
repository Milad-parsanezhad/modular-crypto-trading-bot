# Screened literature review — AI/ML automated crypto trading

Date: 2026-09-12
Status: research synthesis; no paper/live authorization.

## Screening rule

A source is allowed to influence the thesis core only when its useful claim is compatible with causal feature construction, chronological validation, explicit transaction costs, and an untouched holdout. High headline returns or classification accuracy alone are not treated as evidence of tradable alpha.

## High-confidence methodological evidence — ADOPT

### Timotin (2026), MSc, Instituto Politécnico de Santarém
**A Leakage-Aware Benchmark of Hourly Bitcoin Return Forecasting: Naive, Statistical, and Machine Learning Models under Walk-Forward Validation.**

- Hourly BTC, Aug-2017 to Jan-2025.
- Seven models including naive baselines, linear regression, random forest, classical time-series models and an RNN.
- 1h/6h/24h horizons.
- 53 walk-forward segments with a 24-hour purge gap.
- No model beat a zero-return forecast in MAE; the nearest challenger was not statistically significant.

**Adopt:** naive/zero-return benchmark, strict walk-forward, purge, statistical forecast-skill comparison.  
**Reject:** any claim that model complexity itself should justify promotion.

Reference: http://hdl.handle.net/10400.15/6155

### Pindza (2026), Frontiers in Blockchain
**Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets.** DOI: 10.3389/fbloc.2026.1811716

- >3.4m minute observations; six crypto assets; Binance spot and perpetual futures.
- Purged walk-forward validation and realistic cost evaluation.
- Flexible boosted models overfit badly under leakage controls; no 5-minute strategy survived retail fees.
- Cross-asset transfer was weak, while same-asset transfer across spot/futures venues was materially stronger.

**Adopt for v0.43:** asset-specific models trained across development venues, rather than assuming one pooled cross-asset decision boundary.  
**Adopt:** naive benchmark and cost-first evaluation.  
**Defer:** minute microstructure/LOB features to a later execution-layer experiment because the current mother strategy is 4h.

Reference: https://doi.org/10.3389/fbloc.2026.1811716

### Rezanka (2026), MSc, FH Upper Austria
**Evaluation und Backtesting von Machine-Learning-Modellen in algorithmischen Handelsstrategien.**

- BTC/USDT, ETH/USDT, XRP/USDT 5-minute OHLCV.
- Logistic Regression, Random Forest and XGBoost under rolling walk-forward evaluation.
- ML can filter, confirm or weight rule-based signals, but cannot reliably rescue a fundamentally weak base strategy.

**Adopt:** keep ICT/SMC/Ichimoku/Brooks as independent causal strategy engines; ML is a conditional filter/utility estimator, not a replacement for the trading hypothesis.

Reference: https://pure.fh-ooe.at/en/studentTheses/evaluation-und-backtesting-von-machine-learning-modellen-in-algor/

### Grądzki, Wójcik & Lessmann (2025), Financial Innovation
**Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning.** DOI: 10.1186/s40854-025-00866-w

- BTC/ETH tick data, 2018-2023.
- Information-driven sampling and Triple-Barrier labels were compared with simpler sampling/labeling.
- CUSUM + Triple Barrier was promising in the tested settings after costs; Transformer-type models did not universally dominate.

**Adopt later, isolated:** information-driven event sampling is a credible hypothesis, but it must be a separate experiment (planned v0.44) so it is not confounded with v0.43 asset-specific learning.

Reference: https://doi.org/10.1186/s40854-025-00866-w

### Qi (2026), PhD, University of Essex
**Novel Trading Algorithms augmented by Intrinsic Time and Machine Learning.**

A newly deposited doctoral thesis studies directional-change / intrinsic-time event representations together with machine-learning and reinforcement-learning trading systems. Its limitations explicitly include asset-class/time-span generalizability and compute requirements, while its research direction strengthens the case for testing event-time representations separately from fixed clock-time bars.

**Adopt for v0.44 hypothesis generation only:** compare a frozen asset-specific learner under event-driven sampling / intrinsic-time style triggers against the existing 4h clock-time event stream.  
**Do not merge into v0.43:** otherwise learning structure and sampling structure change simultaneously.

Reference: https://repository.essex.ac.uk/43797/

## Useful but not core evidence — DEFER

### Marinis (2025), MSc, University of Piraeus
**Bitcoin High Frequency Trading with Transformers.**

The thesis uses Binance LOB data, fits scaling on training data only, and reports practical problems with overfitting, directional bias, class imbalance and limited predictive success. The conclusion proposes stochastic/distributional outputs rather than relying only on point classification.

**Adopt conceptually:** distributional/uncertainty-aware prediction and train-only preprocessing.  
**Defer Transformer:** no justification to increase model capacity while simpler models still expose temporal instability.

Reference: https://dione.lib.unipi.gr/xmlui/handle/unipi/17815

### Madhavan (2025), MSc, Higher School of Economics
**Predicting Price Movements in Cryptocurrency Market Using Liquidity Analysis and Machine Learning Methods.**

- ETH/USDT Bybit second-level LOB snapshots.
- DeepLOB variants with BiLSTM/GRU, attention and focal loss; best reported accuracy about 66%.

**Defer:** potentially useful for a future execution/microstructure layer. Accuracy is not an economic promotion criterion and the 4h mother-strategy study should not mix in second-level LOB data now.

Reference: https://www.hse.ru/en/edu/vkr/1051241857

### Mondol (2024), MSc, University of Zurich
**A Deep Reinforcement Learning approach with Explainability for cryptocurrency trading.**

DQN/A2C/PPO/DDPG on BTC/ETH using OHLCV, technical indicators and blockchain metrics. Useful ideas include fractional exposure, on-chain features and XAI.

**Defer:** DRL adds many degrees of freedom and environment-design risk. On-chain features and RL may be separate ablations only after a stable predictive edge is established under the existing gates.

### Juchli (2018), MSc, TU Delft
**Limit order placement optimization with Deep Reinforcement Learning: Learning from patterns in cryptocurrency market data.**

This thesis developed a broker-like RL environment and DQN-based limit-order placement for BTC. It is valuable primarily as an **execution-layer** precedent, not as evidence that RL discovers directional alpha.

**Defer to execution layer:** if the thesis bot eventually earns paper/live authorization, order placement can be optimized separately from signal generation so alpha and execution are not confounded.

Reference: https://repository.tudelft.nl/record/uuid:e2e99579-541b-4b5a-8cbb-36ea17a4a93a

### Petre-Luca (2026), TU Delft
**Enhancing Financial Algorithms for Pairs Trading using Reinforcement Learning — Constrained Portfolio Optimization.**

PPO agents were evaluated OOS with and without transaction costs against a classical z-score strategy. The constrained agent learned the spread direction but did not beat the classical rule and tended to over-trade when no arbitrage was available; costs pushed agents toward more conservative policies.

**Adopt as a caution for future RL:** the agent must explicitly learn/permit a no-trade state and must beat a simple benchmark after costs.  
**Do not add PPO now.**

Reference: https://repository.tudelft.nl/record/uuid:5e3700cc-57b9-4f28-8adc-f3899a202205

### van Oosterhout (2025), TU Delft
**Feature Engineering in Reinforcement Learning for Algorithmic Trading.**

The thesis reports strong sensitivity to state representation: in its Forex DQN setup, adding more indicators/history could add noise and worsen generalization, while agent-state information such as trade duration could help.

**Adopt as a feature-governance principle:** more features are not automatically better; additions require preregistered ablation and OOS incremental utility. Our competing-risk engine already models duration explicitly, which is consistent with this direction.

Reference: https://repository.tudelft.nl/record/uuid:b50b1185-38d6-4385-a8a5-40ddc67e5567

## Rejected as immediate next steps

1. **Bigger Transformer/CMamba/PatchTST now** — rejected for v0.43 because current evidence points to temporal/asset transfer instability, not insufficient capacity.
2. **PPO/DDPG trading policy now** — rejected because it would conflate alpha discovery with policy optimization and reward design.
3. **Outcome-based asset pruning** — prohibited; symbol/data screening must never use expectancy, PF, future returns or model score.
4. **Lowering v0.42 thresholds** — prohibited.
5. **Using Kraken to choose the next hypothesis** — prohibited; Kraken remains sealed.
6. **Adding many indicators because they are available** — rejected; new features require incremental-utility ablation because recent RL-thesis evidence shows larger state representations can worsen overfit.

## Resulting research sequence

- **v0.42:** frozen result = pooled cross-asset breadth rejected; 4/5 OOS folds negative.
- **v0.43:** same-asset cross-venue learning + true moving-block perturbation stability + naive Brier-skill benchmark + strict data-quality manifest.
- **v0.44 (only after v0.43):** isolated information-driven/intrinsic-time sampling ablation (CUSUM/range/volume/dollar/directional-change where data permits) with the model and economic gates held fixed.
- **Future:** on-chain ablation, then LOB/execution layer, then DRL only if a stable predictive/economic edge exists.
