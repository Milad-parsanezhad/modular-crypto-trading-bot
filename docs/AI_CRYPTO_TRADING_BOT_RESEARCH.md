# AI Cryptocurrency Trading Bot with Machine Learning, Deep Learning and Reinforcement Learning

This repository documents an MSc-level research program for an **AI cryptocurrency trading bot** that combines systematic financial strategies, financial machine learning, deep temporal models, computer vision research, portfolio risk management and future reinforcement-learning allocation.

## Research scope

The system is not designed around a single indicator or one backtest. It evaluates whether a candidate can survive:

- point-in-time / causal data requirements;
- realistic fees, slippage and turnover;
- purged chronological validation;
- model-selection and multiple-testing controls;
- cross-asset and cross-venue replication;
- mark-to-market drawdown and portfolio overlap;
- correlation and CVaR constraints;
- future-time forward evidence.

## Machine-learning research

The project has evaluated or implemented supervised and deep-learning families including linear models, tree ensembles, gradient boosting, LSTM, GRU, TCN, CNN-LSTM and Transformer architectures. Vision/multimodal work studies whether candlestick geometry and causal market-structure representations add incremental information. Unsupervised regime/anomaly context is treated as supporting state information rather than proof of alpha.

The current research frontier is **risk-aware portfolio arbitration / learning-to-rank**: when several individually acceptable trade events occur at the same time, the system must decide which events deserve scarce risk budget. This is separate from event-level classification.

Reinforcement learning remains gated until deterministic and supervised allocation baselines demonstrate robust incremental value on genuinely fresh evidence.

## Strategy research

Research families include causal Ichimoku, S6/Kumo hypotheses, Order Block/BOS/retest structures, Supply & Demand, liquidity sweeps, ICT/TTrades-derived causal adaptations, correlation divergence and higher-timeframe momentum/regime filters.

Educational or discretionary trading concepts are converted into deterministic hypotheses and tested. They are never accepted as scientific facts merely because they are popular online.

## Validation philosophy

The central rule is **Evidence Before Opinion**. A strategy or model is not promoted because it has high accuracy, AUC, a good local backtest or a green CI job. Promotion requires economic and statistical evidence under a frozen protocol.

The project uses or studies moving-block bootstrap, cost stress, breadth tests, CPCV, Probability of Backtest Overfitting (PBO), Deflated Sharpe-style diagnostics, external replication and prospective PAPER evidence.

## Current evidence status

The v0.24d experiment successfully reproduced the exact persisted v0.24b models under their original dependency environment and then evaluated them on two external venues without refitting. Event-level economics remained positive after transaction-cost stress, but the portfolio layer failed frozen profit-factor, mark-to-market drawdown and bootstrap gates. The result is therefore `EXTERNAL_REPLICATION_FAILED`, not a profitable-strategy claim.

That result motivates v0.25: separate **event selection** from **portfolio admission / ranking**, then test the frozen allocator on future-time data that did not contribute to its design.

## Relevant search terms

AI cryptocurrency trading bot, crypto trading bot machine learning, financial machine learning crypto, reinforcement learning trading bot, LSTM crypto trading, Transformer trading model, algorithmic crypto trading, quantitative trading research, portfolio optimization crypto, CVaR trading risk, Ichimoku AI trading, ICT market structure machine learning, MSc artificial intelligence trading thesis.

## Research navigation

- Main README: `../README.md`
- Project status: `PROJECT_STATUS_2026-09-11.md`
- Research traceability: `RESEARCH_TRACEABILITY_MATRIX.md`
- Defense evidence: `DEFENSE_EVIDENCE_INDEX.md`
- Research changelog: `RESEARCH_CHANGELOG.md`

No result in this project is a guarantee of returns or financial advice. Real-money LIVE execution remains disabled unless a future evidence ladder is explicitly passed.