# Modular Crypto Trading Bot — Research Platform v0.1

Evidence-driven, modular cryptocurrency trading research platform.

> **Research status:** v0.1 is a reproducible baseline laboratory, **not** a production trading bot and not evidence of guaranteed profitability.

## Why this repository changed

The literature review and evidence matrix shifted the project away from a single-indicator or single-model bot. The working architecture is now:

**Data → Alpha features → Regime context → Model tournament → Uncertainty / abstention → Risk → Cost-aware execution → Robust validation → Paper trading**

Current high-priority research streams: order flow/microstructure; on-chain value/network activity; carry/funding/basis; momentum/liquidity; multi-dimensional regime detection; event-time sampling + triple-barrier labeling; uncertainty-aware trade/no-trade; execution/market-making RL; strict anti-overfitting validation.

Ichimoku remains a candidate feature family and must earn its place through ablation.

## What v0.1 implements

- public OHLCV download with CCXT (no API keys),
- real-market BTC/USDT 4h baseline,
- point-in-time technical/liquidity features,
- Ichimoku candidate features,
- rolling regime descriptors,
- chronological 70/30 holdout,
- tree-based ML baseline (`HistGradientBoostingClassifier`),
- probability-based abstention (long/flat),
- explicit fee + slippage assumptions,
- Buy & Hold and momentum baselines,
- Sharpe, Sortino, drawdown and Calmar metrics,
- automated synthetic unit test,
- GitHub Actions cloud execution,
- Colab-ready notebook.

## Run
```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_baseline.py --exchange coinex --symbol BTC/USDT --timeframe 4h --limit 2000
```

## Google Colab
Open `notebooks/Research_Bot_v0_1_Colab.ipynb` in Colab. It clones this repository, installs the package, runs tests, and executes the real-market baseline.

## Current scientific limitations
v0.1 does not claim a validated profitable strategy. Still missing point-in-time on-chain/funding data, trades/LOB order flow, CUSUM/triple-barrier, CPCV/PBO/DSR, multi-seed DL/RL, capacity/market impact, funding/borrow/maker-taker model, untouched final test and forward paper trading.
