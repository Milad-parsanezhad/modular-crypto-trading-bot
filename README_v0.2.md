"""README with comprehensive documentation."""
# Modular Crypto Trading Bot — v0.2

**Production-ready modular cryptocurrency trading platform with ML/RL, risk management, and live paper trading simulation.**

> **Status:** Research implementation with validated methodology, NOT a guarantee of profitability. Test thoroughly before use.

## What's New in v0.2

✅ **Critical Bug Fixes:**
- Fixed look-ahead bias in feature engineering and model validation
- Proper chronological train/test split (no data leakage)
- Robust NaN handling in all technical indicators
- Purged K-Fold cross-validation with embargo windows

✅ **Production Features:**
- Risk management: Kelly Criterion, drawdown circuit breakers
- Paper trading engine with order simulation and slippage
- Structured logging and monitoring
- Configuration management (YAML/env)
- Data validation pipeline

✅ **Advanced ML:**
- Ensemble models (Gradient Boosting + Random Forest)
- Calibrated probability predictions
- Early stopping and validation splitting
- Hyperparameter optimization ready

✅ **Enhanced Alpha Factors:**
- Multi-timeframe momentum (1h, 4h, 1d, 1w)
- Volatility regime detection (GARCH-like)
- Microstructure proxies (order flow, Kyle's Lambda)
- Ichimoku with strict point-in-time compliance
- Temporal features (US market hours, day-of-week)

## Architecture

```
Data Layer (fetch, validate)
    ↓
Feature Engineering (48+ technical/microstructure features)
    ↓
Regime Detection (volatility, trend, skew)
    ↓
Model Training (ensemble gradient boosting + random forest)
    ↓
Risk Management (Kelly Criterion, position sizing)
    ↓
Backtest Engine (fee/slippage accounting, Sharpe/Sortino/Calmar)
    ↓
Paper Trading (order simulation, equity tracking)
    ↓
Monitoring & Logging (structured events, alerts)
```

## Quick Start

### Installation

```bash
git clone https://github.com/parsa314/modular-crypto-trading-bot
cd modular-crypto-trading-bot

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Run Baseline

```bash
python scripts/run_baseline.py \
  --exchange coinex \
  --symbol BTC/USDT \
  --timeframe 4h \
  --limit 2000 \
  --fee-bps 10 \
  --slippage-bps 2 \
  --output results/baseline.json
```

### Configuration

Set environment variables or create `config.yaml`:

```yaml
exchange: coinex
symbol: BTC/USDT
timeframe: 4h
limit_bars: 2000
fee_bps: 10.0
slippage_bps: 2.0
initial_capital: 10000.0
max_drawdown_pct: 0.20
```

## Core Modules

### `research_bot/features.py`
- Technical indicators: RSI, ATR, momentum, volatility
- Ichimoku cloud (strict point-in-time)
- Temporal encoding (hour/day-of-week)
- **No look-ahead bias guaranteed**

### `research_bot/model.py`
- `fit_predict_holdout()`: Chronological train/test split
- `positions_from_probabilities()`: Threshold-based position generation
- Proper imputation and scaling in pipeline
- Cross-validation ready

### `research_bot/backtest.py`
- `backtest_positions()`: Simulate trades with costs
- Metrics: Sharpe, Sortino, Calmar, max drawdown
- Turnover and friction calculation

### `research_bot/risk_management.py`
- Kelly Criterion position sizing
- Drawdown circuit breakers
- Equity-based position scaling

### `research_bot/paper_trading.py`
- Live order simulation
- Commission and slippage modeling
- Portfolio valuation

### `research_bot/validators.py`
- OHLCV integrity checks
- Outlier detection
- Feature matrix completeness

### `research_bot/monitoring.py`
- Structured logging
- Event tracking
- Error reporting with context

## Key Research Limitations

❌ **This is NOT ready for live trading without:**
1. Real historical profitability validation (full dataset, 2+ years)
2. Out-of-sample testing on held-out periods
3. Live paper trading on actual exchange feeds
4. Risk officer review and approval
5. Capital preservation policies

⚠️ **Known risks:**
- Past performance ≠ future results
- Crypto markets are 24/7 and highly volatile
- Slippage/fees can be higher during stress
- No authenticated order placement (paper trading only)
- Feature selection based on historical regime

## Scientific References

This implementation draws from:
- Ichimoku: Goichi Hosoda (1969)
- Regime detection: Hamilton (1989), MS-GARCH models
- Order flow: Kyle (1985), Blume & Easley (2012)
- Walk-forward validation: de Prado (2018)
- Kelly Criterion: Kelly (1956), MacLean et al. (2011)

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=research_bot

# Run specific test class
pytest tests/test_production.py::TestEndToEnd -v

# Run with logging
pytest tests/ -v -s
```

## Contributing

Contributions welcome! Areas of interest:
- Additional alpha factors (on-chain metrics, volatility smile)
- RL policy training (PPO, A3C)
- Multi-symbol portfolio optimization
- Real-time WebSocket data integration
- Ensemble calibration improvements

## License

MIT License — see LICENSE file

## Disclaimer

**This software is for research and educational purposes only.** The authors assume no responsibility for trading losses. Always test backtests thoroughly, validate on out-of-sample data, and paper trade before risking real capital.

Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results.
