"""Production deployment guide and checklist."""
# Production Deployment Guide - نگاهداری تولیدی

## Pre-Deployment Checklist (CRITICAL!) - لیست بررسی قبل از استقرار

### 1. Risk Management Setup - تنظیم مدیریت ریسک
- [ ] Set `max_drawdown_pct` to 0.15 (15% max loss)
- [ ] Set `max_daily_loss_pct` to 0.20 (20% daily max)
- [ ] Set `stop_loss_pct` to 0.05 (5% per trade)
- [ ] Set `take_profit_pct` to 0.10 (10% profit target)
- [ ] Start with **minimum capital** ($100-500 USD)
- [ ] Set `position_size_pct` to 0.50 (50% only, not 95%)
- [ ] Max 1 position at a time
- [ ] Max 5 trades per day

### 2. Exchange Configuration - تنظیم صرافی
- [ ] Use **Testnet ONLY** for first 2 weeks
- [ ] Test API credentials work correctly
- [ ] Verify API has:
  - Read-only access to balances
  - Post-only access to orders
  - No withdrawal permissions
- [ ] Set API IP whitelist to your VPS only
- [ ] Use API keys with minimal scope (no margin trading)

### 3. Model Validation - اعتبارسنجی مدل
- [ ] Backtest on 12 months of historical data
- [ ] Check Sharpe ratio > 0.5 (minimum)
- [ ] Check win rate > 45% (break-even)
- [ ] Check max drawdown < 25%
- [ ] Test on different market conditions (bull, bear, sideways)

### 4. Data Quality - کیفیت داده
- [ ] Verify OHLCV data integrity for past 2 years
- [ ] Check for gaps in data
- [ ] Verify volume data is reasonable
- [ ] Test on multiple exchange sources (CCXT fallback)

### 5. Infrastructure - زیرساخت
- [ ] Use VPS with uptime SLA > 99.9%
- [ ] Set up automatic restart on failure
- [ ] Enable logs and monitoring
- [ ] Set up email alerts for errors
- [ ] Test graceful shutdown procedure
- [ ] Set up disk space monitoring (10GB+ free)

### 6. Monitoring Setup - تنظیم نظارت
- [ ] Real-time PnL tracking
- [ ] Daily equity curve log
- [ ] Error logging with context
- [ ] Trade execution log
- [ ] Weekly performance report

### 7. Capital Management - مدیریت سرمایه
- [ ] Start with test capital ($100-500)
- [ ] DO NOT risk rent/bill money
- [ ] Have 6-month emergency fund separate
- [ ] Set mental stop-loss at -30% (QUIT if this happens)
- [ ] Reinvest only 50% of profits

## Quick Start - شروع سریع

### Step 1: Clone and Setup
```bash
git clone https://github.com/parsa314/modular-crypto-trading-bot
cd modular-crypto-trading-bot
pip install -e ".[dev]"
```

### Step 2: Configure Credentials
```bash
cp .env.example .env
# Edit with your testnet credentials
```

### Step 3: Run Tests
```bash
pytest tests/ -v
python scripts/run_baseline.py --limit 500
```

### Step 4: Start Paper Trading
```bash
python -m research_bot.live_trading
```

## Monitoring Commands - دستورات نظارت

```bash
# Check signals
tail -f logs/live_trading_*.log | grep -i signal

# Check errors
tail -f logs/live_trading_*.log | grep -i error

# Monitor trades
grep "trade_executed" logs/*.log
```

## Emergency Procedures - روش‌های اضطراری

### Position Lost 5% (Stop Loss)
- ✓ Automatic: Will sell position
- Manual: `python scripts/emergency_exit.py`

### Daily Loss > 20%
- ✓ Automatic: Stops all trading
- Re-enables next day

### Model Performance Drops
- If Sharpe < 0.3: STOP immediately
- Investigate market regime
- Retrain on latest 2 years

## Performance Targets - اهداف عملکرد

### Monthly (Conservative)
- Target return: +2-3%
- Max drawdown: -5%
- Win rate: >45%
- Sharpe ratio: >1.0

### Stop Conditions
- Month 1: -5% → Investigate
- Month 2: -10% → Stop and review
- Month 3: -20% → STOP trading

## Common Mistakes - اشتباهات رایج

❌ **Don't:**
- Use production capital immediately
- Skip backtesting
- Trade 24/7 (choose 1 timeframe)
- Use leverage or margin
- Ignore drawdown warnings
- Deploy on personal laptop
- Share API keys
- Change strategy every week
- Trade illiquid pairs
- Forget stop losses

✓ **Do:**
- Start small ($100-500)
- Paper trade for 2 weeks
- Monitor daily
- Keep detailed logs
- Review performance weekly
- Update model monthly
- Use VPS with uptime guarantee
- Enable all safety checks
- Have manual kill switch
- Reinvest profits conservatively

## Support - پشتیبانی

```bash
# Test exchange connection
python -c "import ccxt; print(ccxt.binance().fetch_ticker('BTC/USDT'))"

# Check data quality
python debug_tools.py --data results/baseline.json

# Model diagnostics
python scripts/model_diagnostics.py
```

---

**FINAL REMINDER:** This bot is for research/small-scale trading only. Start small. Test thoroughly. Monitor constantly.

**یادآوری نهایی:** این ربات فقط برای تحقیق است. کوچک شروع کنید. بسیار تست کنید. مدام نظارت کنید.
