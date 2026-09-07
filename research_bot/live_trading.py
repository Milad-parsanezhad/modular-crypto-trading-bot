"""Production-ready live trading engine with real exchange integration."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt

from research_bot.data import fetch_ohlcv
from research_bot.features import add_features
from research_bot.regime import add_regime_features
from research_bot.model import fit_predict_holdout, positions_from_probabilities
from research_bot.backtest import backtest_positions, performance_metrics
from research_bot.risk_management import RiskManager
from research_bot.validators import DataValidator
from research_bot.monitoring import StructuredLogger

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    timestamp: datetime
    symbol: str
    side: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    price: float
    quantity: float
    stop_loss: float
    take_profit: float
    reason: str


@dataclass
class LiveTradeConfig:
    """Configuration for live trading."""
    exchange: str
    symbol: str
    timeframe: str
    api_key: str
    api_secret: str
    
    # Trading parameters
    initial_capital: float = 1000.0  # Min capital for safety
    position_size_pct: float = 0.50  # Use 50% of available capital (conservative)
    fee_bps: float = 10.0
    slippage_bps: float = 5.0  # Real slippage
    
    # Risk management
    max_drawdown_pct: float = 0.15  # Stop at 15% loss
    stop_loss_pct: float = 0.05  # 5% stop loss per trade
    take_profit_pct: float = 0.10  # 10% take profit target
    max_positions: int = 1  # One position at a time
    
    # Safety limits
    max_daily_loss_pct: float = 0.20  # Stop trading if lose 20% daily
    max_trades_per_day: int = 5
    min_volume_24h: float = 1000000.0  # Min daily volume ($)
    
    # Updates
    signal_update_hours: int = 4
    risk_check_minutes: int = 5


class LiveTradingEngine:
    """Production live trading engine with safety checks."""

    def __init__(self, config: LiveTradeConfig):
        """Initialize live trading engine."""
        self.config = config
        self.exchange = self._init_exchange()
        self.risk_manager = RiskManager(
            initial_capital=config.initial_capital,
            max_drawdown_pct=config.max_drawdown_pct,
        )
        self.logger = StructuredLogger(f"live_trading_{config.symbol}")
        
        # State tracking
        self.current_position = 0.0
        self.entry_price = 0.0
        self.trades_today = 0
        self.daily_loss = 0.0
        self.last_signal_time = None
        self.signal_history: List[TradeSignal] = []
        self.performance_history = []
        
        # Model
        self.model = None
        self.model_cols = None
        self.last_model_update = None
        
        logger.info(f"Live trading engine initialized for {config.symbol}")

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange with credentials."""
        exchange_class = getattr(ccxt, self.config.exchange)
        exchange = exchange_class({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "enableRateLimit": True,
            "timeout": 30000,
        })
        
        try:
            balance = exchange.fetch_balance()
            logger.info(f"Exchange connected. Free USDT: {balance['free'].get('USDT', 0):.2f}")
            return exchange
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
            raise

    def check_market_conditions(self) -> bool:
        """Verify market is safe to trade."""
        try:
            # Check volume
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            volume_24h = ticker.get("quoteVolume", 0)
            
            if volume_24h < self.config.min_volume_24h:
                logger.warning(f"Low volume: {volume_24h:.0f} < {self.config.min_volume_24h:.0f}")
                return False
            
            # Check bid-ask spread
            spread = (ticker["ask"] - ticker["bid"]) / ticker["bid"]
            if spread > 0.001:  # >0.1% spread
                logger.warning(f"Wide spread: {spread*100:.2f}%")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Market check failed: {e}")
            return False

    def fetch_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Fetch latest OHLCV and prepare features."""
        try:
            # Fetch with safety buffer
            df = fetch_ohlcv(
                self.config.exchange,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                limit=300  # Extra bars for indicators
            )
            
            # Validate
            is_valid, issues = DataValidator.check_ohlcv_integrity(df)
            if not is_valid:
                logger.warning(f"Data validation issues: {issues}")
                return None
            
            # Add features
            df = add_regime_features(add_features(df))
            
            return df
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return None

    def update_model(self, df: pd.DataFrame) -> bool:
        """Retrain model with latest data."""
        try:
            # Only retrain every N hours
            if self.last_model_update is not None:
                hours_since = (datetime.now() - self.last_model_update).total_seconds() / 3600
                if hours_since < self.config.signal_update_hours:
                    return True  # Use existing model
            
            # Train
            model, train, test, cols = fit_predict_holdout(
                df,
                train_fraction=0.70,
                hurdle_bps=self.config.fee_bps + self.config.slippage_bps
            )
            
            self.model = model
            self.model_cols = cols
            self.last_model_update = datetime.now()
            
            logger.info(f"Model updated: {len(train)} train, {len(test)} test")
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def generate_signal(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal from latest bar."""
        if self.model is None or self.model_cols is None:
            return None
        
        try:
            # Get latest bar
            latest = df.iloc[-1:].copy()
            price = float(latest["close"].iloc[0])
            timestamp = pd.to_datetime(latest["timestamp"].iloc[0], utc=True).to_pydatetime()
            
            # Predict
            X = latest[self.model_cols]
            prob_up = self.model.predict_proba(X)[0, 1]
            
            # Generate position
            if prob_up > 0.56:  # Require high confidence
                position = 1.0  # Long
                side = "BUY"
                confidence = prob_up
                reason = f"High confidence upside: {prob_up:.2%}"
            elif prob_up < 0.40:
                position = 0.0  # Exit
                side = "SELL" if self.current_position > 0 else "HOLD"
                confidence = 1 - prob_up
                reason = f"Low confidence: {prob_up:.2%}"
            else:
                return None  # No clear signal
            
            # Calculate risk levels
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
            
            # Calculate quantity
            balance = self.exchange.fetch_balance()
            available_usdt = balance["free"].get("USDT", 0)
            quantity = (available_usdt * self.config.position_size_pct) / price
            
            signal = TradeSignal(
                timestamp=timestamp,
                symbol=self.config.symbol,
                side=side,
                confidence=confidence,
                price=price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
            )
            
            return signal
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return None

    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute trade with safety checks."""
        try:
            # Safety checks
            if self.trades_today >= self.config.max_trades_per_day:
                logger.warning(f"Max trades reached: {self.trades_today}")
                return False
            
            if self.daily_loss <= -self.config.initial_capital * self.config.max_daily_loss_pct:
                logger.warning("Max daily loss reached. Stopping trading.")
                return False
            
            if signal.side == "BUY":
                # Place buy order
                order = self.exchange.create_limit_buy_order(
                    self.config.symbol,
                    signal.quantity,
                    signal.price
                )
                
                self.current_position = signal.quantity
                self.entry_price = signal.price
                self.trades_today += 1
                
                self.logger.log_event("trade_executed", asdict(signal))
                logger.info(f"BUY: {signal.quantity} @ {signal.price:.2f} USD")
                
                return True
            
            elif signal.side == "SELL" and self.current_position > 0:
                # Place sell order
                order = self.exchange.create_limit_sell_order(
                    self.config.symbol,
                    self.current_position,
                    signal.price
                )
                
                pnl = (signal.price - self.entry_price) * self.current_position
                self.daily_loss += pnl
                self.current_position = 0.0
                self.trades_today += 1
                
                self.logger.log_event("position_closed", {
                    "pnl": pnl,
                    "pnl_pct": (pnl / (self.entry_price * self.current_position) * 100) if self.current_position > 0 else 0
                })
                logger.info(f"SELL: Realized PnL: {pnl:.2f} USD")
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.logger.log_event("trade_error", {"error": str(e)})
            return False

    def check_position_safety(self) -> bool:
        """Check if position needs emergency exit."""
        if self.current_position == 0:
            return True
        
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            current_price = ticker["last"]
            
            # Check stop loss
            stop_loss = self.entry_price * (1 - self.config.stop_loss_pct)
            if current_price <= stop_loss:
                logger.warning(f"Stop loss triggered at {current_price:.2f}")
                return False
            
            # Check take profit
            take_profit = self.entry_price * (1 + self.config.take_profit_pct)
            if current_price >= take_profit:
                logger.info(f"Take profit triggered at {current_price:.2f}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Position safety check failed: {e}")
            return False

    async def run_trading_loop(self):
        """Main async trading loop."""
        logger.info(f"Starting live trading loop for {self.config.symbol}")
        
        while True:
            try:
                # 1. Check market conditions
                if not self.check_market_conditions():
                    logger.warning("Market conditions unfavorable. Skipping signal generation.")
                    await asyncio.sleep(60)
                    continue
                
                # 2. Check position safety
                if not self.check_position_safety():
                    logger.warning("Position safety check failed. Emergency exit triggered.")
                
                # 3. Fetch and prepare data
                df = self.fetch_and_prepare_data()
                if df is None:
                    await asyncio.sleep(60)
                    continue
                
                # 4. Update model
                if not self.update_model(df):
                    await asyncio.sleep(60)
                    continue
                
                # 5. Generate signal
                signal = self.generate_signal(df)
                if signal is not None:
                    logger.info(f"Signal: {signal.side} {signal.reason}")
                    
                    # 6. Execute trade
                    if self.execute_trade(signal):
                        self.signal_history.append(signal)
                
                # 7. Wait for next update
                await asyncio.sleep(self.config.risk_check_minutes * 60)
                
            except Exception as e:
                logger.exception(f"Trading loop error: {e}")
                self.logger.log_event("loop_error", {"error": str(e)})
                await asyncio.sleep(300)  # Wait 5 min before retry

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        try:
            balance = self.exchange.fetch_balance()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.config.symbol,
                "current_position": self.current_position,
                "entry_price": self.entry_price,
                "available_usdt": balance["free"].get("USDT", 0),
                "trades_today": self.trades_today,
                "daily_pnl": self.daily_loss,
                "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {}
