"""Production configuration and settings."""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TradingConfig:
    """Configuration for trading bot."""
    
    # Data
    exchange: str = "coinex"
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    limit_bars: int = 2000
    
    # Costs
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    
    # Model
    train_fraction: float = 0.70
    model_threshold_upper: float = 0.56
    model_threshold_lower: float = 0.44
    allow_shorts: bool = False
    
    # Risk Management
    initial_capital: float = 10000.0
    max_drawdown_pct: float = 0.20
    kelly_fraction: float = 0.25
    max_leverage: float = 1.0
    
    # Output
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load config from environment variables."""
        return cls(
            exchange=os.getenv("TRADING_EXCHANGE", "coinex"),
            symbol=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
            timeframe=os.getenv("TRADING_TIMEFRAME", "4h"),
            limit_bars=int(os.getenv("TRADING_LIMIT_BARS", 2000)),
            fee_bps=float(os.getenv("TRADING_FEE_BPS", 10.0)),
            slippage_bps=float(os.getenv("TRADING_SLIPPAGE_BPS", 2.0)),
            initial_capital=float(os.getenv("TRADING_INITIAL_CAPITAL", 10000.0)),
            max_drawdown_pct=float(os.getenv("TRADING_MAX_DRAWDOWN", 0.20)),
        )


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    n_folds: int = 5
    embargo_pct: float = 0.01
    test_size_pct: float = 0.30
    random_state: int = 42
    
    # Walk-forward
    walk_forward_window: int = 100
    walk_forward_step: int = 20


def get_config(config_path: Optional[str] = None) -> TradingConfig:
    """Get configuration from file or environment."""
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return TradingConfig(**config_dict)
    return TradingConfig.from_env()
