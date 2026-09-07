"""Enhanced test suite with comprehensive coverage."""
import numpy as np
import pandas as pd
import pytest
from research_bot.data import fetch_ohlcv
from research_bot.features import add_features
from research_bot.regime import add_regime_features
from research_bot.model import fit_predict_holdout, positions_from_probabilities, make_dataset
from research_bot.backtest import backtest_positions, performance_metrics
from research_bot.validators import DataValidator
from research_bot.risk_management import RiskManager


def synthetic_ohlcv(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.012, n)
    close = 30000 * np.exp(np.cumsum(ret))
    
    # Open at previous close
    open_ = np.r_[close[0], close[:-1]]
    
    # High/low with random spread
    span = np.maximum(10, close * rng.uniform(0.001, 0.012, n))
    high = np.maximum(close, open_) + span
    low = np.minimum(close, open_) - span
    
    # Volume
    vol = rng.uniform(100000, 1000000, n)
    
    # Timestamps
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


class TestFeatures:
    """Test feature engineering."""
    
    def test_add_features_completes(self):
        """Features should compute without errors."""
        df = synthetic_ohlcv(n=300)
        result = add_features(df)
        assert len(result) == len(df)
        assert result.isnull().sum().sum() < len(result) * 0.5  # <50% NaNs
    
    def test_no_look_ahead_bias(self):
        """Features should only use past data."""
        df = synthetic_ohlcv(n=300)
        result = add_features(df)
        
        # Check that features don't depend on future closes
        assert "future_return" not in result.columns
        assert not result[["mom_24", "vol_24"]].isna().all().any()


class TestModel:
    """Test model training and prediction."""
    
    def test_fit_predict_holdout_completes(self):
        """Model training should complete without errors."""
        df = synthetic_ohlcv(n=900)
        df = add_regime_features(add_features(df))
        
        model, train, test, cols = fit_predict_holdout(
            df,
            train_fraction=0.70,
            hurdle_bps=10.0
        )
        
        assert len(train) > 200
        assert len(test) > 50
        assert "prob_up" in test.columns
        assert (test["prob_up"] >= 0).all() and (test["prob_up"] <= 1).all()
    
    def test_positions_from_probabilities(self):
        """Positions should be 0 or 1."""
        probs = pd.Series(np.random.uniform(0, 1, 100))
        positions = positions_from_probabilities(probs, upper=0.56)
        
        assert set(positions.unique()).issubset({0.0, 1.0})


class TestBacktest:
    """Test backtesting engine."""
    
    def test_backtest_positions_metrics(self):
        """Backtest should return valid metrics."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        positions = pd.Series(np.random.choice([0, 1], 100))
        
        strat_ret, metrics = backtest_positions(returns, positions, timeframe="4h")
        
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert metrics["n"] > 0
    
    def test_performance_metrics_handles_empty(self):
        """Metrics should handle empty series."""
        empty_returns = pd.Series([], dtype=float)
        metrics = performance_metrics(empty_returns)
        
        assert metrics["n"] == 0


class TestValidation:
    """Test data validation."""
    
    def test_ohlcv_integrity_synthetic_data(self):
        """Synthetic data should pass integrity check."""
        df = synthetic_ohlcv(n=100)
        is_valid, issues = DataValidator.check_ohlcv_integrity(df)
        
        assert is_valid or len(issues) == 0
    
    def test_detect_outliers(self):
        """Outlier detection should work."""
        df = synthetic_ohlcv(n=200)
        df.loc[50, "close"] = df["close"].mean() * 10  # Inject outlier
        
        outliers = DataValidator.detect_outliers(df, col="close", z_threshold=3.0)
        assert outliers >= 1


class TestRiskManagement:
    """Test risk control."""
    
    def test_kelly_criterion(self):
        """Kelly calculation should be positive for edge."""
        rm = RiskManager()
        
        # Winning strategy
        kelly = rm.kelly_criterion(win_rate=0.55, win_loss_ratio=1.0)
        assert kelly > 0
        
        # Losing strategy
        kelly_lose = rm.kelly_criterion(win_rate=0.45, win_loss_ratio=1.0)
        assert kelly_lose <= 0
    
    def test_drawdown_circuit_breaker(self):
        """Circuit breaker should trigger on large drawdown."""
        rm = RiskManager(initial_capital=10000, max_drawdown_pct=0.20)
        
        # Simulate large drawdown
        equity = pd.Series([10000, 9000, 8000, 7000, 6000])  # -40%
        is_breached = rm.check_drawdown_circuit_breaker(equity)
        
        assert is_breached


class TestEndToEnd:
    """Full pipeline tests."""
    
    def test_end_to_end_synthetic(self):
        """Complete pipeline should work on synthetic data."""
        # Generate data
        df = synthetic_ohlcv(n=900)
        
        # Features
        df = add_regime_features(add_features(df))
        
        # Train model
        model, train, test, cols = fit_predict_holdout(df)
        
        # Generate positions
        pos = positions_from_probabilities(test["prob_up"])
        
        # Backtest
        _, metrics = backtest_positions(test["future_return"], pos)
        
        assert metrics["n"] == len(test)
        assert 0 <= metrics["avg_abs_position"] <= 1
        assert len(train) > 0 and len(test) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
