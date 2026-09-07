"""Data validation and sanity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate OHLCV data for issues and inconsistencies."""

    @staticmethod
    def check_ohlcv_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check OHLCV data for integrity issues."""
        issues = []
        
        # Check required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
        
        if issues:
            return False, issues
        
        # Check OHLC logic
        invalid_high = df[df["high"] < df["low"]]
        if len(invalid_high) > 0:
            issues.append(f"High < Low in {len(invalid_high)} rows")
        
        invalid_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
        if len(invalid_close) > 0:
            issues.append(f"Close outside high/low in {len(invalid_close)} rows")
        
        # Check for gaps
        if "timestamp" in df.columns:
            df_sorted = df.sort_values("timestamp")
            diffs = df_sorted["timestamp"].diff()
            if (diffs > pd.Timedelta(hours=5)).any():
                issues.append("Large timestamp gaps detected")
        
        # Check for negative volumes
        if (df["volume"] < 0).any():
            issues.append("Negative volumes detected")
        
        return len(issues) == 0, issues

    @staticmethod
    def detect_outliers(df: pd.DataFrame, col: str = "close", z_threshold: float = 5.0) -> int:
        """Detect statistical outliers."""
        returns = df[col].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outliers = (z_scores > z_threshold).sum()
        
        if outliers > 0:
            logger.warning(f"Detected {outliers} outliers in {col}")
        
        return outliers

    @staticmethod
    def validate_feature_matrix(X: pd.DataFrame, min_valid_pct: float = 0.8) -> Tuple[bool, str]:
        """Check feature matrix for completeness."""
        n_rows, n_cols = X.shape
        n_nulls = X.isnull().sum().sum()
        valid_pct = 1 - (n_nulls / (n_rows * n_cols))
        
        if valid_pct < min_valid_pct:
            return False, f"Only {valid_pct:.1%} valid data"
        
        # Check for constant columns
        constant_cols = [col for col in X.columns if X[col].std() == 0]
        if constant_cols:
            return False, f"Constant columns: {constant_cols}"
        
        return True, "Feature matrix valid"
