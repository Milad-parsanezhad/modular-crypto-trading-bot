from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import logging

from .features import FEATURE_COLUMNS
from .regime import REGIME_COLUMNS

logger = logging.getLogger(__name__)


def make_dataset(
    df: pd.DataFrame,
    hurdle_bps: float = 0.0,
) -> Tuple[pd.DataFrame, list]:
    """Create training dataset with proper forward-looking target.
    
    CRITICAL: Target computed from NEXT bar's close, not current.
    No look-ahead bias.
    """
    x = df.copy()
    
    # Forward-looking return (what happens NEXT bar)
    x["future_return"] = x["close"].shift(-1) / x["close"] - 1
    
    # Hurdle: only long if future return > hurdle
    hurdle = hurdle_bps / 10000.0
    x["target_up"] = (x["future_return"] > hurdle).astype(int)
    
    # Select available features
    cols = [c for c in FEATURE_COLUMNS + REGIME_COLUMNS if c in x.columns]
    
    # Drop rows with missing target
    x = x.dropna(subset=["future_return"])
    
    logger.info(f"Dataset: {len(x)} rows, {len(cols)} features, target mean: {x['target_up'].mean():.2%}")
    return x, cols


def fit_predict_holdout(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    hurdle_bps: float = 0.0,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, list]:
    """Chronological split holdout validation (no lookahead)."""
    data, cols = make_dataset(df, hurdle_bps=hurdle_bps)
    
    if len(data) < 300:
        raise ValueError("Need at least 300 usable rows for research holdout.")
    
    # Chronological split
    split = int(len(data) * train_fraction)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    
    # Build pipeline with scaling
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            validation_fraction=0.1,
            early_stopping=True,
            random_state=random_state,
        ))
    ])
    
    # Train
    model.fit(train[cols], train["target_up"])
    
    # Predict on test (never train on test)
    test["prob_up"] = model.predict_proba(test[cols])[:, 1]
    
    return model, train, test, cols


def positions_from_probabilities(
    prob_up: pd.Series,
    upper: float = 0.56,
    lower: float = 0.44,
    allow_short: bool = False,
) -> pd.Series:
    """Convert model probabilities to trading positions."""
    p = pd.Series(prob_up)
    
    if allow_short:
        positions = pd.Series(
            np.where(p > upper, 1.0, np.where(p < lower, -1.0, 0.0)),
            index=p.index,
        )
    else:
        positions = (p > upper).astype(float)
    
    return positions
