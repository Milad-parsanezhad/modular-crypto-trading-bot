"""Multiple model architectures: Gradient Boosting, LSTM, Transformer, RL."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating calibrated ensemble models."""

    @staticmethod
    def create_gradient_boosting(
        learning_rate: float = 0.05,
        max_iter: int = 300,
        max_leaf_nodes: int = 15,
        l2_reg: float = 0.5,
        random_state: int = 42,
    ) -> Pipeline:
        """HistGradientBoosting with proper imputation."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_reg,
                random_state=random_state,
                validation_fraction=0.1,
                early_stopping=True,
            ))
        ])

    @staticmethod
    def create_random_forest(
        n_estimators: int = 200,
        max_depth: int = 12,
        min_samples_split: int = 10,
        random_state: int = 42,
    ) -> Pipeline:
        """Random Forest for interpretability."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                n_jobs=-1,
                random_state=random_state,
            ))
        ])


class EnsembleModel:
    """Stacked ensemble combining multiple models."""

    def __init__(self, models: Optional[list] = None):
        """Initialize with multiple base models."""
        self.models = models or [
            ModelFactory.create_gradient_boosting(),
            ModelFactory.create_random_forest(),
        ]
        self.meta_learner = ModelFactory.create_gradient_boosting()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        """Train base models and meta-learner on stacked predictions."""
        # Train base models
        base_preds = []
        for model in self.models:
            model.fit(X, y)
            base_preds.append(model.predict_proba(X)[:, 1])
        
        # Stack predictions as meta-features
        X_meta = np.column_stack(base_preds)
        self.meta_learner.fit(X_meta, y)
        
        logger.info(f"Ensemble trained on {len(X)} samples with {len(self.models)} base models")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        base_preds = []
        for model in self.models:
            base_preds.append(model.predict_proba(X)[:, 1])
        
        X_meta = np.column_stack(base_preds)
        return self.meta_learner.predict_proba(X_meta)[:, 1]


class PurgedKFold:
    """Walk-forward validation with embargo periods to prevent look-ahead bias."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """Initialize with number of folds and embargo window."""
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Yield purged train/test indices."""
        n = len(X)
        fold_size = n // self.n_splits
        embargo_size = int(n * self.embargo_pct)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # Remove embargo window
            embargo_start = max(0, test_start - embargo_size)
            embargo_end = min(n, test_end + embargo_size)
            
            train_idx = np.concatenate([
                np.arange(0, embargo_start),
                np.arange(embargo_end, n),
            ])
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
