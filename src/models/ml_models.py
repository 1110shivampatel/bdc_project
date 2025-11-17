"""
Machine learning models for BDC forecasting.

Advanced models for better predictive performance:
1. XGBoost
2. LightGBM
3. Random Forest
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class XGBoostForecaster:
    """
    XGBoost gradient boosting model.
    
    Handles missing values natively and provides feature importance.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds.
        max_depth : int
            Maximum tree depth.
        learning_rate : float
            Learning rate (shrinkage).
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.name = f"XGBoost_n{n_estimators}_d{max_depth}_lr{learning_rate}"
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[list] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit XGBoost model."""
        # Select numeric features
        X_clean = X.select_dtypes(include=[np.number]).copy()
        self.feature_names = X_clean.columns.tolist()
        
        # XGBoost handles NaN natively, but we'll fill with median for consistency
        X_clean = X_clean.fillna(X_clean.median())
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_clean, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fitted XGBoost model."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X[self.feature_names].copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        return self.model.predict(X_clean)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)


class LightGBMForecaster:
    """
    LightGBM gradient boosting model.
    
    Fast training and good performance on tabular data.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds.
        max_depth : int
            Maximum tree depth.
        learning_rate : float
            Learning rate (shrinkage).
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.name = f"LightGBM_n{n_estimators}_d{max_depth}_lr{learning_rate}"
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: Optional[list] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit LightGBM model."""
        # Select numeric features
        X_clean = X.select_dtypes(include=[np.number]).copy()
        self.feature_names = X_clean.columns.tolist()
        
        # LightGBM handles NaN natively
        X_clean = X_clean.fillna(X_clean.median())
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X_clean, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fitted LightGBM model."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X[self.feature_names].copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        return self.model.predict(X_clean)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)


class RandomForestForecaster:
    """
    Random Forest ensemble model.
    
    Robust to overfitting and provides feature importance.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int | None
            Maximum tree depth (None = no limit).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.name = f"RandomForest_n{n_estimators}_d{max_depth or 'None'}"
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: Optional[list] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit Random Forest model."""
        # Select numeric features
        X_clean = X.select_dtypes(include=[np.number]).copy()
        self.feature_names = X_clean.columns.tolist()
        
        # Fill NaN with median
        X_clean = X_clean.fillna(X_clean.median())
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_clean, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fitted Random Forest model."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X[self.feature_names].copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        return self.model.predict(X_clean)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)


def get_ml_models() -> list:
    """
    Return list of ML model instances.
    
    Returns
    -------
    list
        List of ML model objects (only includes available models).
    """
    models = []
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models.extend([
            XGBoostForecaster(n_estimators=50, max_depth=4, learning_rate=0.1),
            XGBoostForecaster(n_estimators=100, max_depth=6, learning_rate=0.1),
            XGBoostForecaster(n_estimators=200, max_depth=6, learning_rate=0.05),
        ])
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models.extend([
            LightGBMForecaster(n_estimators=50, max_depth=4, learning_rate=0.1),
            LightGBMForecaster(n_estimators=100, max_depth=6, learning_rate=0.1),
            LightGBMForecaster(n_estimators=200, max_depth=6, learning_rate=0.05),
        ])
    
    # Random Forest (always available)
    models.extend([
        RandomForestForecaster(n_estimators=50, max_depth=10),
        RandomForestForecaster(n_estimators=100, max_depth=15),
        RandomForestForecaster(n_estimators=200, max_depth=20),
    ])
    
    return models

