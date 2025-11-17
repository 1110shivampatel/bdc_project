"""
Baseline models for BDC forecasting.

Simple models to establish performance benchmarks:
1. Naive (last value)
2. Rolling mean
3. Ridge regression
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class NaiveForecaster:
    """
    Naive baseline: predict next quarter = current quarter value.
    
    For percentage change: predicts 0% change.
    For absolute values: predicts same value.
    """
    
    def __init__(self):
        self.name = "Naive"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No training needed for naive model."""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict zero change (or last value if y was absolute)."""
        return np.zeros(len(X))


class RollingMeanForecaster:
    """
    Rolling mean baseline: predict next quarter = mean of last N quarters.
    
    For percentage change: predicts mean of historical changes.
    """
    
    def __init__(self, window: int = 4):
        """
        Parameters
        ----------
        window : int
            Number of previous quarters to average.
        """
        self.window = window
        self.name = f"RollingMean_{window}q"
        self.mean_value: Optional[float] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Compute mean of training target values."""
        self.mean_value = y.mean()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the mean value for all samples."""
        if self.mean_value is None:
            raise ValueError("Model must be fitted before prediction")
        return np.full(len(X), self.mean_value)


class RidgeForecaster:
    """
    Ridge regression baseline with feature scaling.
    
    L2-regularized linear regression to prevent overfitting.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Regularization strength (higher = more regularization).
        """
        self.alpha = alpha
        self.name = f"Ridge_alpha{alpha}"
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit ridge regression model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (may contain NaN).
        y : pd.Series
            Target values.
        """
        # Select numeric features and handle NaN
        X_clean = X.select_dtypes(include=[np.number]).copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fitted ridge model."""
        if self.feature_names is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Select and clean features
        X_clean = X[self.feature_names].copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean)
        
        # Predict
        return self.model.predict(X_scaled)


def get_baseline_models() -> list:
    """
    Return list of baseline model instances.
    
    Returns
    -------
    list
        List of baseline model objects.
    """
    return [
        NaiveForecaster(),
        RollingMeanForecaster(window=2),
        RollingMeanForecaster(window=4),
        RollingMeanForecaster(window=8),
        RidgeForecaster(alpha=0.1),
        RidgeForecaster(alpha=1.0),
        RidgeForecaster(alpha=10.0),
    ]

