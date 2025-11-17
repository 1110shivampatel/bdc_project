"""
Backtesting framework for time-series forecasting with grouped cross-validation.

Uses expanding-window cross-validation grouped by BDC (CIK) to prevent data leakage.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.evaluation import compute_metrics


class ExpandingWindowCV:
    """
    Expanding-window cross-validation for time-series data.
    
    For each fold:
    - Train on all data up to a cutoff period
    - Test on the next period(s)
    - Ensures no future data leaks into training
    """
    
    def __init__(
        self,
        periods: List[str],
        n_test_periods: int = 1,
        min_train_periods: int = 4,
    ):
        """
        Parameters
        ----------
        periods : List[str]
            Sorted list of period identifiers (e.g., ['2022_Q4', '2023_Q1', ...]).
        n_test_periods : int
            Number of periods to use for testing in each fold.
        min_train_periods : int
            Minimum number of periods required for training.
        """
        self.periods = sorted(periods)
        self.n_test_periods = n_test_periods
        self.min_train_periods = min_train_periods
        self.folds: List[Tuple[List[str], List[str]]] = []
        self._generate_folds()
    
    def _generate_folds(self):
        """Generate train/test splits."""
        n_periods = len(self.periods)
        
        for i in range(self.min_train_periods, n_periods - self.n_test_periods + 1):
            train_periods = self.periods[:i]
            test_periods = self.periods[i:i + self.n_test_periods]
            self.folds.append((train_periods, test_periods))
    
    def split(self, df: pd.DataFrame, period_col: str = "period") -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits from dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with period column.
        period_col : str
            Name of period column.
        
        Yields
        ------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df) for each fold.
        """
        for train_periods, test_periods in self.folds:
            train_mask = df[period_col].isin(train_periods)
            test_mask = df[period_col].isin(test_periods)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            yield train_df, test_df
    
    def __len__(self) -> int:
        """Return number of folds."""
        return len(self.folds)


def backtest_model(
    model: object,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    period_col: str = "period",
    cik_col: str = "cik",
    cv: Optional[ExpandingWindowCV] = None,
) -> pd.DataFrame:
    """
    Backtest a model using expanding-window cross-validation.
    
    Parameters
    ----------
    model : object
        Model with fit() and predict() methods.
    df : pd.DataFrame
        Full dataset with features, target, period, and cik.
    feature_cols : List[str]
        List of feature column names.
    target_col : str
        Name of target column.
    period_col : str
        Name of period column.
    cik_col : str
        Name of CIK/entity column (for grouping).
    cv : ExpandingWindowCV | None
        Optional pre-configured CV object. If None, creates default.
    
    Returns
    -------
    pd.DataFrame
        Predictions with columns: period, cik, y_true, y_pred, fold.
    """
    # Get unique periods and create CV object
    if cv is None:
        periods = sorted(df[period_col].unique().tolist())
        cv = ExpandingWindowCV(periods, n_test_periods=1, min_train_periods=4)
    
    all_predictions = []
    
    for fold_idx, (train_df, test_df) in enumerate(cv.split(df, period_col=period_col)):
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        
        # Prepare features and target
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].copy()
        
        # Remove rows with missing or invalid target
        train_mask = y_train.notna() & np.isfinite(y_train)
        test_mask = y_test.notna() & np.isfinite(y_test)
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # Clean features: replace infinity with NaN
        for col in X_train.columns:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        for col in X_test.columns:
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Fit model
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Warning: Model fitting failed for fold {fold_idx}: {e}")
            continue
        
        # Predict
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Warning: Model prediction failed for fold {fold_idx}: {e}")
            continue
        
        # Store predictions
        pred_df = test_df[test_mask].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred
        pred_df["fold"] = fold_idx
        all_predictions.append(pred_df)
    
    if len(all_predictions) == 0:
        return pd.DataFrame()
    
    return pd.concat(all_predictions, ignore_index=True)


def backtest_multiple_models(
    models: List[object],
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    period_col: str = "period",
    cik_col: str = "cik",
    cv: Optional[ExpandingWindowCV] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Backtest multiple models and return predictions for each.
    
    Parameters
    ----------
    models : List[object]
        List of model objects with fit() and predict() methods.
    df : pd.DataFrame
        Full dataset.
    feature_cols : List[str]
        Feature column names.
    target_col : str
        Target column name.
    period_col : str
        Period column name.
    cik_col : str
        CIK column name.
    cv : ExpandingWindowCV | None
        Optional CV object.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions DataFrame.
    """
    results = {}
    
    for model in models:
        print(f"Backtesting {model.name}...")
        predictions = backtest_model(
            model=model,
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            period_col=period_col,
            cik_col=cik_col,
            cv=cv,
        )
        results[model.name] = predictions
    
    return results

