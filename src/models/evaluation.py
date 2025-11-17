"""
Model evaluation metrics and utilities.

Computes MAE, RMSE, MAPE and other metrics for regression tasks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values.
    """
    # Remove NaN/inf values for metric computation
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "n_samples": 0,
        }
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    # Mean Absolute Percentage Error
    # Handle division by zero
    abs_true = np.abs(y_true_clean)
    mape_mask = abs_true > 1e-10  # Avoid division by very small numbers
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
    else:
        mape = np.nan
    
    # R-squared
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else np.nan,
        "r2": float(r2) if not np.isnan(r2) else np.nan,
        "n_samples": int(len(y_true_clean)),
    }


def evaluate_model_predictions(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate model predictions grouped by entity or period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with true and predicted values.
    y_true_col : str
        Column name for true values.
    y_pred_col : str
        Column name for predicted values.
    group_by : str | None
        Optional column to group by (e.g., 'cik', 'period').
    
    Returns
    -------
    pd.DataFrame
        Metrics per group (or overall if group_by is None).
    """
    if group_by is None:
        metrics = compute_metrics(df[y_true_col].values, df[y_pred_col].values)
        return pd.DataFrame([metrics])
    
    results = []
    for group_val, group_df in df.groupby(group_by):
        metrics = compute_metrics(group_df[y_true_col].values, group_df[y_pred_col].values)
        metrics[group_by] = group_val
        results.append(metrics)
    
    return pd.DataFrame(results)


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models' performance metrics.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict[str, float]]
        Dictionary mapping model name to metrics dictionary.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    return pd.DataFrame(results_dict).T

