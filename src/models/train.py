"""
Main training module for BDC forecasting models.

Orchestrates model training, backtesting, evaluation, and artifact saving.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.models.backtesting import ExpandingWindowCV, backtest_multiple_models
from src.models.baselines import get_baseline_models
from src.models.evaluation import compare_models, compute_metrics, evaluate_model_predictions
from src.models.ml_models import get_ml_models
from src.utils.io import ensure_parent_dir


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "target_next_qtr_pct",
    exclude_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Prepare data for training: select features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature panel with target column.
    target_col : str
        Name of target column.
    exclude_cols : List[str] | None
        Columns to exclude from features (e.g., identifiers, other targets).
    
    Returns
    -------
    tuple[pd.DataFrame, List[str], pd.Series]
        (feature_df, feature_cols, target_series)
    """
    if exclude_cols is None:
        exclude_cols = [
            "period",
            "cik",
            "name",
            "target_next_qtr_pct",
            "target_next_qtr_abs",
        ]
    
    # Select feature columns (numeric, not in exclude list)
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    # Prepare feature matrix and target
    feature_df = df[feature_cols + ["period", "cik"]].copy()
    target_series = df[target_col].copy()
    
    return feature_df, feature_cols, target_series


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str = "target_next_qtr_pct",
    include_baselines: bool = True,
    include_ml: bool = True,
    min_train_periods: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Train and evaluate multiple models using backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature panel with target.
    target_col : str
        Target column name.
    include_baselines : bool
        Whether to include baseline models.
    include_ml : bool
        Whether to include ML models.
    min_train_periods : int
        Minimum periods for training in CV.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions DataFrame.
    """
    # Prepare data
    feature_df, feature_cols, target_series = prepare_training_data(df, target_col)
    
    # Combine features and target
    train_df = feature_df.copy()
    train_df[target_col] = target_series
    
    # Remove rows with missing or invalid target
    train_df = train_df[train_df[target_col].notna()].copy()
    
    # Remove infinity values from target
    train_df = train_df[np.isfinite(train_df[target_col])].copy()
    
    # Filter extreme target values (likely data errors or edge cases)
    # For percentage change, reasonable range is -100% to +1000%
    # (allowing for some extreme but plausible cases)
    if "pct" in target_col.lower():
        train_df = train_df[
            (train_df[target_col] > -100) & 
            (train_df[target_col] < 1000)
        ].copy()
    else:
        # For absolute change, use a more lenient filter based on data distribution
        q1 = train_df[target_col].quantile(0.01)
        q99 = train_df[target_col].quantile(0.99)
        train_df = train_df[
            (train_df[target_col] > q1) & 
            (train_df[target_col] < q99)
        ].copy()
    
    # Clean features: replace infinity with NaN, then fill with median
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
    
    print(f"Training data: {len(train_df)} rows, {len(feature_cols)} features")
    print(f"Target: {target_col}")
    if len(train_df) > 0:
        print(f"Target stats: mean={train_df[target_col].mean():.2f}, std={train_df[target_col].std():.2f}")
    else:
        print("WARNING: No valid training data after cleaning!")
    
    # Get models
    models = []
    if include_baselines:
        models.extend(get_baseline_models())
    if include_ml:
        models.extend(get_ml_models())
    
    print(f"\nBacktesting {len(models)} models...")
    
    # Create CV object
    periods = sorted(train_df["period"].unique().tolist())
    cv = ExpandingWindowCV(periods, n_test_periods=1, min_train_periods=min_train_periods)
    print(f"Cross-validation: {len(cv)} folds")
    
    # Backtest all models
    predictions_dict = backtest_multiple_models(
        models=models,
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        cv=cv,
    )
    
    return predictions_dict


def evaluate_all_models(predictions_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate all models and return comparison table.
    
    Parameters
    ----------
    predictions_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics per model.
    """
    results = {}
    
    for model_name, pred_df in predictions_dict.items():
        if len(pred_df) == 0:
            continue
        
        metrics = compute_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
        results[model_name] = metrics
    
    return compare_models(results)


def save_model_artifacts(
    model: object,
    model_name: str,
    feature_cols: List[str],
    metrics: Dict[str, float],
    output_dir: Path,
) -> Path:
    """
    Save trained model and metadata.
    
    Parameters
    ----------
    model : object
        Trained model object.
    model_name : str
        Name of the model.
    feature_cols : List[str]
        List of feature column names.
    metrics : Dict[str, float]
        Model performance metrics.
    output_dir : Path
        Directory to save artifacts.
    
    Returns
    -------
    Path
        Path to saved model file.
    """
    ensure_parent_dir(output_dir)
    
    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    metadata_path = output_dir / f"{model_name}_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    return model_path


def save_predictions(
    predictions_dict: Dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """
    Save all model predictions to Parquet.
    
    Parameters
    ----------
    predictions_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model name to predictions.
    output_path : Path
        Path to save predictions Parquet file.
    
    Returns
    -------
    Path
        Path to saved file.
    """
    ensure_parent_dir(output_path)
    
    # Combine all predictions with model name
    all_predictions = []
    for model_name, pred_df in predictions_dict.items():
        pred_df = pred_df.copy()
        pred_df["model_name"] = model_name
        all_predictions.append(pred_df)
    
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    
    return output_path


if __name__ == "__main__":
    # CLI entrypoint: train and evaluate all models
    import sys
    
    # Load feature panel
    processed_root = get_path("processed_root")
    feature_path = processed_root / "bdc_feature_panel.parquet"
    
    if not feature_path.exists():
        print(f"Error: Feature panel not found at {feature_path}")
        print("Run: python -m src.features.feature_engineering first")
        sys.exit(1)
    
    print("Loading feature panel...")
    df = pd.read_parquet(feature_path)
    print(f"Loaded {len(df)} rows")
    
    # Train and evaluate
    print("\n" + "=" * 60)
    print("Training and Evaluating Models")
    print("=" * 60)
    
    predictions_dict = train_and_evaluate_models(
        df=df,
        target_col="target_next_qtr_pct",
        include_baselines=True,
        include_ml=True,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    comparison = evaluate_all_models(predictions_dict)
    print(comparison.sort_values("mae"))
    
    # Save results
    cfg = load_config()
    models_dir_rel = cfg.get("models", {}).get("models_dir", "data/processed/models")
    models_dir = get_path("data_root").parent / models_dir_rel
    predictions_path = models_dir / "backtest_predictions.parquet"
    comparison_path = models_dir / "model_comparison.csv"
    
    ensure_parent_dir(predictions_path)
    save_predictions(predictions_dict, predictions_path)
    comparison.to_csv(comparison_path, index=True)
    
    print(f"\n[OK] Saved predictions to: {predictions_path}")
    print(f"[OK] Saved comparison to: {comparison_path}")

