"""
Prediction/inference module for generating forecasts.

Loads trained models and generates predictions for new data.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import get_path, load_config
from src.models.train import prepare_training_data
from src.utils.io import ensure_parent_dir


def load_model(model_path: Path) -> tuple[object, Dict]:
    """
    Load trained model and metadata.
    
    Parameters
    ----------
    model_path : Path
        Path to model pickle file.
    
    Returns
    -------
    tuple[object, Dict]
        (model, metadata)
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return model, metadata


def predict_with_model(
    model: object,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """
    Generate predictions using a trained model.
    
    Parameters
    ----------
    model : object
        Trained model with predict() method.
    df : pd.DataFrame
        Feature dataframe.
    feature_cols : List[str]
        List of feature column names.
    
    Returns
    -------
    pd.Series
        Predictions.
    """
    X = df[feature_cols].copy()
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index)


def generate_forecasts(
    df: pd.DataFrame,
    model_name: str,
    target_col: str = "target_next_qtr_pct",
) -> pd.DataFrame:
    """
    Generate forecasts for the latest period using a trained model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature panel with all periods.
    model_name : str
        Name of model to use (e.g., "XGBoost_n100_d6_lr0.1").
    target_col : str
        Target column name (for reference).
    
    Returns
    -------
    pd.DataFrame
        Forecasts with columns: period, cik, name, forecast, and original features.
    """
    # For models that don't need saved files (Naive, RollingMean), recreate them
    from src.models.baselines import NaiveForecaster, RollingMeanForecaster
    from src.models.ml_models import get_ml_models
    
    # Check if it's a baseline that doesn't need saving
    if model_name == "Naive":
        model = NaiveForecaster()
        # Get feature cols from a sample - use all numeric columns
        feature_cols = [col for col in df.columns 
                       if col not in ["period", "cik", "name", "target_next_qtr_pct", "target_next_qtr_abs"]
                       and pd.api.types.is_numeric_dtype(df[col])]
    elif model_name.startswith("RollingMean"):
        # For rolling mean, we'd need to train it, but for now skip it
        raise ValueError(f"RollingMean models require training data. Please use a saved ML model instead.")
    else:
        # Load saved model
        processed_root = get_path("processed_root")
        models_dir = processed_root / "models"
        model_path = models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Note: Only ML models (XGBoost, LightGBM, RandomForest) are saved.\n"
                f"Baseline models (Naive, RollingMean) are not saved."
            )
        
        model, metadata = load_model(model_path)
        feature_cols = metadata["feature_cols"]
    
    # Get latest period
    periods = sorted(df["period"].unique().tolist())
    latest_period = periods[-1]
    
    # Filter to latest period
    latest_df = df[df["period"] == latest_period].copy()
    
    # Prepare features
    feature_df, _, _ = prepare_training_data(latest_df, target_col)
    
    # Generate predictions
    forecasts = predict_with_model(model, feature_df, feature_cols)
    
    # Combine with metadata
    result_df = latest_df[["period", "cik", "name"]].copy()
    result_df["forecast"] = forecasts.values
    result_df["model_name"] = model_name
    
    return result_df


def generate_all_forecasts(
    df: pd.DataFrame,
    target_col: str = "target_next_qtr_pct",
) -> pd.DataFrame:
    """
    Generate forecasts using the best model (lowest MAE from backtesting).
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature panel.
    target_col : str
        Target column name.
    
    Returns
    -------
    pd.DataFrame
        Forecasts for all BDCs in latest period.
    """
    # Load model comparison to find best model
    processed_root = get_path("processed_root")
    models_dir = processed_root / "models"
    comparison_path = models_dir / "model_comparison.csv"
    
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Model comparison not found: {comparison_path}\n"
            "Please train models first by running: python -m src.models.train"
        )
    
    comparison = pd.read_csv(comparison_path, index_col=0)
    
    # Filter out models with NaN or extremely high MAE (likely failed models)
    # Use a more lenient threshold - if all models have high MAE, pick the best one
    # This handles cases where target values are large (e.g., absolute changes in millions)
    valid_models = comparison[
        comparison["mae"].notna() & 
        (comparison["n_samples"] > 0)  # Must have predictions
    ]
    
    if len(valid_models) == 0:
        raise ValueError(
            "No valid models found in comparison. "
            "All models may have failed due to data quality issues.\n"
            "Please check:\n"
            "1. Target column has valid values (not all NaN/inf)\n"
            "2. Features don't contain infinity values\n"
            "3. Retrain models after fixing data quality issues."
        )
    
    # If all models have very high MAE, still pick the best one
    # But warn the user
    if valid_models["mae"].min() > 1e8:
        print(f"WARNING: All models have very high MAE (min: {valid_models['mae'].min():.2e})")
        print("This may indicate data quality issues or that the target scale is very large.")
        print("Proceeding with best available model...")
    
    best_model = valid_models.sort_values("mae").index[0]
    
    print(f"Using best model: {best_model} (MAE: {valid_models.loc[best_model, 'mae']:.4f})")
    
    return generate_forecasts(df, best_model, target_col)


def save_forecasts(forecasts_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save forecasts to Parquet.
    
    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecasts dataframe.
    output_path : Path | None
        Optional output path. If None, uses default location.
    
    Returns
    -------
    Path
        Path to saved file.
    """
    if output_path is None:
        processed_root = get_path("processed_root")
        output_path = processed_root / "forecasts.parquet"
    
    ensure_parent_dir(output_path)
    forecasts_df.to_parquet(output_path, index=False)
    
    return output_path


if __name__ == "__main__":
    # CLI entrypoint: generate forecasts
    import sys
    
    # Load feature panel
    processed_root = get_path("processed_root")
    feature_path = processed_root / "bdc_feature_panel.parquet"
    
    if not feature_path.exists():
        print(f"Error: Feature panel not found at {feature_path}")
        sys.exit(1)
    
    print("Loading feature panel...")
    df = pd.read_parquet(feature_path)
    
    print("Generating forecasts...")
    forecasts = generate_all_forecasts(df)
    
    output_path = save_forecasts(forecasts)
    print(f"\n[OK] Saved forecasts ({len(forecasts)} BDCs) to: {output_path}")
    print(f"\nForecast summary:")
    print(forecasts[["name", "forecast"]].head(10))

