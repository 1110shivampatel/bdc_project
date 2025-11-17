"""
Utility functions for the Streamlit dashboard.

Handles data loading, caching, and common transformations.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.config import get_path


@st.cache_data
def load_feature_panel() -> pd.DataFrame:
    """Load and cache the feature panel."""
    processed_root = get_path("processed_root")
    path = processed_root / "bdc_feature_panel.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_quarterly_panel() -> pd.DataFrame:
    """Load and cache the quarterly panel."""
    processed_root = get_path("processed_root")
    path = processed_root / "bdc_quarterly_panel.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_entity_reference() -> pd.DataFrame:
    """Load and cache the entity reference table."""
    processed_root = get_path("processed_root")
    path = processed_root / "bdc_entity_reference.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """Load and cache forecasts."""
    processed_root = get_path("processed_root")
    path = processed_root / "forecasts.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load and cache model comparison metrics."""
    processed_root = get_path("processed_root")
    path = processed_root / "models" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_backtest_predictions() -> pd.DataFrame:
    """Load and cache backtest predictions."""
    processed_root = get_path("processed_root")
    path = processed_root / "models" / "backtest_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def get_bdc_list(feature_panel: pd.DataFrame) -> list:
    """Get sorted list of unique BDC names."""
    if "name" in feature_panel.columns:
        return sorted(feature_panel["name"].dropna().unique().tolist())
    return []


def get_latest_period(feature_panel: pd.DataFrame) -> str:
    """Get the latest period in the data."""
    if "period" in feature_panel.columns:
        return sorted(feature_panel["period"].unique().tolist())[-1]
    return ""


def format_currency(value: float, decimals: int = 1) -> str:
    """Format a number as currency (millions)."""
    if pd.isna(value) or value == 0:
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    return f"${value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def get_bdc_data(feature_panel: pd.DataFrame, bdc_name: str) -> pd.DataFrame:
    """Get all historical data for a specific BDC."""
    return feature_panel[feature_panel["name"] == bdc_name].sort_values("period").copy()


def get_bdc_forecast(forecasts: pd.DataFrame, bdc_name: str) -> Optional[float]:
    """Get forecast for a specific BDC."""
    if forecasts.empty or "name" not in forecasts.columns:
        return None
    bdc_forecast = forecasts[forecasts["name"] == bdc_name]
    if len(bdc_forecast) > 0:
        return bdc_forecast.iloc[0]["forecast"]
    return None

