"""
Feature engineering for BDC analytics and forecasting.

This module computes derived metrics from:
1. Quarterly panel (size, leverage, income proxies)
2. SOI holdings data (portfolio mix, sector exposure, rate types)
3. Optional macro features (Fed rates, yield curve)

Features are designed for forecasting next-quarter NAV change or health metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.utils.io import ensure_parent_dir


def compute_fundamental_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fundamental features from the quarterly panel.

    Features include:
    - Size metrics: total_assets, net_assets
    - Leverage: debt_to_assets, debt_to_equity
    - Income ratios: nii_to_assets, income_to_expenses
    - Growth: qoq change in assets, income

    Parameters
    ----------
    panel_df : pd.DataFrame
        Quarterly panel with columns: period, cik, name, and metric columns.

    Returns
    -------
    pd.DataFrame
        Panel with additional feature columns.
    """
    df = panel_df.copy()

    # Ensure period is sortable (convert to datetime-like if needed)
    # For now, assume period is string like "2025_Q1" or "2025_06"
    df = df.sort_values(["cik", "period"])

    # Helper function for safe division using np.where to avoid recursion issues
    def safe_divide(numerator, denominator):
        """Divide with zero handling, returning pd.NA for invalid divisions."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                (denominator != 0) & pd.notna(denominator) & pd.notna(numerator),
                numerator / denominator,
                pd.NA
            )
        return pd.Series(result, index=numerator.index)

    # 1. Leverage features (handle division by zero)
    if "total_assets" in df.columns and "debt" in df.columns:
        df["debt_to_assets"] = safe_divide(df["debt"], df["total_assets"])
    if "total_assets" in df.columns and "total_liabilities" in df.columns:
        df["liabilities_to_assets"] = safe_divide(df["total_liabilities"], df["total_assets"])
    if "net_assets" in df.columns and "debt" in df.columns:
        df["debt_to_equity"] = safe_divide(df["debt"], df["net_assets"])

    # 2. Income efficiency ratios
    if "net_investment_income" in df.columns and "total_assets" in df.columns:
        df["nii_to_assets"] = safe_divide(df["net_investment_income"], df["total_assets"])
    if "investment_income" in df.columns and "total_assets" in df.columns:
        df["income_to_assets"] = safe_divide(df["investment_income"], df["total_assets"])
    if "net_investment_income" in df.columns and "total_expenses" in df.columns:
        df["nii_to_expenses"] = safe_divide(df["net_investment_income"], df["total_expenses"])

    # 3. Quarter-over-quarter growth (lagged for forecasting)
    for metric in ["total_assets", "net_assets", "net_investment_income", "investment_income"]:
        if metric in df.columns:
            df[f"{metric}_qoq_pct"] = df.groupby("cik")[metric].pct_change() * 100
            df[f"{metric}_qoq_abs"] = df.groupby("cik")[metric].diff()

    # 4. Rolling statistics (last 4 quarters)
    for metric in ["total_assets", "net_investment_income"]:
        if metric in df.columns:
            df[f"{metric}_ma4"] = df.groupby("cik")[metric].rolling(window=4, min_periods=1).mean().reset_index(0, drop=True)
            df[f"{metric}_std4"] = df.groupby("cik")[metric].rolling(window=4, min_periods=1).std().reset_index(0, drop=True)

    return df


def load_soi_for_period(period: str) -> pd.DataFrame:
    """
    Load SOI (Schedule of Investments) table for a given period.

    Parameters
    ----------
    period : str
        Period key, e.g. '2025_Q1'.

    Returns
    -------
    pd.DataFrame
        SOI dataframe with holdings-level data.
    """
    raw_root = get_path("raw_root")
    soi_path = raw_root / period / "soi.tsv"

    if not soi_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(soi_path, sep="\t", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["period"] = period
    return df


def compute_portfolio_features(soi_df: pd.DataFrame, cik: Optional[int] = None) -> pd.Series:
    """
    Compute portfolio-level features from SOI data.

    Features include:
    - Sector concentration (top sector %, HHI)
    - Floating vs fixed rate exposure
    - Average maturity / WAM
    - Fair value vs cost gap
    - Top-10 concentration

    Parameters
    ----------
    soi_df : pd.DataFrame
        SOI holdings dataframe. Must have columns for issuer, fair value, cost, etc.
    cik : int | None
        Optional CIK to filter to a specific BDC.

    Returns
    -------
    pd.Series
        Series with portfolio feature values.
    """
    if soi_df.empty:
        return pd.Series(dtype=float)

    df = soi_df.copy()

    if cik is not None and "cik" in df.columns:
        df = df[df["cik"] == cik]

    features = pd.Series(dtype=float)

    # Find fair value and cost columns (case-insensitive search)
    fv_cols = [c for c in df.columns if "fair" in c.lower() and "value" in c.lower()]
    cost_cols = [c for c in df.columns if "cost" in c.lower() and "basis" in c.lower()]

    if fv_cols and cost_cols:
        fv_col = fv_cols[0]
        cost_col = cost_cols[0]

        # Convert to numeric
        df[fv_col] = pd.to_numeric(df[fv_col], errors="coerce")
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

        total_fv = df[fv_col].sum()
        total_cost = df[cost_col].sum()

        if total_fv > 0:
            features["portfolio_fv"] = total_fv
            features["portfolio_cost"] = total_cost
            features["fv_cost_gap_pct"] = ((total_fv - total_cost) / total_cost * 100) if total_cost > 0 else None

            # Top-10 concentration
            df_sorted = df.nlargest(10, fv_col)
            top10_pct = (df_sorted[fv_col].sum() / total_fv * 100) if total_fv > 0 else None
            features["top10_concentration_pct"] = top10_pct

            # Herfindahl-Hirschman Index (concentration measure)
            df["fv_pct"] = (df[fv_col] / total_fv * 100) if total_fv > 0 else 0
            hhi = (df["fv_pct"] ** 2).sum()
            features["portfolio_hhi"] = hhi

    # Sector exposure (if sector/industry column exists)
    sector_cols = [c for c in df.columns if "sector" in c.lower() or "industry" in c.lower()]
    if sector_cols and fv_cols:
        sector_col = sector_cols[0]
        fv_col = fv_cols[0]
        df[fv_col] = pd.to_numeric(df[fv_col], errors="coerce")
        total_fv = df[fv_col].sum()

        if total_fv > 0 and df[sector_col].notna().any():
            sector_exposure = df.groupby(sector_col)[fv_col].sum() / total_fv * 100
            features["top_sector_pct"] = sector_exposure.max()
            features["sector_count"] = sector_exposure.count()

    # Floating vs fixed rate (if rate type column exists)
    rate_type_cols = [c for c in df.columns if "rate" in c.lower() and ("type" in c.lower() or "floating" in c.lower() or "fixed" in c.lower())]
    if rate_type_cols and fv_cols:
        rate_col = rate_type_cols[0]
        fv_col = fv_cols[0]
        df[fv_col] = pd.to_numeric(df[fv_col], errors="coerce")
        total_fv = df[fv_col].sum()

        if total_fv > 0:
            # Try to detect floating vs fixed from column values
            floating_mask = df[rate_col].astype(str).str.lower().str.contains("float", na=False)
            fixed_mask = df[rate_col].astype(str).str.lower().str.contains("fix", na=False)

            floating_fv = df[floating_mask][fv_col].sum()
            fixed_fv = df[fixed_mask][fv_col].sum()

            features["floating_rate_pct"] = (floating_fv / total_fv * 100) if total_fv > 0 else None
            features["fixed_rate_pct"] = (fixed_fv / total_fv * 100) if total_fv > 0 else None

    return features


def build_feature_panel(
    panel_df: pd.DataFrame,
    include_soi: bool = True,
    periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a comprehensive feature panel by combining:
    1. Fundamental features from quarterly panel
    2. Portfolio features from SOI (if available)

    Parameters
    ----------
    panel_df : pd.DataFrame
        Quarterly panel dataframe.
    include_soi : bool
        If True, attempt to load and merge SOI-based portfolio features.
    periods : List[str] | None
        Optional list of periods to process. If None, uses all periods in panel_df.

    Returns
    -------
    pd.DataFrame
        Feature panel with all computed features.
    """
    # Start with fundamental features
    feature_panel = compute_fundamental_features(panel_df)

    if not include_soi:
        return feature_panel

    # Add SOI-based portfolio features
    if periods is None:
        periods = feature_panel["period"].unique().tolist()

    soi_features_list = []

    for period in periods:
        soi_df = load_soi_for_period(period)
        if soi_df.empty:
            continue

        # Get unique CIKs for this period
        if "cik" in soi_df.columns:
            ciks = soi_df["cik"].unique()
        elif "cik" in feature_panel.columns:
            ciks = feature_panel[feature_panel["period"] == period]["cik"].unique()
        else:
            continue

        for cik in ciks:
            features = compute_portfolio_features(soi_df, cik=cik)
            if not features.empty:
                features["period"] = period
                features["cik"] = cik
                soi_features_list.append(features)

    if soi_features_list:
        soi_features_df = pd.DataFrame(soi_features_list)
        # Merge with feature panel
        merge_cols = ["period", "cik"]
        feature_panel = feature_panel.merge(soi_features_df, on=merge_cols, how="left", suffixes=("", "_soi"))

    return feature_panel


def create_target_labels(feature_panel: pd.DataFrame, target_metric: str = "net_assets") -> pd.DataFrame:
    """
    Create target labels for forecasting: next-quarter change in target metric.

    Parameters
    ----------
    feature_panel : pd.DataFrame
        Feature panel dataframe.
    target_metric : str
        Base metric to compute change for (e.g., 'net_assets', 'total_assets').

    Returns
    -------
    pd.DataFrame
        Feature panel with added target columns:
        - target_next_qtr_pct: Next quarter % change
        - target_next_qtr_abs: Next quarter absolute change
    """
    df = feature_panel.copy()
    df = df.sort_values(["cik", "period"])

    if target_metric not in df.columns:
        raise ValueError(f"Target metric '{target_metric}' not found in feature panel")

    # Compute forward-looking change (shift backward to align with current period)
    # Get next quarter value
    next_qtr = df.groupby("cik")[target_metric].shift(-1)
    current_qtr = df[target_metric]
    
    # Compute percentage change, handling edge cases
    with np.errstate(divide='ignore', invalid='ignore'):
        # Avoid division by zero
        denominator = current_qtr.replace(0, np.nan)
        pct_change = (next_qtr - current_qtr) / denominator * 100
        abs_change = next_qtr - current_qtr
    
    # Replace inf with NaN
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
    
    df[f"target_next_qtr_pct"] = pct_change
    df[f"target_next_qtr_abs"] = abs_change

    return df


def save_feature_panel(feature_df: pd.DataFrame, overwrite: bool = True) -> Path:
    """
    Save the feature panel to Parquet.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature panel dataframe.
    overwrite : bool
        If True, overwrite existing file.

    Returns
    -------
    Path
        Path to saved feature file.
    """
    cfg = load_config()
    processed_root = get_path("processed_root")
    feature_path = processed_root / "bdc_feature_panel.parquet"

    if feature_path.exists() and not overwrite:
        raise FileExistsError(f"Feature panel exists at {feature_path} and overwrite=False")

    ensure_parent_dir(feature_path)
    feature_df.to_parquet(feature_path, index=False)
    return feature_path


if __name__ == "__main__":
    # CLI entrypoint: build feature panel from quarterly panel
    import sys
    from pathlib import Path

    # Load quarterly panel
    processed_root = get_path("processed_root")
    panel_path = processed_root / "bdc_quarterly_panel.parquet"

    if not panel_path.exists():
        print(f"Error: Quarterly panel not found at {panel_path}")
        print("Run: python -m src.data.panel first")
        sys.exit(1)

    panel_df = pd.read_parquet(panel_path)
    print(f"Loaded quarterly panel: {len(panel_df)} rows")

    print("Building feature panel...")
    feature_panel = build_feature_panel(panel_df, include_soi=True)

    print("Creating target labels...")
    feature_panel = create_target_labels(feature_panel, target_metric="net_assets")

    feature_path = save_feature_panel(feature_panel, overwrite=True)
    print(f"[OK] Saved feature panel ({len(feature_panel)} rows) to: {feature_path}")

