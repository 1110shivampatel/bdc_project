"""
Entity resolution and quarterly panel construction for BDC analytics.

This module:
1. Builds a BDC reference table (entity master) from submission metadata
2. Aggregates numeric facts into a quarterly panel (one row per BDC per period)
3. Handles entity name normalization and CIK-based deduplication
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

from src.config import get_path, load_config
from src.utils.io import ensure_parent_dir


def build_bdc_reference_table() -> pd.DataFrame:
    """
    Build a BDC reference table from submission metadata across all periods.

    This creates a master entity table with:
    - CIK (stable identifier)
    - Canonical name (most common name across periods)
    - First seen period, last seen period
    - Count of filings

    Returns
    -------
    pd.DataFrame
        Reference table with columns: cik, name_canonical, first_period, last_period, filing_count
    """
    cfg = load_config()
    duckdb_path = get_path("duckdb_path")
    table_name = cfg.get("build_store", {}).get("numeric_table_name", "numeric_facts")

    conn = duckdb.connect(str(duckdb_path))

    try:
        # Extract unique (cik, name, period) combinations from numeric_facts
        query = f"""
        SELECT DISTINCT
            cik,
            name,
            period
        FROM {table_name}
        WHERE cik IS NOT NULL AND name IS NOT NULL
        ORDER BY cik, period, name
        """
        df = conn.execute(query).df()

        if df.empty:
            raise ValueError("No BDC entities found in numeric_facts table")

        # For each CIK, pick the most common name as canonical
        # Also track first/last period and filing count
        entity_ref = (
            df.groupby("cik")
            .agg(
                {
                    "name": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],  # Most common name
                    "period": ["min", "max", "count"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        entity_ref.columns = ["cik", "name_canonical", "first_period", "last_period", "filing_count"]

        return entity_ref

    finally:
        conn.close()


def build_quarterly_panel(
    tag_mappings: Optional[Dict[str, List[str]]] = None,
    aggregation_method: str = "latest",
) -> pd.DataFrame:
    """
    Build a quarterly panel: one row per BDC per period.

    Parameters
    ----------
    tag_mappings : Dict[str, List[str]] | None
        Optional mapping from metric name to list of tag patterns to search.
        If None, uses default mappings for common BDC metrics.
    aggregation_method : str
        How to aggregate multiple values for the same (BDC, period, tag):
        - "latest": Use the most recent ddate value
        - "mean": Average across all values
        - "sum": Sum all values (for flow metrics)

    Returns
    -------
    pd.DataFrame
        Panel with columns: period, cik, name, and one column per metric.

    Notes
    -----
    This function pivots the numeric_facts table to create a wide-format panel.
    It uses tag pattern matching (substring search) to find relevant tags for each metric.
    """
    if tag_mappings is None:
        tag_mappings = {
            "total_assets": ["totalassets", "assets", "netassets"],
            "net_investment_income": ["netinvestmentincome"],
            "investment_income": ["investmentincome", "grossinvestmentincome"],
            "total_expenses": ["totalexpenses", "operatingexpenses"],
            "net_assets": ["netassets", "totalnetassets"],
            "total_liabilities": ["totalliabilities"],
            "debt": ["debt", "borrowings", "notespayable"],
        }

    cfg = load_config()
    duckdb_path = get_path("duckdb_path")
    table_name = cfg.get("build_store", {}).get("numeric_table_name", "numeric_facts")

    conn = duckdb.connect(str(duckdb_path))

    try:
        # Load all numeric facts
        query = f"""
        SELECT
            period,
            cik,
            name,
            tag,
            val_num,
            ddate,
            uom
        FROM {table_name}
        WHERE val_num IS NOT NULL
        ORDER BY period, cik, tag, ddate DESC
        """
        df = conn.execute(query).df()

        if df.empty:
            raise ValueError("No numeric facts found in store")

        # Normalize tag names to lowercase for matching
        df["tag_lower"] = df["tag"].str.lower()

        # Build panel: one row per (period, cik)
        panel_rows = []

        for (period, cik), group in df.groupby(["period", "cik"]):
            row = {"period": period, "cik": cik, "name": group["name"].iloc[0]}

            # For each metric, find matching tags and aggregate
            for metric_name, tag_patterns in tag_mappings.items():
                # Find rows where tag matches any pattern
                mask = group["tag_lower"].apply(
                    lambda t: any(pattern in t for pattern in tag_patterns)
                )
                matches = group[mask].copy()

                if len(matches) == 0:
                    row[metric_name] = None
                elif aggregation_method == "latest":
                    # Use the most recent value (already sorted by ddate DESC)
                    row[metric_name] = matches.iloc[0]["val_num"]
                elif aggregation_method == "mean":
                    row[metric_name] = matches["val_num"].mean()
                elif aggregation_method == "sum":
                    row[metric_name] = matches["val_num"].sum()
                else:
                    raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

            panel_rows.append(row)

        panel = pd.DataFrame(panel_rows)

        # Sort by period, then cik
        panel = panel.sort_values(["period", "cik"]).reset_index(drop=True)

        return panel

    finally:
        conn.close()


def save_panel(panel_df: pd.DataFrame, overwrite: bool = True) -> Path:
    """
    Save the quarterly panel to Parquet.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel dataframe to save.
    overwrite : bool
        If True, overwrite existing file. If False, raise error if file exists.

    Returns
    -------
    Path
        Path to the saved panel file.
    """
    cfg = load_config()
    processed_root = get_path("processed_root")
    panel_path = processed_root / "bdc_quarterly_panel.parquet"

    if panel_path.exists() and not overwrite:
        raise FileExistsError(f"Panel file exists at {panel_path} and overwrite=False")

    ensure_parent_dir(panel_path)
    panel_df.to_parquet(panel_path, index=False)
    return panel_path


def save_entity_reference(entity_df: pd.DataFrame, overwrite: bool = True) -> Path:
    """
    Save the BDC entity reference table to Parquet.

    Parameters
    ----------
    entity_df : pd.DataFrame
        Entity reference dataframe to save.
    overwrite : bool
        If True, overwrite existing file. If False, raise error if file exists.

    Returns
    -------
    Path
        Path to the saved reference file.
    """
    cfg = load_config()
    processed_root = get_path("processed_root")
    ref_path = processed_root / "bdc_entity_reference.parquet"

    if ref_path.exists() and not overwrite:
        raise FileExistsError(f"Entity reference file exists at {ref_path} and overwrite=False")

    ensure_parent_dir(ref_path)
    entity_df.to_parquet(ref_path, index=False)
    return ref_path


if __name__ == "__main__":
    # CLI entrypoint: build and save both entity reference and panel
    print("Building BDC entity reference table...")
    entity_ref = build_bdc_reference_table()
    ref_path = save_entity_reference(entity_ref, overwrite=True)
    print(f"[OK] Saved entity reference ({len(entity_ref)} entities) to: {ref_path}")

    print("\nBuilding quarterly panel...")
    panel = build_quarterly_panel()
    panel_path = save_panel(panel, overwrite=True)
    print(f"[OK] Saved quarterly panel ({len(panel)} rows) to: {panel_path}")

