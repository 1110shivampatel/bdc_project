"""
End-to-end tests for panel construction and feature engineering.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import get_path
from src.data.panel import build_bdc_reference_table, build_quarterly_panel
from src.features.feature_engineering import (
    build_feature_panel,
    compute_fundamental_features,
    create_target_labels,
)


def test_entity_reference_exists():
    """Test that entity reference table can be built and has expected structure."""
    entity_ref = build_bdc_reference_table()
    assert not entity_ref.empty
    assert "cik" in entity_ref.columns
    assert "name_canonical" in entity_ref.columns
    assert len(entity_ref) > 0


def test_quarterly_panel_exists():
    """Test that quarterly panel can be built and has expected structure."""
    panel = build_quarterly_panel()
    assert not panel.empty
    assert "period" in panel.columns
    assert "cik" in panel.columns
    assert len(panel) > 0


def test_feature_panel_has_features():
    """Test that feature panel includes computed features."""
    processed_root = get_path("processed_root")
    panel_path = processed_root / "bdc_quarterly_panel.parquet"

    if not panel_path.exists():
        pytest.skip("Quarterly panel not found - run panel builder first")

    panel_df = pd.read_parquet(panel_path)
    feature_panel = compute_fundamental_features(panel_df)

    # Check that some feature columns were added
    feature_cols = [c for c in feature_panel.columns if any(
        suffix in c for suffix in ["_to_", "_qoq_", "_ma4", "_std4"]
    )]
    assert len(feature_cols) > 0, "No feature columns found in feature panel"


def test_target_labels_created():
    """Test that target labels can be created from feature panel."""
    processed_root = get_path("processed_root")
    panel_path = processed_root / "bdc_quarterly_panel.parquet"

    if not panel_path.exists():
        pytest.skip("Quarterly panel not found - run panel builder first")

    panel_df = pd.read_parquet(panel_path)
    feature_panel = compute_fundamental_features(panel_df)

    if "net_assets" not in feature_panel.columns:
        pytest.skip("net_assets column not found - cannot create target labels")

    target_panel = create_target_labels(feature_panel, target_metric="net_assets")
    assert "target_next_qtr_pct" in target_panel.columns
    assert "target_next_qtr_abs" in target_panel.columns

