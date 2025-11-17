"""Quick verification script for Phase 2 outputs."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import get_path

processed_root = get_path("processed_root")

print("=" * 60)
print("Phase 2 Output Verification")
print("=" * 60)

# Check entity reference
entity_path = processed_root / "bdc_entity_reference.parquet"
if entity_path.exists():
    df_entity = pd.read_parquet(entity_path)
    print(f"\n[OK] Entity Reference: {len(df_entity)} entities")
    print(f"     Columns: {list(df_entity.columns)}")
else:
    print("\n[ERROR] Entity reference not found")

# Check quarterly panel
panel_path = processed_root / "bdc_quarterly_panel.parquet"
if panel_path.exists():
    df_panel = pd.read_parquet(panel_path)
    print(f"\n[OK] Quarterly Panel: {len(df_panel)} rows")
    print(f"     Periods: {df_panel['period'].nunique()}")
    print(f"     BDCs: {df_panel['cik'].nunique()}")
    print(f"     Columns: {len(df_panel.columns)}")
else:
    print("\n[ERROR] Quarterly panel not found")

# Check feature panel
feature_path = processed_root / "bdc_feature_panel.parquet"
if feature_path.exists():
    df_features = pd.read_parquet(feature_path)
    print(f"\n[OK] Feature Panel: {len(df_features)} rows")
    print(f"     Total columns: {len(df_features.columns)}")
    
    # Count feature types
    ratio_features = [c for c in df_features.columns if "_to_" in c]
    growth_features = [c for c in df_features.columns if "_qoq_" in c]
    rolling_features = [c for c in df_features.columns if "_ma4" in c or "_std4" in c]
    target_features = [c for c in df_features.columns if "target_" in c]
    soi_features = [c for c in df_features.columns if any(x in c for x in ["portfolio_", "concentration", "sector", "rate"])]
    
    print(f"     Ratio features: {len(ratio_features)}")
    print(f"     Growth features: {len(growth_features)}")
    print(f"     Rolling stats: {len(rolling_features)}")
    print(f"     Target labels: {len(target_features)}")
    print(f"     SOI features: {len(soi_features)}")
    
    # Show sample
    print(f"\n     Sample columns: {list(df_features.columns[:10])}")
else:
    print("\n[ERROR] Feature panel not found")

print("\n" + "=" * 60)

