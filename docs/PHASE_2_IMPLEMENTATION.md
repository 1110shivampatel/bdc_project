# Phase 2 Implementation: Entity Resolution, Panel Construction & Feature Engineering

## Overview

This document describes the implementation of **Phase 2** of the BDC Analytics & Forecasting Platform, which includes:

1. **Entity Resolution** - Building a BDC reference table from submission metadata
2. **Panel Construction** - Aggregating numeric facts into quarterly BDC panels
3. **Feature Engineering** - Computing derived metrics for forecasting

---

## 1. Entity Resolution (`src/data/panel.py`)

### Purpose

Creates a **master entity reference table** that maps CIK (Central Index Key) to canonical BDC names and tracks filing history.

### Implementation

**Function: `build_bdc_reference_table()`**

- Queries the DuckDB `numeric_facts` table to extract unique (CIK, name, period) combinations
- For each CIK, selects the **most common name** as the canonical name
- Tracks:
  - `first_period`: Earliest period this BDC appears
  - `last_period`: Most recent period
  - `filing_count`: Number of periods with filings

**Output:**
- Saved to `data/processed/bdc_entity_reference.parquet`
- Columns: `cik`, `name_canonical`, `first_period`, `last_period`, `filing_count`

### Results

- **185 unique BDC entities** identified across all periods
- Each entity has a stable CIK identifier and canonical name

---

## 2. Quarterly Panel Construction (`src/data/panel.py`)

### Purpose

Transforms the long-format `numeric_facts` table into a **wide-format quarterly panel**: one row per BDC per period.

### Implementation

**Function: `build_quarterly_panel(tag_mappings, aggregation_method)`**

**Tag Mappings:**
Default mappings for common BDC metrics:
- `total_assets`: ["totalassets", "assets", "netassets"]
- `net_investment_income`: ["netinvestmentincome"]
- `investment_income`: ["investmentincome", "grossinvestmentincome"]
- `total_expenses`: ["totalexpenses", "operatingexpenses"]
- `net_assets`: ["netassets", "totalnetassets"]
- `total_liabilities`: ["totalliabilities"]
- `debt`: ["debt", "borrowings", "notespayable"]

**Aggregation Method:**
- `"latest"` (default): Uses the most recent `ddate` value when multiple values exist for the same (BDC, period, tag)
- `"mean"`: Averages all values
- `"sum"`: Sums all values (for flow metrics)

**Process:**
1. Loads all numeric facts from DuckDB
2. For each (period, cik) combination:
   - Searches for tags matching each metric pattern (case-insensitive substring match)
   - Aggregates values according to the chosen method
   - Creates one row with all metrics as columns

**Output:**
- Saved to `data/processed/bdc_quarterly_panel.parquet`
- Columns: `period`, `cik`, `name`, plus one column per metric

### Results

- **1,655 panel rows** (BDC-period combinations)
- Covers all 15 periods in the dataset
- Each row contains key financial metrics for that BDC in that period

---

## 3. Feature Engineering (`src/features/feature_engineering.py`)

### Purpose

Computes **derived features** from the quarterly panel for forecasting next-quarter NAV change or health metrics.

### Implementation

#### 3.1 Fundamental Features (`compute_fundamental_features()`)

**Leverage Ratios:**
- `debt_to_assets`: Debt / Total Assets
- `liabilities_to_assets`: Total Liabilities / Total Assets
- `debt_to_equity`: Debt / Net Assets

**Income Efficiency Ratios:**
- `nii_to_assets`: Net Investment Income / Total Assets
- `income_to_assets`: Investment Income / Total Assets
- `nii_to_expenses`: Net Investment Income / Total Expenses

**Quarter-over-Quarter Growth:**
- `{metric}_qoq_pct`: Percentage change from previous quarter
- `{metric}_qoq_abs`: Absolute change from previous quarter
- Computed for: `total_assets`, `net_assets`, `net_investment_income`, `investment_income`

**Rolling Statistics (4-quarter window):**
- `{metric}_ma4`: 4-quarter moving average
- `{metric}_std4`: 4-quarter rolling standard deviation
- Computed for: `total_assets`, `net_investment_income`

**Technical Note:**
- Uses `np.where` for safe division (handles division by zero and NaN gracefully)
- All ratios return `pd.NA` for invalid divisions

#### 3.2 Portfolio Features from SOI (`compute_portfolio_features()`)

**Fair Value Metrics:**
- `portfolio_fv`: Total fair value of portfolio
- `portfolio_cost`: Total cost basis
- `fv_cost_gap_pct`: (Fair Value - Cost) / Cost * 100

**Concentration Metrics:**
- `top10_concentration_pct`: Percentage of portfolio in top 10 positions
- `portfolio_hhi`: Herfindahl-Hirschman Index (concentration measure)

**Sector Exposure:**
- `top_sector_pct`: Percentage in largest sector
- `sector_count`: Number of distinct sectors

**Rate Type Exposure:**
- `floating_rate_pct`: Percentage of portfolio with floating rates
- `fixed_rate_pct`: Percentage with fixed rates

**Process:**
- Loads SOI (Schedule of Investments) TSV for each period
- Aggregates holdings-level data by BDC (CIK)
- Detects relevant columns via case-insensitive substring matching

#### 3.3 Target Labels (`create_target_labels()`)

**Target Variables:**
- `target_next_qtr_pct`: Next quarter percentage change in target metric
- `target_next_qtr_abs`: Next quarter absolute change

**Default Target:** `net_assets` (can be configured)

**Process:**
- Shifts target metric forward one quarter
- Computes change from current period
- Aligns with current-period features for supervised learning

### Output

**Function: `build_feature_panel()`**
- Combines fundamental features + SOI portfolio features
- Merges on `(period, cik)`
- Saved to `data/processed/bdc_feature_panel.parquet`

### Results

- **1,655 feature panel rows** (matches quarterly panel)
- **20+ feature columns** including:
  - 7 base metrics (from panel)
  - 6 leverage/income ratios
  - 8+ growth/rolling statistics
  - 8+ SOI portfolio features (when available)

---

## Usage

### Command-Line Interface

**Build entity reference and panel:**
```bash
python -m src.data.panel
```

**Build feature panel:**
```bash
python -m src.features.feature_engineering
```

### Programmatic Usage

```python
from src.data.panel import build_bdc_reference_table, build_quarterly_panel
from src.features.feature_engineering import build_feature_panel, create_target_labels

# Build entity reference
entity_ref = build_bdc_reference_table()

# Build quarterly panel
panel = build_quarterly_panel()

# Build feature panel
feature_panel = build_feature_panel(panel, include_soi=True)

# Add target labels
feature_panel = create_target_labels(feature_panel, target_metric="net_assets")
```

---

## File Structure

```
data/processed/
├── bdc_entity_reference.parquet    # Entity master table
├── bdc_quarterly_panel.parquet     # Quarterly panel (wide format)
└── bdc_feature_panel.parquet       # Feature panel with derived metrics

src/
├── data/
│   └── panel.py                     # Entity resolution & panel construction
└── features/
    └── feature_engineering.py      # Feature computation & target labeling
```

---

## Design Decisions

1. **Tag Pattern Matching**: Uses substring matching (case-insensitive) rather than exact tag names because:
   - XBRL tags vary across filers (e.g., "TotalAssets" vs "AssetsTotal")
   - More robust to naming variations

2. **Latest Value Aggregation**: Defaults to "latest" aggregation because:
   - Financial statements may have amendments
   - Most recent value is typically the most accurate

3. **Safe Division**: Uses `np.where` instead of pandas `.replace()` to avoid recursion issues with mixed dtypes

4. **SOI Features Optional**: Portfolio features are optional because:
   - SOI data may not be available for all periods
   - Some BDCs may not report detailed holdings

5. **Target Lagging**: Targets are forward-looking (next quarter) but aligned with current-period features for proper train/test split

---

## Next Steps

With Phase 2 complete, the next phases are:

1. **Modeling** (`src/models/`):
   - Baseline models (naive, rolling mean, ridge regression)
   - ML models (XGBoost, LightGBM, Random Forest)
   - Backtesting with expanding-window CV

2. **Dashboard** (`dashboards/`):
   - Streamlit app with Overview, Portfolio, and Compare tabs
   - Visualizations pulling from feature panel

3. **Documentation**:
   - Data dictionary
   - Model cards
   - Final report

---

## Testing

Run end-to-end tests:
```bash
pytest tests/test_panel_features.py -v
```

Tests verify:
- Entity reference table structure
- Quarterly panel structure
- Feature computation
- Target label creation

---

## Notes

- All modules are **config-driven** via `config.yml`
- Outputs are **Parquet format** for efficient storage and fast loading
- Code is **modular** and can be extended with additional features/metrics
- **Type hints** and **docstrings** included for maintainability

