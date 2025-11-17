# Phase 1 Implementation: Data Acquisition, Ingestion & Storage

## Overview

This document describes the implementation of **Phase 1** of the BDC Analytics & Forecasting Platform, which establishes the foundation for data processing:

1. **Project Configuration** - Centralized config management
2. **Data Acquisition & Ingestion** - Manifest generation for raw SEC BDC files
3. **Schema & Joins** - DuckDB-based numeric facts store construction
4. **Data Cleaning & Quality Checks** - Normalization and validation

---

## 1. Project Configuration (`config.yml` + `src/config.py`)

### Purpose

Provides a **centralized, config-driven** approach to managing paths, pipeline options, and settings across all modules.

### Implementation

**File: `config.yml`** (project root)

```yaml
paths:
  data_root: data
  raw_root: data/raw
  processed_root: data/processed
  duckdb_path: data/bdc_store.duckdb
  manifest_path: data/bdc_manifest.parquet

ingest:
  include_extensions: [".tsv", ".zip", ".json", ".htm"]
  periods: []

build_store:
  periods: []
  numeric_table_name: numeric_facts
  write_parquet: true
  overwrite_parquet: false
```

**Module: `src/config.py`**

**Functions:**
- `load_config(config_path=None)`: Loads and caches YAML config (singleton pattern)
- `get_path(key)`: Returns absolute Path for entries under `paths` section
- `pretty_print_config()`: JSON-formatted config dump for debugging

**Key Features:**
- **No hardcoded paths** - All file locations defined in one place
- **Environment-agnostic** - Works across different machines/directories
- **Extensible** - Easy to add new config sections for future modules

### Benefits

- **Reproducibility**: All paths are version-controlled
- **Maintainability**: Change data directory structure without touching code
- **Consistency**: Notebooks and CLI use the same paths

---

## 2. IO Utilities (`src/utils/io.py`)

### Purpose

Provides reusable helper functions for file operations, path resolution, and hashing.

### Implementation

**Functions:**

1. **`get_project_root()`**
   - Returns project root directory (assumes `src/utils/` structure)
   - Used for resolving relative paths

2. **`resolve_from_root(*parts)`**
   - Builds paths relative to project root
   - Example: `resolve_from_root("data", "raw")` → `project_root/data/raw`

3. **`ensure_parent_dir(path)`**
   - Creates parent directories if they don't exist
   - Prevents `FileNotFoundError` when writing files

4. **`sha256_file(path, chunk_size=1MB)`**
   - Computes SHA-256 hash of a file
   - Uses chunked reading for memory efficiency
   - Critical for **reproducibility tracking**

5. **`iter_files(root, include_extensions=None)`**
   - Recursively yields files under `root`
   - Optional extension filtering (case-insensitive)
   - Used by ingestion manifest builder

### Why These Utilities Matter

- **Hashing**: Enables data lineage and change detection
- **Path resolution**: Consistent across Windows/Linux/Mac
- **File iteration**: Efficient scanning of large directory trees

---

## 3. Data Acquisition & Ingestion Manifest (`src/data/ingest.py`)

### Purpose

Scans `data/raw/` and builds a **reproducible manifest** cataloging all raw SEC BDC files with metadata, hashes, and source URLs.

### Implementation

**Data Structure: `ManifestRow`**

```python
@dataclass
class ManifestRow:
    period: str                    # e.g., "2025_Q1"
    relative_path: str             # Relative to data/raw/
    filename: str                  # Just the filename
    file_type: str                 # Extension (e.g., ".tsv")
    size_bytes: int                # File size
    sha256: str                    # SHA-256 hash
    mtime_utc: str                 # Modification time (ISO format)
    source_url: Optional[str]      # SEC download URL (if available)
```

**Function: `build_manifest()`**

**Process:**
1. Reads `config.yml` to get `raw_root` and `ingest.include_extensions`
2. Recursively scans `data/raw/` for matching files
3. For each file:
   - Extracts period from directory structure
   - Computes SHA-256 hash
   - Records file metadata (size, mtime)
   - Attempts to extract source URL from `bdc_metadata.json`
4. Returns DataFrame sorted by `(period, relative_path)`

**Function: `save_manifest(df, overwrite=True)`**

- Writes manifest to Parquet at `data/bdc_manifest.parquet`
- Parquet format chosen for:
  - Fast columnar reads
  - Efficient filtering/querying
  - Small file size

### Output

**File: `data/bdc_manifest.parquet`**

- **165 rows** (one per raw file)
- Columns: `period`, `relative_path`, `filename`, `file_type`, `size_bytes`, `sha256`, `mtime_utc`, `source_url`

### Use Cases

1. **Reproducibility**: Verify raw data hasn't changed (compare hashes)
2. **Audit Trail**: Track which files were processed and when
3. **Change Detection**: Identify new/updated files when re-running ingestion
4. **Data Lineage**: Link processed outputs back to source files

### Command-Line Usage

```bash
python -m src.data.ingest
```

---

## 4. Schema & Joins - DuckDB Store Builder (`src/data/build_store.py`)

### Purpose

Loads raw TSV files, performs schema joins, and builds a **unified DuckDB store** of numeric facts with submission and tag metadata.

### Implementation

**Function: `build_numeric_store(periods=None)`**

**Process Overview:**

1. **Load Period Tables** (`_load_period_tables()`)
   - Loads TSV files from `data/raw/<period>/datasets/`:
     - `num.tsv` - Numeric facts
     - `sub.tsv` - Submission metadata
     - `tag.tsv` - Tag dictionary
     - `pre.tsv` - Presentation relationships
     - `cal.tsv`, `txt.tsv`, `non.tsv` - Additional tables
   - Also loads `soi.tsv` from period root (for future use)
   - Normalizes all column names to lowercase

2. **Build Numeric Facts** (`_build_numeric_facts_for_period()`)
   
   **Step 1: Numeric Conversion**
   - Detects numeric column in `num.tsv` (searches: `val`, `value`, `amount`, `numeric`, `amt`)
   - Converts to `val_num` using `pd.to_numeric(..., errors="coerce")`
   - Handles non-numeric entries gracefully (e.g., ranges, notes)

   **Step 2: Date Conversion**
   - Attempts datetime conversion for date-like columns:
     - `ddate` → `ddate_dt`
     - `date`, `period`, `dd` → `{col}_dt`
   - Uses `pd.to_datetime(..., errors="coerce")` for safety

   **Step 3: Schema Joins**
   - **num + sub**: Join on `adsh` (accession number = filing ID)
     - Brings in: `cik`, `name`, `form`, filing date, etc.
   - **+ tag**: Join on `["tag", "version"]` (preferred) or just `tag`
     - Brings in: `tlabel` (human-readable tag description)
   - **+ pre**: Optional join on `["adsh", "tag"]`
     - Brings in: `stmt` (statement type: Balance Sheet, Income Statement, etc.)

   **Step 4: Column Selection**
   - Keeps curated set: `adsh`, `tag`, `val_num`, `uom`, `ddate`, `qtrs`, `form`, `cik`, `name`, `tlabel`, `period`
   - Adds `period` column for time-series tracking

3. **Store in DuckDB** (`_append_to_duckdb()`)
   - Creates table `numeric_facts` if it doesn't exist
   - Appends each period's data (supports incremental updates)
   - DuckDB chosen for:
     - Fast analytical queries
     - SQL interface
     - Efficient columnar storage
     - No external server required

4. **Write Parquet** (optional)
   - Per-period Parquet files: `data/processed/<period>/numeric_facts.parquet`
   - Excludes datetime columns (uses original `ddate` integer instead)
   - Enables fast loading in notebooks/analysis

### Output

**DuckDB Store: `data/bdc_store.duckdb`**

- **Table: `numeric_facts`**
- **~15.8 million rows** (aggregated across all periods)
- Columns: `adsh`, `tag`, `val_num`, `uom`, `ddate`, `qtrs`, `form`, `cik`, `name`, `tlabel`, `period`, `ddate_dt`

**Per-Period Parquet: `data/processed/<period>/numeric_facts.parquet`**

- One file per period (15 files total)
- Same schema as DuckDB (minus datetime columns)
- Enables parallel processing and fast columnar reads

### Data Quality Features

1. **Missing Value Handling**
   - Non-numeric values → `pd.NA` (not dropped)
   - Invalid dates → `pd.NaT`
   - Preserves data for quality analysis

2. **Duplicate Handling**
   - Multiple values for same `(adsh, tag, ddate)` are preserved
   - Can be deduplicated later if needed
   - Reflects "as-filed" nature of SEC data

3. **Unit Normalization**
   - Preserves `uom` (unit of measure) column
   - Common units: `USD`, `shares`, `pure` (percentages)
   - Future: Can normalize to common units if needed

### Command-Line Usage

```bash
python -m src.data.build_store
```

**Output:**
```
-> Processing period: 2022_Q4
   - Appended 749,676 rows to DuckDB table 'numeric_facts'
   - Wrote Parquet to: data/processed/2022_Q4/numeric_facts.parquet
...
[OK] DuckDB store updated at: data/bdc_store.duckdb
```

---

## 5. Data Cleaning & Quality Checks

### Implicit Cleaning (Built into `build_store.py`)

1. **Type Conversion**
   - Numeric values: Coerced to float, invalid → `pd.NA`
   - Dates: Parsed with error handling

2. **Column Normalization**
   - All column names lowercased for consistency
   - Handles variations in SEC file formats

3. **Join Safety**
   - Left joins used to preserve all numeric facts
   - Missing metadata doesn't drop facts

### Explicit Quality Checks (Future Enhancement)

**Planned validations:**
- Fair value ≈ cost + unrealized gain/loss (sanity check)
- Outlier detection (Z-score > 3)
- Missing tag detection (common tags expected but absent)
- Unit consistency (same tag should have same unit)

**Note:** These can be added as separate validation module (`src/utils/validators.py`)

---

## File Structure

```
config.yml                          # Central configuration

src/
├── config.py                       # Config loader
├── utils/
│   └── io.py                       # IO helpers (hashing, paths)
└── data/
    ├── ingest.py                   # Manifest builder
    └── build_store.py              # DuckDB store builder

data/
├── raw/                            # Raw SEC TSV files
│   ├── 2022_Q4/
│   │   ├── datasets/              # num.tsv, sub.tsv, tag.tsv, etc.
│   │   └── soi.tsv
│   └── ...
├── bdc_manifest.parquet            # Ingestion manifest (165 rows)
├── bdc_store.duckdb                # DuckDB numeric facts store
└── processed/
    ├── 2022_Q4/
    │   └── numeric_facts.parquet   # Per-period Parquet
    └── ...
```

---

## Design Decisions

### 1. Why DuckDB?

- **Fast analytical queries**: Columnar storage optimized for aggregations
- **SQL interface**: Familiar query language
- **No server required**: Embedded database, no setup overhead
- **Efficient joins**: Fast merges of large tables
- **Python integration**: Native pandas interop

### 2. Why Parquet for Outputs?

- **Columnar format**: Fast column reads (perfect for analytics)
- **Compression**: Smaller file sizes than CSV
- **Schema preservation**: Maintains data types
- **Cross-platform**: Works with pandas, polars, DuckDB, Spark

### 3. Why Manifest?

- **Reproducibility**: Hash-based change detection
- **Audit trail**: Track what was processed when
- **Data lineage**: Link outputs to source files
- **Compliance**: Document data sources for reports

### 4. Why Config-Driven?

- **No hardcoded paths**: Works across environments
- **Version control**: Config changes are tracked
- **Flexibility**: Easy to add new settings
- **Documentation**: Config serves as documentation

### 5. Why Per-Period Parquet?

- **Parallel processing**: Load only needed periods
- **Fast columnar reads**: Efficient for feature engineering
- **Backup**: Redundant storage (DuckDB + Parquet)
- **Notebook-friendly**: Easy to load in Jupyter

---

## Usage Examples

### Build Manifest

```python
from src.data.ingest import build_manifest, save_manifest

manifest_df = build_manifest()
save_manifest(manifest_df, overwrite=True)
```

### Build Store

```python
from src.data.build_store import build_numeric_store

# Process all periods
build_numeric_store()

# Process specific periods
build_numeric_store(periods=["2025_Q1", "2025_Q2"])
```

### Query DuckDB Store

```python
import duckdb
from src.config import get_path

duckdb_path = get_path("duckdb_path")
conn = duckdb.connect(str(duckdb_path))

# Count rows per period
query = """
SELECT period, COUNT(*) as row_count
FROM numeric_facts
GROUP BY period
ORDER BY period
"""
df = conn.execute(query).df()
print(df)

conn.close()
```

### Load Per-Period Parquet

```python
import pandas as pd
from src.config import get_path

processed_root = get_path("processed_root")
period = "2025_Q1"

df = pd.read_parquet(processed_root / period / "numeric_facts.parquet")
print(f"Loaded {len(df):,} rows for {period}")
```

---

## Results Summary

### Data Coverage

- **15 periods** processed:
  - Quarterly: `2022_Q4`, `2023_Q1-Q4`, `2024_Q1-Q4`, `2025_Q1`
  - Monthly: `2025_04`, `2025_05`, `2025_06`, `2025_07`, `2025_08`

- **~15.8 million numeric facts** in DuckDB store
- **165 raw files** cataloged in manifest

### Output Artifacts

1. **`data/bdc_manifest.parquet`**: 165 rows (file catalog)
2. **`data/bdc_store.duckdb`**: ~15.8M rows (unified numeric facts)
3. **`data/processed/<period>/numeric_facts.parquet`**: 15 files (per-period snapshots)

---

## Troubleshooting

### Issue: Config file not found

**Error:** `FileNotFoundError: Config file not found at: ...`

**Solution:** Ensure `config.yml` exists at project root. The path resolution assumes `src/config.py` is two levels below project root.

### Issue: Unicode encoding errors

**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution:** Replaced emoji characters with ASCII alternatives (`[OK]` instead of `✅`). All print statements are now Windows-compatible.

### Issue: Parquet timestamp precision errors

**Error:** `ArrowInvalid: Casting from timestamp[ns] to timestamp[us] would lose data`

**Solution:** Exclude datetime columns from Parquet writes. Original integer `ddate` column is preserved, and datetime columns remain in DuckDB.

### Issue: Duplicate rows in DuckDB

**Note:** Re-running `build_numeric_store()` will append duplicate rows. To rebuild cleanly:
1. Delete `data/bdc_store.duckdb`
2. Re-run `build_numeric_store()`

**Future enhancement:** Add deduplication logic based on `(period, adsh, tag, ddate)`.

---

## Next Steps

With Phase 1 complete, the pipeline can proceed to:

1. **Phase 2**: Entity resolution, panel construction, and feature engineering
2. **Modeling**: Train forecasting models using the cleaned numeric facts
3. **Dashboard**: Build Streamlit app querying DuckDB store

---

## Testing

### Unit Tests

```bash
pytest tests/test_ingest.py      # Test manifest builder
pytest tests/test_build_store.py # Test store builder
```

### Manual Verification

```python
# Verify manifest
import pandas as pd
manifest = pd.read_parquet("data/bdc_manifest.parquet")
print(f"Manifest: {len(manifest)} files")

# Verify DuckDB store
import duckdb
conn = duckdb.connect("data/bdc_store.duckdb")
count = conn.execute("SELECT COUNT(*) FROM numeric_facts").fetchone()[0]
print(f"DuckDB: {count:,} rows")
conn.close()
```

---

## Notes

- All modules are **type-hinted** and **documented**
- Code follows **modular design** for easy extension
- **Config-driven** approach ensures reproducibility
- **Parquet + DuckDB** dual storage for flexibility
- **Incremental updates** supported (append to DuckDB)

---

## References

- **SEC BDC Data Sets**: https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets
- **SEC BDC README**: https://www.sec.gov/files/bdc_readme.pdf
- **DuckDB Documentation**: https://duckdb.org/docs/
- **Parquet Format**: https://parquet.apache.org/

