from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import duckdb
import pandas as pd

from src.config import get_path, load_config
from src.utils.io import ensure_parent_dir


DEFAULT_KEEP_COLS: List[str] = [
    "adsh",
    "tag",
    "val_num",
    "uom",
    "unit",
    "decimals",
    "ddate",
    "ddate_dt",
    "qtrs",
    "form",
    "cik",
    "name",
    "tlabel",
]


def _period_dirs(raw_root: Path, explicit_periods: Optional[Iterable[str]]) -> List[Path]:
    """
    Enumerate period directories to process.
    """
    if explicit_periods:
        return [raw_root / p for p in explicit_periods if (raw_root / p).is_dir()]
    return sorted(p for p in raw_root.iterdir() if p.is_dir())


def _load_period_tables(raw_root: Path, period: str) -> Dict[str, pd.DataFrame]:
    """
    Load core TSV tables for a given period into pandas DataFrames.

    Parameters
    ----------
    raw_root : Path
        Root directory for raw data (e.g., data/raw).
    period : str
        Period key, e.g. '2025_Q1' or '2025_06'.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from table name to DataFrame.
    """
    period_dir = raw_root / period
    datasets_dir = period_dir / "datasets"

    tables: Dict[str, pd.DataFrame] = {}
    tsv_files = {
        "num": datasets_dir / "num.tsv",
        "sub": datasets_dir / "sub.tsv",
        "tag": datasets_dir / "tag.tsv",
        "pre": datasets_dir / "pre.tsv",
        "cal": datasets_dir / "cal.tsv",
        "txt": datasets_dir / "txt.tsv",
        "non": datasets_dir / "non.tsv",
    }

    for name, path in tsv_files.items():
        if path.exists():
            df = pd.read_csv(path, sep="\t", low_memory=False)
            df.columns = [c.lower() for c in df.columns]
            tables[name] = df
        else:
            tables[name] = pd.DataFrame()

    # SOI is outside datasets/ at the period root
    soi_path = period_dir / "soi.tsv"
    if soi_path.exists():
        soi_df = pd.read_csv(soi_path, sep="\t", low_memory=False)
        soi_df.columns = [c.lower() for c in soi_df.columns]
        tables["soi"] = soi_df
    else:
        tables["soi"] = pd.DataFrame()

    return tables


def _build_numeric_facts_for_period(tables: Dict[str, pd.DataFrame], period: str) -> pd.DataFrame:
    """
    Build a cleaned numeric facts table for a single period.

    This mirrors and generalizes the logic in notebooks/2025_Q1.ipynb.
    """
    num_df = tables["num"].copy()
    sub_df = tables["sub"].copy()
    tag_df = tables["tag"].copy()
    pre_df = tables["pre"].copy()

    if num_df.empty:
        raise ValueError(f"num.tsv is missing or empty for period {period}")

    # Identify numeric column and convert to val_num
    val_col = None
    for candidate in ["val", "value", "amount", "numeric", "amt"]:
        if candidate in num_df.columns:
            val_col = candidate
            break

    if val_col is None:
        raise ValueError(f"Could not detect numeric column in num for period {period}")

    num_df["val_num"] = pd.to_numeric(num_df[val_col], errors="coerce")

    # Try adding datetime conversions for date-like columns
    for col in ["ddate", "date", "period", "dd"]:
        if col in num_df.columns:
            num_df[f"{col}_dt"] = pd.to_datetime(num_df[col], errors="coerce")

    # Merge num + sub on adsh (filing)
    merged = num_df.merge(sub_df, on="adsh", how="left", suffixes=("", "_sub"))

    # Attach tag metadata; prefer join on tag + version if present
    if "version" in tag_df.columns and "version" in merged.columns:
        merged = merged.merge(tag_df, on=["tag", "version"], how="left", suffixes=("", "_tag"))
    else:
        merged = merged.merge(tag_df, on="tag", how="left", suffixes=("", "_tag"))

    # Attach presentation info if both adsh and tag exist in pre
    if not pre_df.empty and {"adsh", "tag"}.issubset(pre_df.columns):
        cols = [c for c in ["adsh", "tag", "stmt"] if c in pre_df.columns]
        merged = merged.merge(pre_df[cols], on=["adsh", "tag"], how="left")

    # Select a stable set of columns, intersecting with what exists
    keep_cols = [c for c in DEFAULT_KEEP_COLS if c in merged.columns]
    cleaned = merged[keep_cols].copy()
    cleaned["period"] = period

    return cleaned


def _append_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """
    Append a DataFrame to a DuckDB table, creating the table if it does not yet exist.
    """
    conn.register("df_tmp", df)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} AS
        SELECT * FROM df_tmp LIMIT 0
        """
    )
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_tmp")
    conn.unregister("df_tmp")


def build_numeric_store(
    periods: Optional[Iterable[str]] = None,
) -> None:
    """
    Build/refresh the numeric facts DuckDB table from all (or selected) periods.

    Parameters
    ----------
    periods : Iterable[str] | None
        Optional iterable of period keys (e.g., ['2025_Q1', '2025_06']).
        If None, infers all period directories under data/raw.

    Notes
    -----
    - This function can be rerun as new periods are added.
    - It appends to the DuckDB table; if you want a full rebuild, delete the
      DuckDB file beforehand.
    """
    cfg = load_config()
    raw_root = get_path("raw_root")
    processed_root = get_path("processed_root")
    duckdb_path = get_path("duckdb_path")

    build_cfg = cfg.get("build_store", {})
    table_name = build_cfg.get("numeric_table_name", "numeric_facts")
    write_parquet = bool(build_cfg.get("write_parquet", True))
    overwrite_parquet = bool(build_cfg.get("overwrite_parquet", False))

    explicit_cfg_periods = build_cfg.get("periods") or None
    periods_to_process = list(periods) if periods is not None else explicit_cfg_periods
    if periods_to_process is None:
        period_dirs = _period_dirs(raw_root, None)
    else:
        period_dirs = _period_dirs(raw_root, periods_to_process)

    if not period_dirs:
        raise RuntimeError(f"No period directories found under {raw_root}")

    ensure_parent_dir(duckdb_path)
    conn = duckdb.connect(str(duckdb_path))

    try:
        for period_dir in period_dirs:
            period = period_dir.name
            print(f"-> Processing period: {period}")

            tables = _load_period_tables(raw_root, period)
            numeric_df = _build_numeric_facts_for_period(tables, period)

            # Append to DuckDB
            _append_to_duckdb(conn, numeric_df, table_name)
            print(f"   - Appended {len(numeric_df):,} rows to DuckDB table '{table_name}'")

            # Optionally, write per-period Parquet under data/processed/<period>/
            if write_parquet:
                out_dir = processed_root / period
                out_path = out_dir / "numeric_facts.parquet"
                if out_path.exists() and not overwrite_parquet:
                    print(f"   - Skipping Parquet (exists, overwrite=False): {out_path}")
                else:
                    ensure_parent_dir(out_path)
                    # Exclude datetime columns from Parquet (we have original ddate as integer)
                    # This avoids PyArrow timestamp precision issues
                    df_for_parquet = numeric_df.copy()
                    datetime_cols = [col for col in df_for_parquet.columns 
                                    if pd.api.types.is_datetime64_any_dtype(df_for_parquet[col])]
                    if datetime_cols:
                        df_for_parquet = df_for_parquet.drop(columns=datetime_cols)
                    df_for_parquet.to_parquet(out_path, index=False)
                    print(f"   - Wrote Parquet to: {out_path}")

    finally:
        conn.close()
        print(f"[OK] DuckDB store updated at: {duckdb_path}")


if __name__ == "__main__":
    # Lightweight CLI: python -m src.data.build_store
    build_numeric_store()


