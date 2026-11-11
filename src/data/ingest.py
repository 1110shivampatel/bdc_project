from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import get_path, load_config
from src.utils.io import ensure_parent_dir, iter_files, sha256_file


@dataclass
class ManifestRow:
    """
    Row representing a single raw data file in the ingestion manifest.
    """

    period: str
    relative_path: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    mtime_utc: str
    source_url: Optional[str] = None


def _extract_period_from_path(path: Path, raw_root: Path) -> str:
    """
    Extract the period key from the path (immediate child of raw_root).
    """
    relative = path.relative_to(raw_root)
    return relative.parts[0]  # e.g., '2025_Q1' or '2025_06'


def _load_metadata_for_period(period_dir: Path) -> Dict[str, str]:
    """
    Try to load bdc_metadata.json for a period, returning any URLs we can find.

    The SEC metadata schema can evolve; we keep this flexible and optional.
    """
    meta_path = period_dir / "bdc_metadata.json"
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return {}

    # Heuristic: look for url-like values in the JSON.
    urls: Dict[str, str] = {}
    if isinstance(meta, dict):
        for key, value in meta.items():
            if isinstance(value, str) and value.startswith("http"):
                urls[key] = value
    return urls


def build_manifest() -> pd.DataFrame:
    """
    Scan data/raw and build a manifest of all relevant files.

    Returns
    -------
    pd.DataFrame
        Manifest with one row per file, including hashes and optional URLs.

    Notes
    -----
    - Only files whose extensions are in config['ingest']['include_extensions']
      are included.
    - The `source_url` column is best-effort, based on bdc_metadata.json.
    """
    cfg = load_config()
    raw_root = get_path("raw_root")
    ingest_cfg = cfg.get("ingest", {})
    include_ext = ingest_cfg.get("include_extensions", [".tsv", ".zip", ".json", ".htm"])

    rows: List[ManifestRow] = []

    # Build a small cache of metadata per period
    period_meta: Dict[str, Dict[str, str]] = {}
    for period_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        period_key = period_dir.name
        period_meta[period_key] = _load_metadata_for_period(period_dir)

    for path in iter_files(raw_root, include_extensions=include_ext):
        period_key = _extract_period_from_path(path, raw_root)
        rel_path = path.relative_to(raw_root).as_posix()

        # Best-effort source URL: if the file is the ZIP, look up a URL in metadata
        meta = period_meta.get(period_key, {})
        source_url = None
        if path.suffix.lower() == ".zip":
            # Use any URL we find; if multiple, this column can be refined later
            source_url = next(iter(meta.values()), None)

        stat = path.stat()
        mtime_utc = datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z"

        row = ManifestRow(
            period=period_key,
            relative_path=rel_path,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            size_bytes=stat.st_size,
            sha256=sha256_file(path),
            mtime_utc=mtime_utc,
            source_url=source_url,
        )
        rows.append(row)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.sort_values(["period", "relative_path"], inplace=True)
    return df


def save_manifest(df: pd.DataFrame, overwrite: bool | None = None) -> Path:
    """
    Save the manifest to the path specified in config['paths']['manifest_path'].

    Parameters
    ----------
    df : pd.DataFrame
        Manifest dataframe.
    overwrite : bool | None
        If False and the file exists, raise an error.
        If True, overwrite existing.
        If None, default to overwrite (safe for now, as file is derived).

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    cfg = load_config()
    manifest_rel = cfg.get("paths", {}).get("manifest_path", "data/bdc_manifest.parquet")
    data_root = get_path("data_root")
    manifest_path = data_root.parent / manifest_rel

    if manifest_path.exists() and overwrite is False:
        raise FileExistsError(f"Manifest already exists at {manifest_path} and overwrite=False")

    ensure_parent_dir(manifest_path)
    df.to_parquet(manifest_path, index=False)
    return manifest_path


if __name__ == "__main__":
    # Lightweight CLI: python -m src.data.ingest
    manifest_df = build_manifest()
    out_path = save_manifest(manifest_df, overwrite=True)
    print(f"[OK] Wrote manifest with {len(manifest_df):,} rows to: {out_path}")


