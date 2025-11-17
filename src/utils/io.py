from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def get_project_root() -> Path:
    """
    Return the project root directory.

    Assumes this file lives in `src/utils/`, so project root is two levels up.
    """
    return Path(__file__).resolve().parents[2]


def resolve_from_root(*parts: str) -> Path:
    """
    Resolve a path relative to the project root.

    Examples
    --------
    >>> resolve_from_root("data", "raw")
    PosixPath('/.../data/raw')
    """
    return get_project_root().joinpath(*parts)


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists, creating it if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 hash of a file.

    Parameters
    ----------
    path : Path
        Path to the file to hash.
    chunk_size : int
        Size of chunks to read into memory, in bytes.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(
    root: Path,
    include_extensions: Iterable[str] | None = None,
) -> Iterable[Path]:
    """
    Yield files under `root`, optionally filtered by extension.

    Parameters
    ----------
    root : Path
        Directory to scan recursively.
    include_extensions : Iterable[str] | None
        If provided, only files whose suffix is in this collection (case-insensitive)
        will be yielded.

    Yields
    ------
    Path
        Paths to files matching the criteria.
    """
    include_ext_lower = {ext.lower() for ext in include_extensions} if include_extensions else None

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if include_ext_lower is not None:
            if p.suffix.lower() not in include_ext_lower:
                continue
        yield p


