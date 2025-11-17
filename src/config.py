from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CONFIG_CACHE: Dict[str, Any] | None = None


def _default_config_path() -> Path:
    """
    Return the default config path (project_root / 'config.yml').

    Assumes this file lives in src/, so project_root is one level up from src/.
    """
    return Path(__file__).resolve().parents[1] / "config.yml"


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load the project configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path | None
        Optional explicit path to a YAML config. If None, uses the default
        project_root / 'config.yml'.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Notes
    -----
    The configuration is cached in-memory so repeated calls are cheap.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file at {path} must define a mapping, got: {type(cfg)}")

    _CONFIG_CACHE = cfg
    return cfg


def get_path(key: str) -> Path:
    """
    Convenience accessor for path-like entries under `paths` in the config.

    Examples
    --------
    >>> get_path("raw_root")
    PosixPath('/.../data/raw')
    """
    cfg = load_config()
    paths_cfg = cfg.get("paths", {})
    rel = paths_cfg.get(key)
    if rel is None:
        raise KeyError(f"Key '{key}' not found under 'paths' section in config.yml")

    project_root = Path(__file__).resolve().parents[1]
    return project_root / rel


def pretty_print_config() -> str:
    """
    Return a JSON-formatted string of the loaded configuration.

    Useful for debugging from notebooks or CLI.
    """
    cfg = load_config()
    return json.dumps(cfg, indent=2, sort_keys=True)


