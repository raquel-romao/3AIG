"""Configuration helpers for local and Kaggle runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_path(path_like: str | Path) -> Path:
    """Return absolute path, resolving relative paths from project root."""
    p = Path(path_like)
    return p if p.is_absolute() else project_root() / p
