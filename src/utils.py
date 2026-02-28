"""Shared utilities: config, logging, seeding, IO helpers."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped format.

    Uses force=True and sys.stdout so logs always appear in Jupyter/Colab cells.
    """
    import sys
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def set_global_seed(seed: int) -> None:
    """Set random seed across python, numpy, and tensorflow for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(path: str | Path, data: Any) -> None:
    """Save data as JSON with pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save numpy arrays as compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_paths(cfg: dict) -> dict:
    """Resolve paths: keys starting with 'data_raw_' use data_root; rest use drive_root."""
    runtime = cfg.get("runtime", {})
    if runtime.get("colab_mode", False):
        root = Path(runtime["drive_root"])
        data_root = Path(runtime["data_root"]) if runtime.get("data_root") else root
    else:
        root = Path(".")
        data_root = Path(runtime["data_root"]) if runtime.get("data_root") else root

    paths = cfg.get("paths", {})
    resolved = {}
    for key, val in paths.items():
        base = data_root if key.startswith("data_raw_") else root
        resolved[key] = str(base / val)
    cfg["_resolved_paths"] = resolved
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Get a resolved path from config."""
    return Path(cfg["_resolved_paths"][key])


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
