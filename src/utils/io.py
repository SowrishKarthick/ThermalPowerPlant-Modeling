"""
Input/output helper utilities for simulation runs.

The simulation core and dashboard use these helpers to discover run folders,
load structured data, and persist new results without duplicating filesystem
code in multiple places.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


SIM_OUT_DIR = Path("sim_out")


def ensure_output_root() -> Path:
    """Create the `sim_out/` directory if it does not exist."""
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return SIM_OUT_DIR


def _format_runtime_label(runtime_hours: float) -> str:
    """Return a filesystem-safe runtime label, e.g., '24h' or '12-5h'."""
    value = f"{runtime_hours:.2f}".rstrip("0").rstrip(".")
    if not value:
        value = "0"
    safe_value = value.replace(".", "-")
    return f"{safe_value}h"


def create_run_directory(timestamp: str | None = None, runtime_hours: float | None = None) -> Path:
    """
    Create a new timestamped run directory and return its path.

    The format is `sim_run_YYYY-MM-DD_HH-MM`. If a directory with the same
    name already exists, a numeric suffix is appended to avoid overwriting
    previous runs.
    """
    from datetime import datetime

    ensure_output_root()
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_name = f"sim_run_{ts}"
    if runtime_hours is not None:
        base_name = f"{base_name}_{_format_runtime_label(runtime_hours)}"
    run_path = SIM_OUT_DIR / base_name
    counter = 1
    while run_path.exists():
        run_path = SIM_OUT_DIR / f"{base_name}_{counter}"
        counter += 1
    run_path.mkdir(parents=True, exist_ok=False)
    return run_path


def list_run_directories() -> List[Path]:
    """Return available run folders sorted by newest first."""
    if not SIM_OUT_DIR.exists():
        return []
    return sorted([p for p in SIM_OUT_DIR.iterdir() if p.is_dir()], reverse=True)


def save_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    """Persist a DataFrame to CSV using UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a dictionary to JSON with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@lru_cache(maxsize=64)
def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning an empty dict if the file is missing."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=32)
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file, returning an empty DataFrame if the file is missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clear_load_caches() -> None:
    """Reset memoised loaders (useful after creating a new run)."""
    load_json.cache_clear()
    load_csv.cache_clear()
