"""Small helpers shared by all modules: timing, memory, JSON and pipeline state."""
from __future__ import annotations

import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src import config


@dataclass
class Measurement:
    """Wall-clock time and peak traced memory of a code block."""
    seconds: float = 0.0
    peak_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@contextmanager
def measure() -> Iterator[Measurement]:
    """Measure elapsed time and peak memory allocated inside the block.

    tracemalloc sees allocations made by Python and NumPy (pandas columns
    are NumPy/Arrow buffers), which is what we need to compare approaches.
    """
    result = Measurement()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        result.peak_mb = peak / 2**20
        if not already_tracing:
            tracemalloc.stop()


def frame_memory_mb(df: pd.DataFrame) -> float:
    """Real memory footprint of a DataFrame, including object columns."""
    return float(df.memory_usage(deep=True).sum()) / 2**20


def file_size_mb(path: Path) -> float:
    """Size of a file, or the total size of all files inside a directory."""
    path = Path(path)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20
    return path.stat().st_size / 2**20


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, float):
        return None if np.isnan(value) else round(value, 6)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Save a result table as CSV in reports/tables."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.6g")
    return path


# --- pipeline state -------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Which pipeline steps have been completed, and when."""
    try:
        return load_json(config.STATE_PATH)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def mark_done(step: str, **info: Any) -> None:
    state = load_state()
    state[step] = {"finished_at": datetime.now().isoformat(timespec="seconds"), **info}
    save_json(state, config.STATE_PATH)


def require(path: Path, step_hint: str) -> Path:
    """Fail with a helpful message when an earlier pipeline step is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path.name} not found. Run step: {step_hint}")
    return path


# --- console output -------------------------------------------------------

def headline(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def print_table(df: pd.DataFrame, max_rows: int = 20, floatfmt: str = "{:,.4f}") -> None:
    """Print a DataFrame in a compact, aligned way."""
    with pd.option_context("display.max_rows", max_rows, "display.width", 160,
                           "display.max_columns", 20,
                           "display.float_format", floatfmt.format):
        print(df.head(max_rows).to_string())
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows")
