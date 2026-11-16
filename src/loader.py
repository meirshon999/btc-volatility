"""Loading the raw minute CSV and storing it as a partitioned Parquet dataset.

The raw file has ~7.6 million rows (400 MB). It is read in chunks, every
chunk is converted to compact dtypes and written to Parquet partitioned by
year, so later steps can read one year at a time instead of everything.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src import config
from src.utils import file_size_mb, frame_memory_mb, measure, require

RAW_TO_CLEAN = {"Timestamp": "ts", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"}
VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]
# Compact dtypes used after loading: float32 halves the memory of prices.
OPTIMISED_DTYPES = {"Timestamp": "float64", "Open": "float32", "High": "float32",
                    "Low": "float32", "Close": "float32", "Volume": "float32"}


def find_raw_csv(raw_dir: Path = config.RAW_DIR) -> Path:
    """Locate the Kaggle CSV inside data/raw."""
    expected = raw_dir / config.RAW_CSV_NAME
    if expected.exists():
        return expected
    candidates = sorted(raw_dir.glob("*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No CSV found in {raw_dir}. Download the dataset from {config.DATASET_URL} "
        f"and put {config.RAW_CSV_NAME} into data/raw/."
    )


def optimise_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse the Unix timestamp and downcast values to float32."""
    chunk = chunk.rename(columns=RAW_TO_CLEAN)
    chunk = chunk.dropna(subset=["ts"])
    optimised = pd.DataFrame({
        "ts": pd.to_datetime(chunk["ts"].astype("int64"), unit="s").astype("datetime64[ns]")
    })
    for column in VALUE_COLUMNS:
        optimised[column] = chunk[column].astype("float32").to_numpy()
    return optimised


def float32_relative_error(original: pd.Series, compact: pd.Series) -> float:
    """Largest relative rounding error introduced by storing prices as float32."""
    original = original.to_numpy(dtype="float64")
    compact = compact.to_numpy(dtype="float64")
    mask = original > 0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(compact[mask] - original[mask]) / original[mask]))


def convert_csv_to_parquet(csv_path: Path | None = None,
                           out_dir: Path = config.MINUTE_PARQUET_DIR,
                           chunksize: int = config.CHUNKSIZE,
                           verbose: bool = True) -> dict:
    """Stream the CSV chunk by chunk into year-partitioned Parquet files.

    Returns statistics: rows, time, memory per chunk before and after the
    dtype optimisation, file sizes and the float32 rounding error.
    """
    csv_path = Path(csv_path) if csv_path else find_raw_csv()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cutoff = pd.Timestamp(config.DATA_END) + pd.Timedelta(days=1) if config.DATA_END else None
    stats = {"csv_path": csv_path.name, "chunksize": chunksize, "data_end": config.DATA_END,
             "chunks": 0, "rows": 0, "rows_without_timestamp": 0, "rows_after_data_end": 0,
             "memory_default_mb": 0.0, "memory_optimised_mb": 0.0,
             "max_float32_relative_error": 0.0}
    with measure() as m:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            stats["chunks"] += 1
            stats["memory_default_mb"] += frame_memory_mb(chunk)
            optimised = optimise_chunk(chunk)
            stats["memory_optimised_mb"] += frame_memory_mb(optimised)
            stats["rows_without_timestamp"] += len(chunk) - len(optimised)
            stats["max_float32_relative_error"] = max(
                stats["max_float32_relative_error"],
                float32_relative_error(chunk.loc[optimised.index, "Close"], optimised["close"]))

            if cutoff is not None:                  # keep the report period fixed
                after_end = optimised["ts"] >= cutoff
                stats["rows_after_data_end"] += int(after_end.sum())
                optimised = optimised[~after_end]
                if optimised.empty:
                    continue

            for year, part in optimised.groupby(optimised["ts"].dt.year, sort=True):
                target = out_dir / f"year={year}"
                target.mkdir(exist_ok=True)
                part.to_parquet(target / f"part-{i:03d}.parquet", index=False,
                                engine="pyarrow", compression="snappy")
            stats["rows"] += len(optimised)
            if verbose:
                print(f"  chunk {i + 1:>2}: {stats['rows']:>10,} rows  "
                      f"({optimised['ts'].iloc[0]:%Y-%m-%d} .. {optimised['ts'].iloc[-1]:%Y-%m-%d})")

    stats["seconds"] = m.seconds
    stats["peak_memory_mb"] = m.peak_mb
    stats["csv_size_mb"] = file_size_mb(csv_path)
    stats["parquet_size_mb"] = file_size_mb(out_dir)
    stats["memory_saving_pct"] = 100 * (1 - stats["memory_optimised_mb"] / stats["memory_default_mb"])
    stats["years"] = available_years(out_dir)
    return stats


def available_years(parquet_dir: Path = config.MINUTE_PARQUET_DIR) -> list[int]:
    """Years present in the partitioned dataset (from folder names year=YYYY)."""
    if not parquet_dir.exists():
        return []
    return sorted(int(p.name.split("=")[1]) for p in parquet_dir.glob("year=*") if p.is_dir())


def load_minute(years: list[int] | None = None,
                columns: list[str] | None = None) -> pd.DataFrame:
    """Read minute candles from Parquet, optionally only some years and columns."""
    require(config.MINUTE_PARQUET_DIR, "1 - convert CSV to Parquet")
    filters = [("year", "in", [int(y) for y in years])] if years else None
    if columns is not None and "ts" not in columns:
        columns = ["ts", *columns]
    df = pd.read_parquet(config.MINUTE_PARQUET_DIR, columns=columns,
                         filters=filters, engine="pyarrow")
    if "year" in df.columns:
        df = df.drop(columns="year")
    if not df["ts"].is_monotonic_increasing:
        df = df.sort_values("ts", ignore_index=True)
    return df


def iter_years(columns: list[str] | None = None) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (year, minute data of that year) - keeps memory low in later steps."""
    for year in available_years():
        yield year, load_minute([year], columns)
