"""Comparison of data-processing approaches.

1. Storage formats: full CSV read, CSV with compact dtypes, chunked CSV,
   Parquet (all columns / two columns / one partition).
2. Engines: the same aggregation computed in pandas and in SQLite.
Criteria: time, peak memory, size, and whether the results match.
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src import config
from src.database import connect
from src.features import load_daily, load_hourly
from src.loader import OPTIMISED_DTYPES, find_raw_csv
from src.utils import file_size_mb, frame_memory_mb, measure, save_json, save_table


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (what the OS really gave it)."""
    try:
        import resource                     # Linux, macOS, Colab
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except ImportError:                     # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20


def _chunked_rows(csv) -> int:
    total = 0
    for chunk in pd.read_csv(csv, usecols=["Timestamp", "Close"], chunksize=config.CHUNKSIZE,
                             dtype={"Timestamp": "float64", "Close": "float32"}):
        total += len(chunk)
    return total


def storage_cases() -> dict:
    """Ways to read the same 7.8 million minute candles."""
    csv, parquet = find_raw_csv(), config.MINUTE_PARQUET_DIR
    return {
        "csv_default": ("CSV, default dtypes (baseline)", lambda: pd.read_csv(csv), csv),
        "csv_compact": ("CSV, compact dtypes", lambda: pd.read_csv(csv, dtype=OPTIMISED_DTYPES), csv),
        "csv_chunked": ("CSV, chunked, 2 columns", lambda: _chunked_rows(csv), csv),
        "parquet_all": ("Parquet, all columns", lambda: pd.read_parquet(parquet), parquet),
        "parquet_cols": ("Parquet, 2 columns", lambda: pd.read_parquet(parquet, columns=["ts", "close"]), parquet),
        "parquet_year": ("Parquet, one year (partition)",
                         lambda: pd.read_parquet(parquet, filters=[("year", "=", 2025)]), parquet / "year=2025"),
    }


def _child(key: str, repeat: int = 3) -> None:
    """Runs in a fresh Python process, so every case starts with the same memory."""
    name, read, _ = storage_cases()[key]
    base = peak_rss_mb()
    best, memory, rows, peak = float("inf"), float("nan"), 0, 0.0
    for i in range(repeat):
        start = time.perf_counter()
        result = read()
        best = min(best, time.perf_counter() - start)
        if i == 0:      # memory of the first read only: later reads may reuse freed pages
            peak = peak_rss_mb() - base
            if isinstance(result, pd.DataFrame):
                memory, rows = frame_memory_mb(result), len(result)
            else:
                rows = int(result)
        del result
        gc.collect()
    print(json.dumps({"seconds": best, "peak_memory_mb": peak,
                      "dataframe_memory_mb": memory, "rows": rows}))


def _measure_case(key: str) -> dict:
    command = [sys.executable, "-c", f"from src.benchmark import _child; _child({key!r})"]
    done = subprocess.run(command, cwd=config.ROOT, capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise RuntimeError(f"benchmark case {key} failed: {done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def benchmark_storage(verbose: bool = True) -> pd.DataFrame:
    """Time (best of 3) and extra peak process memory of every reading approach."""
    rows = []
    for key, (name, _, path) in storage_cases().items():
        result = _measure_case(key)
        rows.append({"approach": name, **result, "storage_mb": file_size_mb(path)})
        if verbose:
            print(f"  {name:<32} {result['seconds']:6.2f} s   peak memory +{result['peak_memory_mb']:7.1f} MB")
    table = pd.DataFrame(rows)
    table["speedup_vs_baseline"] = table.loc[0, "seconds"] / table["seconds"]
    table["memory_vs_baseline_pct"] = 100 * table["peak_memory_mb"] / table.loc[0, "peak_memory_mb"]
    return table


# --- pandas vs SQL -----------------------------------------------------------

def _yearly_pandas(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.year)
    out = pd.DataFrame({"days": g.size(), "avg_vol": g["realized_vol"].mean(),
                        "btc_traded": g["volume"].sum()})
    return out[out["days"] >= 300]


YEARLY_SQL = """
SELECT c.year, COUNT(*) AS days, AVG(d.realized_vol) AS avg_vol, SUM(d.volume) AS btc_traded
FROM candle_daily d JOIN calendar c ON c.date = d.date
GROUP BY c.year HAVING COUNT(*) >= 300 ORDER BY c.year"""


def _rolling_pandas(daily: pd.DataFrame) -> pd.Series:
    return daily["realized_vol"].rolling(30, min_periods=1).mean()


ROLLING_SQL = """
SELECT date, AVG(realized_vol) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS vol_30d
FROM candle_daily ORDER BY date"""


def _intraday_pandas(hourly: pd.DataFrame) -> pd.DataFrame:
    abs_ret = 100 * (hourly["close"] / hourly["close"].shift(1) - 1).abs()
    frame = pd.DataFrame({"weekday": hourly.index.day_of_week, "hour": hourly.index.hour, "r": abs_ret})
    return frame.dropna().groupby(["weekday", "hour"])["r"].mean()


INTRADAY_SQL = """
WITH r AS (SELECT date, hour, 100.0 * ABS(close / LAG(close) OVER (ORDER BY ts) - 1) AS r
           FROM candle_hourly)
SELECT c.weekday, r.hour, AVG(r.r) AS r
FROM r JOIN calendar c ON c.date = r.date
WHERE r.r IS NOT NULL GROUP BY c.weekday, r.hour ORDER BY c.weekday, r.hour"""


def _timed(func, repeat: int = 5):
    """Best of several runs, to reduce noise from the operating system."""
    best, result, peak = np.inf, None, 0.0
    for _ in range(repeat):
        with measure() as m:
            result = func()
        best, peak = min(best, m.seconds), max(peak, m.peak_mb)
    return result, best, peak


def benchmark_engines(verbose: bool = True) -> pd.DataFrame:
    """The same three aggregations in pandas (in memory) and in SQLite."""
    daily, hourly = load_daily(), load_hourly()
    rows = []
    with connect() as conn:
        tasks = [
            ("yearly summary (GROUP BY + HAVING)", lambda: _yearly_pandas(daily),
             lambda: pd.read_sql_query(YEARLY_SQL, conn),
             lambda p, s: np.allclose(p["avg_vol"].to_numpy(), s["avg_vol"].to_numpy())),
            ("30-day rolling mean (window)", lambda: _rolling_pandas(daily),
             lambda: pd.read_sql_query(ROLLING_SQL, conn),
             lambda p, s: np.allclose(p.to_numpy(), s["vol_30d"].to_numpy())),
            ("intraday profile (LAG + GROUP BY)", lambda: _intraday_pandas(hourly),
             lambda: pd.read_sql_query(INTRADAY_SQL, conn),
             lambda p, s: np.allclose(p.to_numpy(), s["r"].to_numpy())),
        ]
        for name, pandas_func, sql_func, same in tasks:
            p_res, p_sec, p_mem = _timed(pandas_func)
            s_res, s_sec, s_mem = _timed(sql_func)
            match = bool(same(p_res, s_res))
            rows.append({"task": name, "pandas_seconds": p_sec, "sql_seconds": s_sec,
                         "pandas_peak_mb": p_mem, "sql_peak_mb": s_mem,
                         "sql_slower_x": s_sec / p_sec, "results_match": match})
            if verbose:
                print(f"  {name:<36} pandas {1000 * p_sec:7.1f} ms | SQLite {1000 * s_sec:7.1f} ms "
                      f"| match: {match}")
    return pd.DataFrame(rows)


def run_benchmarks(verbose: bool = True) -> dict:
    if verbose:
        print("  Storage formats (7.8 M minute rows):")
    storage = benchmark_storage(verbose)
    if verbose:
        print("\n  pandas vs SQLite (same result, different engine):")
    engines = benchmark_engines(verbose)
    save_table(storage, "benchmark_storage")
    save_table(engines, "benchmark_engines")
    save_json({"storage": storage.to_dict("records"), "engines": engines.to_dict("records")},
              config.REPORTS_DIR / "benchmarks.json")
    return {"storage": storage, "engines": engines}
