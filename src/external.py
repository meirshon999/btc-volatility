"""Optional external daily series from FRED (VIX, S&P 500, U.S. dollar index).

They are used only in the correlation analysis, never as model inputs, so the
pipeline still runs offline: if the download fails, the step is skipped.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def series_path(name: str) -> Path:
    return config.EXTERNAL_DIR / f"{name}.csv"


def download_series(name: str, series_id: str, timeout: int = 30, attempts: int = 3) -> Path:
    """Download one FRED series as CSV into data/external (retries on network errors)."""
    url = config.FRED_URL.format(series_id=series_id)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = series_path(name)
    path.write_bytes(body)
    return path


def download_all(verbose: bool = True) -> dict[str, str]:
    """Try to download every configured series; report what worked."""
    status = {}
    for name, series_id in config.FRED_SERIES.items():
        try:
            path = download_series(name, series_id)
            rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
            status[name] = f"ok ({rows} rows)"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            status[name] = f"failed: {error}"
        if verbose:
            print(f"  {name:<10} {status[name]}")
    return status


def load_external() -> pd.DataFrame:
    """Cached external series as one daily table (empty if nothing was downloaded)."""
    frames = []
    for name in config.FRED_SERIES:
        path = series_path(name)
        if not path.exists():
            continue
        raw = pd.read_csv(path, na_values=["."])
        raw.columns = ["date", name]
        raw["date"] = pd.to_datetime(raw["date"])
        frames.append(raw.set_index("date")[name].astype("float64"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()
