"""From clean minute candles to hourly/daily bars, realised volatility and model features.

Realised volatility (RV) of a day is the square root of the sum of squared
5-minute log returns inside that day. 5-minute sampling is the usual compromise
between using more information (finer sampling) and microstructure noise
(bid-ask bounce dominates at 1-minute sampling).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src import config
from src.loader import VALUE_COLUMNS, iter_years
from src.quality import clean_minute
from src.utils import require

OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
MINUTES_PER_DAY = 1440


def resample_ohlcv(minute: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate candles to a coarser frequency (e.g. '1h', '1D')."""
    bars = minute.set_index("ts").resample(rule, label="left", closed="left").agg(OHLCV_AGG)
    return bars.dropna(subset=["close"])


def realised_variance(close: pd.Series) -> pd.DataFrame:
    """Daily realised variance (in %^2) from an evenly spaced close-price series."""
    r = np.log(close).diff()
    day = r.index.floor("D")
    out = pd.DataFrame({
        "realized_var": (r ** 2).groupby(day).sum(min_count=1) * 1e4,
        "n_intervals": r.groupby(day).count(),
    })
    return out


def parkinson_volatility(high: pd.Series, low: pd.Series) -> pd.Series:
    """Range-based daily volatility estimator (Parkinson, 1980), in %."""
    return 100 * np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))


def halving_epoch(dates: pd.DatetimeIndex) -> np.ndarray:
    """Index of the halving epoch each date belongs to."""
    starts = pd.to_datetime([h["start_date"] for h in config.HALVINGS])
    return np.searchsorted(starts.values, dates.values, side="right") - 1


def days_since_halving(dates: pd.DatetimeIndex) -> np.ndarray:
    starts = pd.to_datetime([h["start_date"] for h in config.HALVINGS])
    epoch = halving_epoch(dates)
    return (dates - starts[epoch]).days.to_numpy()


def build_tables(verbose: bool = True) -> dict:
    """Clean every year of minute data and build hourly and daily tables.

    Processing is done one year at a time; only the small aggregated tables
    (hourly ~130k rows, daily ~5k rows) are kept in memory.
    """
    hourly_parts, daily_parts, close_5min = [], [], []
    cleaning = Counter()
    for year, minute in iter_years():
        minute, log = clean_minute(minute)
        cleaning.update(log)
        minute = minute.astype({c: "float64" for c in VALUE_COLUMNS})  # exact sums

        hourly_parts.append(resample_ohlcv(minute, "1h"))
        daily = resample_ohlcv(minute, "1D")
        day = minute["ts"].dt.floor("D")
        daily["minutes"] = minute.groupby(day).size()
        daily["zero_volume_share"] = (minute["volume"] == 0).groupby(day).mean()
        daily_parts.append(daily)
        close_5min.append(minute.set_index("ts")["close"].resample(config.RV_SAMPLING).last())
        if verbose:
            print(f"  {year}: {len(minute):>9,} minutes -> {len(daily):>3} days")

    hourly = pd.concat(hourly_parts)
    daily = pd.concat(daily_parts)
    # an empty 5-minute interval means no trade: the price stays the same
    rv = realised_variance(pd.concat(close_5min).ffill())
    daily = daily.join(rv, how="left")
    daily = add_daily_measures(daily)

    # the last day is usually incomplete (the file ends during the day)
    if daily["minutes"].iloc[-1] < 0.9 * MINUTES_PER_DAY:
        daily = daily.iloc[:-1]
        hourly = hourly[hourly.index < daily.index[-1] + pd.Timedelta(days=1)]

    hourly.index.name = "ts"
    daily.index.name = "date"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(config.HOURLY_PATH)
    daily.to_parquet(config.DAILY_PATH)
    return {"cleaning": dict(cleaning), "hourly_rows": len(hourly), "daily_rows": len(daily),
            "first_day": daily.index[0], "last_day": daily.index[-1]}


def add_daily_measures(daily: pd.DataFrame) -> pd.DataFrame:
    """Returns and volatility measures of each day (all in %)."""
    daily = daily.copy()
    daily["log_return"] = 100 * np.log(daily["close"] / daily["close"].shift(1))
    daily["realized_vol"] = np.sqrt(daily["realized_var"])
    daily["parkinson_vol"] = parkinson_volatility(daily["high"], daily["low"])
    daily["epoch"] = halving_epoch(daily.index)
    return daily


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(require(config.DAILY_PATH, "3 - build hourly/daily tables"))


def load_hourly() -> pd.DataFrame:
    return pd.read_parquet(require(config.HOURLY_PATH, "3 - build hourly/daily tables"))


# --- model inputs ---------------------------------------------------------

def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Predictors for the volatility of day t, using only days up to t-1.

    Every column is either shifted by at least one day or is a calendar
    value known in advance (weekday, days since halving), so there is no
    look-ahead. The target is log realised volatility of day t.
    """
    log_vol = np.log(daily["realized_vol"].where(daily["realized_vol"] > 0))
    past = log_vol.shift(1)
    log_volume = np.log1p(daily["volume"])

    f = pd.DataFrame(index=daily.index)
    for k in range(1, 8):
        f[f"log_vol_lag{k}"] = log_vol.shift(k)
    f["log_vol_mean_5"] = past.rolling(5).mean()        # HAR weekly component
    f["log_vol_mean_22"] = past.rolling(22).mean()      # HAR monthly component
    f["log_vol_std_22"] = past.rolling(22).std()
    f["return_lag1"] = daily["log_return"].shift(1)
    f["abs_return_lag1"] = daily["log_return"].abs().shift(1)
    f["negative_return_lag1"] = daily["log_return"].clip(upper=0).shift(1)   # leverage effect
    f["log_parkinson_lag1"] = np.log(daily["parkinson_vol"].where(daily["parkinson_vol"] > 0)).shift(1)
    f["log_volume_lag1"] = log_volume.shift(1)
    f["volume_change_lag1"] = log_volume.shift(1) - log_volume.shift(2)
    f["weekday"] = daily.index.day_of_week
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    f["days_since_halving"] = days_since_halving(daily.index)
    f["target_log_vol"] = log_vol
    return f


def split_periods(frame: pd.DataFrame, start: str = config.MODEL_START,
                  test_start: str = config.TEST_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train / test split - no shuffling."""
    frame = frame.loc[start:]
    train = frame.loc[: pd.Timestamp(test_start) - pd.Timedelta(days=1)]
    test = frame.loc[test_start:]
    return train, test
