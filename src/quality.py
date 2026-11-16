"""Data quality audit and cleaning rules for the minute candles.

The audit runs year by year over the Parquet partitions, so it never needs
the whole dataset in memory. Results go to reports/quality_report.json and
reports/tables/quality_*.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.loader import iter_years
from src.utils import save_json, save_table

MINUTE = pd.Timedelta(minutes=1)
EXTREME_MOVE = 0.10        # |log return| above 10 % within one minute
SPIKE_REVERSAL = 0.02      # a spike that returns to within 2 % next minute


def detect_spikes(close: pd.Series) -> pd.Series:
    """Isolated one-minute spikes: a jump of >10 % that reverses immediately.

    Real crashes do not revert within the next minute, bad ticks do.
    """
    r = np.log(close.astype("float64")).diff()
    nxt = r.shift(-1)
    return (r.abs() > EXTREME_MOVE) & (nxt.abs() > EXTREME_MOVE) & ((r + nxt).abs() < SPIKE_REVERSAL)


def audit_year(df: pd.DataFrame, year: int, prev_ts: pd.Timestamp | None) -> dict:
    """Quality indicators of one year of minute candles."""
    prices = df[["open", "high", "low", "close"]]
    ts = df["ts"]
    # the first year starts with the first record, the others on 1 January
    start = ts.iloc[0].floor("min") if prev_ts is None else pd.Timestamp(year=year, month=1, day=1)
    end = ts.iloc[-1].floor("min")
    expected = int((end - start) / MINUTE) + 1

    steps = ts.diff()
    if prev_ts is not None and len(ts):
        steps.iloc[0] = ts.iloc[0] - prev_ts
    gap_minutes = (steps / MINUTE - 1).clip(lower=0).fillna(0)

    body_high = np.maximum(df["open"], df["close"])
    body_low = np.minimum(df["open"], df["close"])
    invalid = (df["high"] < body_high) | (df["low"] > body_low) | (df["high"] < df["low"])
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])

    log_ret = np.log(df["close"].astype("float64")).diff()
    return {
        "year": year,
        "rows": len(df),
        "expected_minutes": expected,
        "coverage_pct": 100 * len(df) / expected,
        "missing_minutes": int(gap_minutes.sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "nan_values": int(df.isna().sum().sum()),
        "non_positive_prices": int((prices <= 0).any(axis=1).sum()),
        "invalid_candles": int(invalid.sum()),
        "zero_volume_share_pct": 100 * float((df["volume"] == 0).mean()),
        "flat_candle_share_pct": 100 * float(flat.mean()),
        "extreme_moves": int((log_ret.abs() > EXTREME_MOVE).sum()),
        "spikes": int(detect_spikes(df["close"]).sum()),
        "max_abs_minute_return_pct": 100 * float(log_ret.abs().max()),
    }


def largest_gaps(df: pd.DataFrame, prev_ts: pd.Timestamp | None, top: int = 10) -> pd.DataFrame:
    """The longest stretches without any candle."""
    ts = df["ts"]
    previous = ts.shift(1)
    if prev_ts is not None:
        previous.iloc[0] = prev_ts
    gaps = pd.DataFrame({"gap_start": previous, "gap_end": ts,
                         "missing_minutes": (ts - previous) / MINUTE - 1})
    return gaps[gaps["missing_minutes"] > 0].nlargest(top, "missing_minutes")


def extreme_minutes(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """The largest one-minute price moves (to check whether they are real)."""
    r = np.log(df["close"].astype("float64")).diff()
    out = df.assign(log_return_pct=100 * r, abs_move=r.abs())
    out = out.nlargest(top, "abs_move")
    return out[["ts", "open", "high", "low", "close", "volume", "log_return_pct"]]


def run_audit(verbose: bool = True) -> dict:
    """Audit every year partition and save the report."""
    rows, gaps, extremes = [], [], []
    prev_ts = first_ts = None
    for year, df in iter_years():
        first_ts = first_ts if first_ts is not None else df["ts"].iloc[0]
        rows.append(audit_year(df, year, prev_ts))
        gaps.append(largest_gaps(df, prev_ts))
        extremes.append(extreme_minutes(df))
        prev_ts = df["ts"].iloc[-1]
        if verbose:
            r = rows[-1]
            print(f"  {year}: {r['rows']:>9,} rows, coverage {r['coverage_pct']:6.2f} %, "
                  f"zero volume {r['zero_volume_share_pct']:5.1f} %")

    by_year = pd.DataFrame(rows)
    gaps_df = pd.concat(gaps).nlargest(15, "missing_minutes")
    extremes_df = (pd.concat(extremes)
                   .assign(abs_move=lambda d: d["log_return_pct"].abs())
                   .nlargest(15, "abs_move").drop(columns="abs_move"))

    report = {
        "first_timestamp": str(first_ts),
        "last_timestamp": str(prev_ts),
        "rows": int(by_year["rows"].sum()),
        "expected_minutes": int(by_year["expected_minutes"].sum()),
        "missing_minutes": int(by_year["missing_minutes"].sum()),
        "coverage_pct": 100 * by_year["rows"].sum() / by_year["expected_minutes"].sum(),
        "duplicate_timestamps": int(by_year["duplicate_timestamps"].sum()),
        "nan_values": int(by_year["nan_values"].sum()),
        "non_positive_prices": int(by_year["non_positive_prices"].sum()),
        "invalid_candles": int(by_year["invalid_candles"].sum()),
        "extreme_moves": int(by_year["extreme_moves"].sum()),
        "spikes": int(by_year["spikes"].sum()),
        "recommended_model_start": recommend_model_start(by_year),
        "configured_model_start": config.MODEL_START,
    }
    save_json(report, config.REPORTS_DIR / "quality_report.json")
    save_table(by_year, "quality_by_year")
    save_table(gaps_df, "quality_largest_gaps")
    save_table(extremes_df, "quality_extreme_minutes")
    return {"summary": report, "by_year": by_year, "gaps": gaps_df, "extremes": extremes_df}


def recommend_model_start(by_year: pd.DataFrame, max_zero_volume_pct: float = 15.0) -> str:
    """First year after which every year has few zero-volume (no-trade) minutes.

    Minutes without trades repeat the previous price, which pushes realised
    volatility towards zero. Such years would bias the models.
    """
    full_years = by_year[by_year["expected_minutes"] > 300 * 1440]
    ok = full_years["zero_volume_share_pct"] <= max_zero_volume_pct
    for i, year in enumerate(full_years["year"]):
        if ok.iloc[i:].all():
            return f"{int(year)}-01-01"
    return config.MODEL_START


def clean_minute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules derived from the audit.

    1. drop rows with missing values;
    2. drop duplicate timestamps (keep the last record);
    3. drop rows with non-positive prices;
    4. repair candles whose high/low do not contain open/close;
    5. replace isolated one-minute spikes (bad ticks) by the previous close.
    Zero-volume minutes are kept: they are real minutes without trades.
    """
    log = {"rows_in": len(df)}
    df = df.dropna()
    log["dropped_nan"] = log["rows_in"] - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="ts", keep="last")
    log["dropped_duplicates"] = before - len(df)

    before = len(df)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    log["dropped_non_positive"] = before - len(df)

    prices = df[["open", "high", "low", "close"]]
    true_high, true_low = prices.max(axis=1), prices.min(axis=1)
    repaired = (df["high"] != true_high) | (df["low"] != true_low)
    df = df.assign(high=true_high, low=true_low)
    log["repaired_candles"] = int(repaired.sum())

    spikes = detect_spikes(df["close"])
    if spikes.any():
        previous_close = df["close"].shift(1)
        df = df.assign(**{c: df[c].where(~spikes, previous_close) for c in ["open", "high", "low", "close"]})
    log["replaced_spikes"] = int(spikes.sum())
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log
