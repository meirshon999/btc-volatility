"""SQLite database: schema, loading the tables and running the named queries."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from src import config
from src.external import load_external
from src.features import halving_epoch, load_daily, load_hourly
from src.models import default_models
from src.utils import require

DEFAULT_PARAMS = {"start": config.MODEL_START, "end": "9999-12-31"}


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build_calendar(first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Calendar dimension; one extra day at the end holds the next-day forecast."""
    dates = pd.date_range(first, last + pd.Timedelta(days=1), freq="D")
    iso = dates.isocalendar()
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "year": dates.year, "quarter": dates.quarter, "month": dates.month,
        "iso_week": iso["week"].to_numpy(), "weekday": dates.day_of_week,
        "is_weekend": (dates.day_of_week >= 5).astype(int),
        "epoch": halving_epoch(dates),
    })


def _daily_rows(daily: pd.DataFrame) -> pd.DataFrame:
    rows = daily[["open", "high", "low", "close", "volume", "log_return",
                  "realized_vol", "parkinson_vol", "zero_volume_share"]].copy()
    rows.insert(0, "date", daily.index.strftime("%Y-%m-%d"))
    return rows


def _hourly_rows(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = hourly[["open", "high", "low", "close", "volume"]].copy()
    rows.insert(0, "hour", hourly.index.hour)
    rows.insert(0, "date", hourly.index.strftime("%Y-%m-%d"))
    rows.insert(0, "ts", hourly.index.strftime("%Y-%m-%d %H:%M:%S"))
    return rows


def _market_rows(external: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return pd.DataFrame(columns=["date", "series", "value"])
    long = (external.rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="series", value_name="value").dropna())
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    return long[long["date"].isin(calendar["date"])]


def build_database(verbose: bool = True) -> dict[str, int]:
    """Create the schema and load candles, calendar and market data."""
    daily, hourly = load_daily(), load_hourly()
    calendar = build_calendar(daily.index[0], daily.index[-1])
    halving = pd.DataFrame(config.HALVINGS)

    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(config.SCHEMA_SQL.read_text(encoding="utf-8"))
        tables = {"halving": halving, "calendar": calendar,
                  "candle_daily": _daily_rows(daily), "candle_hourly": _hourly_rows(hourly),
                  "market_indicator": _market_rows(load_external(), calendar)}
        for name, frame in tables.items():
            frame.to_sql(name, conn, if_exists="append", index=False, chunksize=20_000)
            if verbose:
                print(f"  {name:<17} {len(frame):>8,} rows")
    if (config.TABLES_DIR / "test_metrics.csv").exists():
        store_results(verbose=verbose)
    return table_counts()


def store_results(verbose: bool = True) -> None:
    """Load model descriptions, experiment metrics and forecasts into the database."""
    cv = pd.read_csv(require(config.TABLES_DIR / "cv_folds.csv", "8 - train models"))
    test = pd.read_csv(config.TABLES_DIR / "test_metrics.csv")
    predictions = pd.read_csv(config.TABLES_DIR / "test_predictions.csv", parse_dates=["date"])
    next_day = pd.read_csv(config.TABLES_DIR / "next_day_forecast.csv")

    models = pd.DataFrame([{"model_id": i, "name": m.name, "family": m.family,
                            "description": m.description}
                           for i, m in enumerate(default_models(), start=1)])
    ids = dict(zip(models["name"], models["model_id"]))

    test_days = predictions["date"]
    cv_rows = cv.assign(stage="cv", period_start=cv["val_start"], period_end=cv["val_end"])
    test_rows = test.assign(stage="test", fold=None,
                            period_start=test_days.min().strftime("%Y-%m-%d"),
                            period_end=test_days.max().strftime("%Y-%m-%d"))
    experiment = pd.concat([cv_rows, test_rows], ignore_index=True)
    experiment["model_id"] = experiment["model"].map(ids)
    experiment = experiment.rename(columns={"n": "n_days"})[
        ["model_id", "stage", "fold", "period_start", "period_end", "n_days",
         "mae", "rmse", "mase", "smape", "qlike", "fit_seconds", "peak_memory_mb"]]

    forecast = (predictions.drop(columns="actual")
                .melt(id_vars="date", var_name="model", value_name="predicted_vol").dropna())
    forecast["date"] = forecast["date"].dt.strftime("%Y-%m-%d")
    upcoming = next_day.rename(columns={"forecast_vol": "predicted_vol"})
    forecast = pd.concat([forecast, upcoming[["date", "model", "predicted_vol"]]], ignore_index=True)
    forecast["model_id"] = forecast["model"].map(ids)

    with connect() as conn:
        for table in ("forecast", "experiment", "model"):
            conn.execute(f"DELETE FROM {table}")
        models.to_sql("model", conn, if_exists="append", index=False)
        experiment.to_sql("experiment", conn, if_exists="append", index=False)
        forecast[["model_id", "date", "predicted_vol"]].to_sql(
            "forecast", conn, if_exists="append", index=False, chunksize=20_000)
    if verbose:
        print(f"  model {len(models)} rows, experiment {len(experiment)} rows, "
              f"forecast {len(forecast)} rows")


def table_counts() -> dict[str, int]:
    require(config.DB_PATH, "5 - build the SQLite database")
    with connect() as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


# --- named queries ----------------------------------------------------------

def load_queries(path: Path = config.QUERIES_SQL) -> dict[str, dict[str, str]]:
    """Split queries.sql into {name: {"description": ..., "sql": ...}}."""
    text = path.read_text(encoding="utf-8")
    queries = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.MULTILINE)[1:]:
        name, _, body = block.partition("\n")
        description = ""
        match = re.match(r"-- description:\s*(.*)\n", body)
        if match:
            description = match.group(1).strip()
            body = body[match.end():]
        queries[name.strip()] = {"description": description, "sql": body.strip()}
    return queries


def run_query(name: str, params: dict | None = None) -> pd.DataFrame:
    """Run a named query and return the result as a DataFrame."""
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"unknown query {name!r}; available: {', '.join(queries)}")
    require(config.DB_PATH, "5 - build the SQLite database")
    with connect() as conn:
        return pd.read_sql_query(queries[name]["sql"], conn, params={**DEFAULT_PARAMS, **(params or {})})


def run_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a read-only SELECT on a read-only connection (dashboard SQL tab, menu look-ups)."""
    if not re.match(r"^\s*(WITH|SELECT)\b", sql, flags=re.IGNORECASE):
        raise ValueError("only SELECT / WITH queries are allowed")
    uri = f"file:{config.DB_PATH.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=params or {})
