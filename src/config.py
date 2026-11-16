"""Project-wide paths and constants.

Every module imports its settings from here, so changing a path or a date
split in one place changes it for the whole pipeline.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- data -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV_NAME = "btcusd_1-min_data.csv"
MINUTE_PARQUET_DIR = DATA_DIR / "parquet" / "minute"   # partitioned by year
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
DB_PATH = DATA_DIR / "btc.db"

HOURLY_PATH = PROCESSED_DIR / "hourly.parquet"
DAILY_PATH = PROCESSED_DIR / "daily.parquet"

DATASET_URL = "https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data"

# --- outputs --------------------------------------------------------------
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"
STATE_PATH = REPORTS_DIR / "pipeline_state.json"

SQL_DIR = ROOT / "sql"
SCHEMA_SQL = SQL_DIR / "schema.sql"
QUERIES_SQL = SQL_DIR / "queries.sql"

# --- processing -----------------------------------------------------------
CHUNKSIZE = 1_000_000          # rows per CSV chunk
RV_SAMPLING = "5min"           # sampling frequency for realised volatility
SEASON_LENGTH = 7              # weekly seasonality of daily volatility

# --- fixed end of the data ---------------------------------------------------
# The Kaggle file is updated every day. Cutting it at the last complete day used in
# the report makes every number reproducible whenever the file is downloaded.
# Set to None to analyse everything the downloaded file contains.
DATA_END = "2026-11-15"

# --- modelling period (justified by the data-quality audit) ---------------
MODEL_START = "2017-01-01"     # liquid market: few zero-volume minutes
TEST_START = "2025-01-01"      # chronological hold-out, never used for fitting
CV_SPLITS = 5                  # TimeSeriesSplit folds inside the training period

# --- Bitcoin halvings (block reward cut in half) -------------------------
# epoch 0 starts with the genesis block; each following epoch starts at a halving.
HALVINGS = [
    {"epoch": 0, "start_date": "2009-01-03", "block_height": 0, "block_reward": 50.0},
    {"epoch": 1, "start_date": "2012-11-28", "block_height": 210_000, "block_reward": 25.0},
    {"epoch": 2, "start_date": "2016-07-09", "block_height": 420_000, "block_reward": 12.5},
    {"epoch": 3, "start_date": "2020-05-11", "block_height": 630_000, "block_reward": 6.25},
    {"epoch": 4, "start_date": "2024-04-20", "block_height": 840_000, "block_reward": 3.125},
]

# --- well-known market shocks, used only as chart annotations -------------
# Dates checked against the data: each day is in the top 17 % of daily volatility
# since 2017 (most in the top 3 %).
MARKET_EVENTS = [
    {"date": "2020-03-12", "label": "COVID crash"},
    {"date": "2021-05-19", "label": "China crypto crackdown crash"},
    {"date": "2022-05-12", "label": "Terra / LUNA collapse"},
    {"date": "2022-11-08", "label": "FTX collapse"},
    {"date": "2024-01-10", "label": "US spot Bitcoin ETFs approved"},
]

# --- forecast interval -----------------------------------------------------
INTERVAL_QUANTILES = (0.1, 0.9)    # 80 % prediction interval from LightGBM quantile models

# --- external daily series from FRED (optional, downloaded at runtime) ----
FRED_SERIES = {
    "VIX": "VIXCLS",           # CBOE volatility index
    "SP500": "SP500",          # S&P 500 index level
    "USD_INDEX": "DTWEXBGS",   # broad U.S. dollar index
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (RAW_DIR, MINUTE_PARQUET_DIR, PROCESSED_DIR, EXTERNAL_DIR,
                 FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
