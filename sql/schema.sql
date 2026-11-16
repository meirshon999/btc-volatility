-- Normalised schema of the Bitcoin volatility database (SQLite).
-- Dimensions: halving, calendar, model. Facts: candle_daily, candle_hourly,
-- market_indicator, experiment, forecast.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS forecast;
DROP TABLE IF EXISTS experiment;
DROP TABLE IF EXISTS model;
DROP TABLE IF EXISTS market_indicator;
DROP TABLE IF EXISTS candle_hourly;
DROP TABLE IF EXISTS candle_daily;
DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS halving;

-- Bitcoin halving epochs: the block reward is cut in half every 210,000 blocks.
CREATE TABLE halving (
    epoch         INTEGER PRIMARY KEY,
    start_date    TEXT    NOT NULL,          -- YYYY-MM-DD
    block_height  INTEGER NOT NULL,
    block_reward  REAL    NOT NULL           -- BTC per block during the epoch
);

-- One row per calendar day (UTC).
CREATE TABLE calendar (
    date        TEXT    PRIMARY KEY,         -- YYYY-MM-DD
    year        INTEGER NOT NULL,
    quarter     INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    iso_week    INTEGER NOT NULL,
    weekday     INTEGER NOT NULL,            -- 0 = Monday ... 6 = Sunday
    is_weekend  INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
    epoch       INTEGER NOT NULL REFERENCES halving (epoch)
);

-- Daily bars with realised volatility (all returns and volatilities in %).
CREATE TABLE candle_daily (
    date               TEXT PRIMARY KEY REFERENCES calendar (date),
    open               REAL NOT NULL,
    high               REAL NOT NULL,
    low                REAL NOT NULL,
    close              REAL NOT NULL,
    volume             REAL NOT NULL,        -- BTC traded
    log_return         REAL,                 -- 100 * ln(close / previous close)
    realized_vol       REAL,                 -- sqrt of the sum of squared 5-minute returns
    parkinson_vol      REAL,                 -- range-based estimator
    zero_volume_share  REAL NOT NULL         -- share of minutes without trades
);

-- Hourly bars, used for intraday patterns.
CREATE TABLE candle_hourly (
    ts      TEXT    PRIMARY KEY,             -- YYYY-MM-DD HH:00:00
    date    TEXT    NOT NULL REFERENCES calendar (date),
    hour    INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  REAL    NOT NULL
);
CREATE INDEX idx_candle_hourly_date ON candle_hourly (date);

-- External market data (FRED), long format: one row per day and series.
CREATE TABLE market_indicator (
    date    TEXT NOT NULL REFERENCES calendar (date),
    series  TEXT NOT NULL,                   -- VIX, SP500, USD_INDEX
    value   REAL NOT NULL,
    PRIMARY KEY (date, series)
);

-- Forecasting approaches that were compared.
CREATE TABLE model (
    model_id     INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL UNIQUE,
    family       TEXT    NOT NULL CHECK (family IN ('baseline', 'econometric', 'ml')),
    description  TEXT
);

-- One row per model and evaluation run (cross-validation fold or hold-out test).
CREATE TABLE experiment (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER NOT NULL REFERENCES model (model_id),
    stage           TEXT    NOT NULL CHECK (stage IN ('cv', 'test')),
    fold            INTEGER,                 -- NULL for the test stage
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    n_days          INTEGER NOT NULL,
    mae             REAL    NOT NULL,
    rmse            REAL    NOT NULL,
    mase            REAL    NOT NULL,
    smape           REAL    NOT NULL,
    qlike           REAL    NOT NULL,
    fit_seconds     REAL    NOT NULL,
    peak_memory_mb  REAL    NOT NULL
);

-- One-step-ahead forecasts on the hold-out period. The actual value is not
-- duplicated here: it is joined from candle_daily.
CREATE TABLE forecast (
    model_id       INTEGER NOT NULL REFERENCES model (model_id),
    date           TEXT    NOT NULL REFERENCES calendar (date),
    predicted_vol  REAL    NOT NULL,
    PRIMARY KEY (model_id, date)
);
