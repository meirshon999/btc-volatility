-- Analytical queries of the project. Each query starts with "-- name:" and
-- "-- description:" lines; src/database.py loads them by name and returns a
-- pandas DataFrame. Parameters :start and :end limit the period (YYYY-MM-DD).

-- name: yearly_summary
-- description: Yearly trading activity and volatility (JOIN, GROUP BY, HAVING)
SELECT c.year,
       COUNT(*)                                  AS days,
       ROUND(AVG(d.realized_vol), 3)             AS avg_daily_vol_pct,
       ROUND(MAX(d.realized_vol), 3)             AS max_daily_vol_pct,
       ROUND(SUM(d.volume), 0)                   AS btc_traded,
       ROUND(100 * AVG(d.zero_volume_share), 2)  AS zero_volume_minutes_pct
FROM candle_daily AS d
JOIN calendar AS c ON c.date = d.date
WHERE d.date BETWEEN :start AND :end
GROUP BY c.year
HAVING COUNT(*) >= 300
ORDER BY c.year;

-- name: halving_epochs
-- description: Price change and volatility in each halving epoch (CTE, FIRST_VALUE / LAST_VALUE window)
WITH epoch_days AS (
    SELECT c.epoch, d.date, d.realized_vol,
           FIRST_VALUE(d.close) OVER w AS first_close,
           LAST_VALUE(d.close)  OVER w AS last_close
    FROM candle_daily AS d
    JOIN calendar AS c ON c.date = d.date
    WHERE d.date BETWEEN :start AND :end
    WINDOW w AS (PARTITION BY c.epoch ORDER BY d.date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
)
SELECT h.epoch,
       h.start_date                                         AS halving_date,
       h.block_reward,
       MIN(e.date)                                          AS first_day_in_data,
       MAX(e.date)                                          AS last_day_in_data,
       COUNT(*)                                             AS days,
       ROUND(AVG(e.realized_vol), 3)                        AS avg_daily_vol_pct,
       ROUND(100.0 * (MAX(e.last_close) / MAX(e.first_close) - 1), 1) AS price_change_pct
FROM epoch_days AS e
JOIN halving AS h ON h.epoch = e.epoch
GROUP BY h.epoch, h.start_date, h.block_reward
ORDER BY h.epoch;

-- name: rolling_volatility
-- description: 30-day rolling volatility and 200-day moving average of the price (window AVG over rows)
WITH rolling AS (
    SELECT date, close, realized_vol,
           AVG(realized_vol) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)  AS vol_30d,
           AVG(close)        OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200
    FROM candle_daily
)
SELECT date,
       ROUND(close, 2)                               AS close,
       ROUND(realized_vol, 4)                        AS realized_vol,
       ROUND(vol_30d, 4)                             AS vol_30d,
       ROUND(vol_30d * 19.1049731745428, 2)          AS vol_30d_annualised_pct,   -- x sqrt(365)
       ROUND(sma_200, 2)                             AS sma_200
FROM rolling
WHERE date BETWEEN :start AND :end
ORDER BY date;

-- name: max_drawdown_by_year
-- description: Deepest fall from the all-time high in each year (running MAX window, ROW_NUMBER)
WITH running AS (
    SELECT d.date, c.year, d.close,
           MAX(d.close) OVER (ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
    FROM candle_daily AS d
    JOIN calendar AS c ON c.date = d.date
), ranked AS (
    SELECT year, date, close, peak,
           100.0 * (close / peak - 1)                                  AS drawdown_pct,
           ROW_NUMBER() OVER (PARTITION BY year ORDER BY close / peak) AS rn
    FROM running
)
SELECT year, date AS trough_date, ROUND(close, 2) AS close,
       ROUND(peak, 2) AS all_time_high_before, ROUND(drawdown_pct, 1) AS drawdown_pct
FROM ranked
WHERE rn = 1
ORDER BY year;

-- name: top_volatile_days
-- description: Three most volatile days of every year (CTE, ROW_NUMBER partitioned by year)
WITH ranked AS (
    SELECT c.year, d.date, d.realized_vol, d.log_return,
           ROW_NUMBER() OVER (PARTITION BY c.year ORDER BY d.realized_vol DESC) AS rank_in_year
    FROM candle_daily AS d
    JOIN calendar AS c ON c.date = d.date
    WHERE d.date BETWEEN :start AND :end
)
SELECT year, rank_in_year, date,
       ROUND(realized_vol, 2) AS realized_vol_pct,
       ROUND(log_return, 2)   AS log_return_pct
FROM ranked
WHERE rank_in_year <= 3
ORDER BY year, rank_in_year;

-- name: intraday_profile
-- description: Average absolute hourly return by weekday and hour, UTC (LAG window, JOIN, GROUP BY)
WITH hourly_returns AS (
    SELECT h.date, h.hour, h.volume,
           100.0 * ABS(h.close / LAG(h.close) OVER (ORDER BY h.ts) - 1) AS abs_return_pct
    FROM candle_hourly AS h
)
SELECT c.weekday,
       r.hour,
       COUNT(*)                         AS hours,
       ROUND(AVG(r.abs_return_pct), 4)  AS avg_abs_return_pct,
       ROUND(AVG(r.volume), 2)          AS avg_btc_volume
FROM hourly_returns AS r
JOIN calendar AS c ON c.date = r.date
WHERE r.date BETWEEN :start AND :end
  AND r.abs_return_pct IS NOT NULL
GROUP BY c.weekday, r.hour
ORDER BY c.weekday, r.hour;

-- name: weekend_effect
-- description: Volatility and volume on weekdays vs weekends by year (CASE, GROUP BY, HAVING)
SELECT c.year,
       CASE WHEN c.is_weekend = 1 THEN 'weekend' ELSE 'weekday' END AS day_type,
       COUNT(*)                        AS days,
       ROUND(AVG(d.realized_vol), 3)   AS avg_vol_pct,
       ROUND(AVG(d.volume), 0)         AS avg_btc_volume
FROM candle_daily AS d
JOIN calendar AS c ON c.date = d.date
WHERE d.date BETWEEN :start AND :end
GROUP BY c.year, day_type
HAVING COUNT(*) >= 50
ORDER BY c.year, day_type;

-- name: turbulent_months
-- description: Months with volatility 50 % above the period average (scalar subquery in HAVING)
SELECT substr(d.date, 1, 7)                                   AS month,
       COUNT(*)                                               AS days,
       ROUND(AVG(d.realized_vol), 3)                          AS avg_vol_pct,
       SUM(CASE WHEN ABS(d.log_return) > 5 THEN 1 ELSE 0 END) AS days_with_move_over_5pct,
       ROUND(MIN(d.close), 0)                                 AS min_close,
       ROUND(MAX(d.close), 0)                                 AS max_close
FROM candle_daily AS d
WHERE d.date BETWEEN :start AND :end
GROUP BY month
HAVING AVG(d.realized_vol) > (SELECT 1.5 * AVG(realized_vol)
                              FROM candle_daily
                              WHERE date BETWEEN :start AND :end)
ORDER BY avg_vol_pct DESC;

-- name: completeness_by_year
-- description: Data completeness check against the calendar (LEFT JOIN, CTE, COALESCE)
WITH hours_per_day AS (
    SELECT date, COUNT(*) AS hours
    FROM candle_hourly
    GROUP BY date
)
SELECT c.year,
       COUNT(c.date)                                                        AS calendar_days,
       COUNT(d.date)                                                        AS days_with_data,
       COALESCE(SUM(h.hours), 0)                                            AS hourly_bars,
       24 * COUNT(d.date)                                                   AS expected_hourly_bars,
       ROUND(100.0 * COALESCE(SUM(h.hours), 0) / (24 * COUNT(d.date)), 2)   AS hourly_completeness_pct
FROM calendar AS c
LEFT JOIN candle_daily  AS d ON d.date = c.date
LEFT JOIN hours_per_day AS h ON h.date = c.date
GROUP BY c.year
HAVING COUNT(d.date) > 0
ORDER BY c.year;

-- name: model_leaderboard
-- description: Hold-out ranking of the models with improvement over the naive forecast (CTE, subquery, RANK)
WITH test AS (
    SELECT m.name, m.family, e.mae, e.rmse, e.mase, e.qlike, e.fit_seconds
    FROM experiment AS e
    JOIN model AS m ON m.model_id = e.model_id
    WHERE e.stage = 'test'
)
SELECT RANK() OVER (ORDER BY t.mae)                                       AS rank,
       t.name                                                             AS model,
       t.family,
       ROUND(t.mae, 4)                                                    AS mae,
       ROUND(t.rmse, 4)                                                   AS rmse,
       ROUND(t.mase, 4)                                                   AS mase,
       ROUND(t.qlike, 4)                                                  AS qlike,
       ROUND(100.0 * (t.mae / (SELECT mae FROM test WHERE name = 'Naive') - 1), 1) AS mae_vs_naive_pct,
       ROUND(t.fit_seconds, 3)                                            AS fit_seconds
FROM test AS t
ORDER BY rank;

-- name: cv_stability
-- description: How stable each model is across cross-validation folds (GROUP BY, HAVING, variance)
SELECT m.name                                                     AS model,
       COUNT(*)                                                   AS folds,
       ROUND(AVG(e.mae), 4)                                       AS mean_mae,
       ROUND(MIN(e.mae), 4)                                       AS best_fold_mae,
       ROUND(MAX(e.mae), 4)                                       AS worst_fold_mae,
       ROUND(AVG(e.mae * e.mae) - AVG(e.mae) * AVG(e.mae), 5)     AS mae_variance
FROM experiment AS e
JOIN model AS m ON m.model_id = e.model_id
WHERE e.stage = 'cv'
GROUP BY m.name
HAVING COUNT(*) >= 3
ORDER BY mean_mae;

-- name: monthly_winners
-- description: Which model had the lowest error in each test month (JOIN of three tables, CTE, RANK by month)
WITH monthly AS (
    SELECT substr(f.date, 1, 7)                         AS month,
           m.name,
           AVG(ABS(f.predicted_vol - d.realized_vol))   AS mae
    FROM forecast AS f
    JOIN candle_daily AS d ON d.date = f.date
    JOIN model AS m ON m.model_id = f.model_id
    GROUP BY month, m.name
), ranked AS (
    SELECT month, name, mae, RANK() OVER (PARTITION BY month ORDER BY mae) AS rnk
    FROM monthly
)
SELECT name AS model, COUNT(*) AS months_won, GROUP_CONCAT(month, ', ') AS months
FROM ranked
WHERE rnk = 1
GROUP BY name
ORDER BY months_won DESC;

-- name: forecast_errors_by_month
-- description: Monthly mean absolute error of every model on the test period (JOIN, GROUP BY)
SELECT substr(f.date, 1, 7)                                   AS month,
       m.name                                                 AS model,
       COUNT(*)                                               AS days,
       ROUND(AVG(ABS(f.predicted_vol - d.realized_vol)), 4)   AS mae
FROM forecast AS f
JOIN candle_daily AS d ON d.date = f.date
JOIN model AS m ON m.model_id = f.model_id
GROUP BY month, m.name
ORDER BY month, mae;

-- name: market_alignment
-- description: Bitcoin volatility next to the VIX and the S&P 500 (LEFT JOIN on a long table)
SELECT d.date,
       d.realized_vol,
       d.log_return,
       v.value AS vix,
       s.value AS sp500
FROM candle_daily AS d
LEFT JOIN market_indicator AS v ON v.date = d.date AND v.series = 'VIX'
LEFT JOIN market_indicator AS s ON s.date = d.date AND s.series = 'SP500'
WHERE d.date BETWEEN :start AND :end
ORDER BY d.date;

-- name: next_day_forecast
-- description: Forecasts for the day after the data end (anti-join: no actual value yet)
SELECT f.date, m.name AS model, m.family, ROUND(f.predicted_vol, 3) AS predicted_vol_pct
FROM forecast AS f
JOIN model AS m ON m.model_id = f.model_id
LEFT JOIN candle_daily AS d ON d.date = f.date
WHERE d.date IS NULL
ORDER BY f.predicted_vol;
