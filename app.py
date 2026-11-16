"""Bitcoin Volatility Lab - interactive dashboard.

Run after the pipeline (python main.py --all):
    streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, config
from src.database import load_queries, run_query, run_sql
from src.eda import correlation_matrix, rolling_vix_correlation
from src.evaluate import naive_scale, score
from src.external import load_external
from src.style import MODEL_ORDER
from src.utils import load_json

st.set_page_config(page_title="Bitcoin Volatility Lab", page_icon=":material/monitoring:",
                   layout="wide")

GLOSSARY = """
**Realised volatility** - how much the price moved during one day: the square root of the
sum of squared 5-minute log returns, in % per day. 3 % means a typical move of about ±3 %.

**Annualised volatility** - daily volatility × √365, the usual way to quote volatility.

**Halving** - every ~4 years the Bitcoin block reward is cut in half (2012, 2016, 2020, 2024).

**Naive forecast** - "tomorrow equals today". Every model has to beat it.

**HAR-RV** - linear regression on yesterday's, last week's and last month's volatility.

**GARCH(1,1)** - the classic econometric volatility model; it uses daily returns only.

**LightGBM** - gradient-boosted decision trees trained on 19 features.

**MAE** - mean absolute error of the forecast, in percentage points.

**MASE** - MAE divided by the naive forecast's MAE; below 1 = better than naive.

**QLIKE** - an error measure that punishes under-predicting risk more than over-predicting.

**Diebold-Mariano test** - checks whether one model's errors are significantly smaller.

**80 % range** - the forecast interval: reality should fall inside on about 8 days out of 10.
"""

HOW_IT_WORKS = """
1. **Data.** 7.8 million one-minute BTC/USD candles (2012-2026) are read in chunks, compressed
   to Parquet and checked for gaps, duplicates and bad ticks.
2. **Measure.** For every day we compute *realised volatility* from 288 five-minute returns:
   a precise, model-free measure of how turbulent the day was.
3. **Forecast.** Seven approaches predict tomorrow's volatility using only information
   available today: four simple baselines, HAR-RV, GARCH and LightGBM.
4. **Check honestly.** Models are compared on 2025-2026, a period they never saw during
   training, with significance tests; everything is stored in SQLite and shown here.
"""

PRESETS = ["1 year", "3 years", "Since 2017", "All", "Custom"]


def max_drawdown_pct(close: pd.Series) -> float:
    """Deepest fall from a previous peak inside the series, in %."""
    return float(100 * (close / close.cummax() - 1).min())


COLUMN_LABELS = {
    "model": "Model", "family": "Type", "n": "Days", "mae": "MAE", "rmse": "RMSE", "mase": "MASE",
    "smape": "sMAPE, %", "qlike": "QLIKE", "mae_vs_naive_pct": "Error vs naive, %",
    "fit_seconds": "Training, s", "predict_seconds": "Prediction, s", "peak_memory_mb": "Peak memory, MB",
    "model_size_kb": "Model size, KB", "rank": "Rank", "year": "Year", "days": "Days",
    "avg_daily_vol_pct": "Avg volatility, %", "max_daily_vol_pct": "Max volatility, %",
    "btc_traded": "BTC traded", "zero_volume_minutes_pct": "Minutes without trades, %",
    "epoch": "Epoch", "halving_date": "Halving date", "block_reward": "Reward, BTC per block",
    "first_day_in_data": "First day", "last_day_in_data": "Last day", "price_change_pct": "Price change, %",
    "rank_in_year": "Rank in year", "date": "Date", "realized_vol_pct": "Volatility, %",
    "log_return_pct": "Return, %", "predicted_vol_pct": "Forecast, %", "reference": "Compared with",
    "loss": "Error measure", "verdict": "Verdict", "dm_statistic": "DM statistic", "p_value": "p-value",
    "mae_mean": "MAE, mean", "mae_std": "MAE, std", "rmse_mean": "RMSE, mean", "rmse_std": "RMSE, std",
    "mase_mean": "MASE, mean", "mase_std": "MASE, std", "qlike_mean": "QLIKE, mean", "qlike_std": "QLIKE, std",
    "fit_seconds_mean": "Training, s", "fit_seconds_std": "Training std, s", "approach": "Approach",
    "seconds": "Time, s", "storage_mb": "On disk, MB", "speedup_vs_baseline": "Speed-up vs CSV, ×",
    "task": "Task", "pandas_seconds": "pandas, s", "sql_seconds": "SQLite, s",
    "sql_slower_x": "SQLite slower, ×", "results_match": "Same result?", "coverage_pct": "Coverage, %",
    "zero_volume_share_pct": "Minutes without trades, %", "flat_candle_share_pct": "Flat candles, %",
    "spikes": "Bad ticks", "rows": "Rows", "actual": "Actual, %", "error": "Error, pp",
    "abs_error": "Abs. error, pp", "log_return": "Return, %",
}

QUALITY_LABELS = {
    "first_timestamp": "First minute", "last_timestamp": "Last minute", "rows": "Minutes (rows)",
    "expected_minutes": "Expected minutes", "missing_minutes": "Missing minutes",
    "coverage_pct": "Coverage, %", "duplicate_timestamps": "Duplicate timestamps", "nan_values": "Empty values",
    "non_positive_prices": "Zero or negative prices", "invalid_candles": "Impossible candles (e.g. high < low)",
    "extreme_moves": "One-minute moves over 10 %", "spikes": "Bad ticks (jump and immediate reversal)",
    "recommended_model_start": "Recommended start of modelling", "configured_model_start": "Start of modelling used",
}


def nice_table(df: pd.DataFrame, index: bool = False) -> None:
    """A table with readable column names and sensible number formats."""
    shown = df.rename(columns=lambda c: COLUMN_LABELS.get(c, str(c).replace("_", " ")))
    formats = {}
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            name = str(column)
            if "p-value" in name:
                fmt = "%.4f"
            elif "BTC" in name:
                fmt = "localized"
            elif "%" in name or "MB" in name or "KB" in name:
                fmt = "%.1f"
            else:
                fmt = "%.3f"
            formats[column] = st.column_config.NumberColumn(column, format=fmt)
    st.dataframe(shown, hide_index=not index, column_config=formats)


def with_verdict(dm: pd.DataFrame) -> pd.DataFrame:
    """Plain-language reading of the Diebold-Mariano tests."""
    verdict = np.where(dm["p_value"] >= 0.05, "no significant difference",
                       np.where(dm["dm_statistic"] < 0, "significantly more accurate",
                                "significantly less accurate"))
    return dm.assign(verdict=verdict)[["model", "reference", "loss", "verdict", "p_value"]]


def explain(how: str, takeaway: str | None = None) -> None:
    """One line on how to read a chart, one line on what it shows."""
    text = f"**How to read:** {how}"
    if takeaway:
        text += f"  \n**Takeaway:** {takeaway}"
    st.caption(text)


# --- data (cached; the cache refreshes when the pipeline rewrites the files) ---------------
# `version` is the file's modification time: it is part of the cache key, so new results
# are picked up automatically. (Streamlit ignores arguments whose name starts with "_".)

def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading data...")
def load_daily(version: float) -> pd.DataFrame:
    return pd.read_parquet(config.DAILY_PATH)


@st.cache_data(show_spinner=False)
def load_results(version: float) -> dict:
    tables = config.TABLES_DIR
    if not (tables / "test_metrics.csv").exists():
        return {}
    results = {
        "metrics": pd.read_csv(tables / "test_metrics.csv"),
        "cv": pd.read_csv(tables / "cv_summary.csv"),
        "cv_folds": pd.read_csv(tables / "cv_folds.csv"),
        "dm": pd.read_csv(tables / "dm_tests.csv"),
        "predictions": pd.read_csv(tables / "test_predictions.csv", index_col="date", parse_dates=["date"]),
        "by_month": pd.read_csv(tables / "errors_by_month.csv", index_col=0),
        "by_regime": pd.read_csv(tables / "errors_by_regime.csv", index_col=0),
        "worst": pd.read_csv(tables / "errors_worst_days.csv"),
        "importance": pd.read_csv(tables / "lgbm_feature_importance.csv", index_col=0).iloc[:, 0],
        "next_day": pd.read_csv(tables / "next_day_forecast.csv"),
        "summary": load_json(config.REPORTS_DIR / "experiment.json"),
        "intervals": None, "stl": None,
    }
    if (tables / "test_intervals.csv").exists():
        results["intervals"] = pd.read_csv(tables / "test_intervals.csv", index_col="date", parse_dates=["date"])
    if (config.REPORTS_DIR / "stl_summary.json").exists():
        results["stl"] = load_json(config.REPORTS_DIR / "stl_summary.json")
    return results


@st.cache_data(show_spinner=False)
def cached_query(name: str, start: str, end: str, version: float) -> pd.DataFrame:
    return run_query(name, {"start": start, "end": end})


@st.cache_data(show_spinner=False)
def load_market(version: float) -> pd.DataFrame:
    return load_external()


if not config.DAILY_PATH.exists() or not config.DB_PATH.exists():
    st.error("No processed data yet. Run the pipeline first:  `python main.py --all`")
    st.stop()

daily_all = load_daily(_mtime(config.DAILY_PATH))
results = load_results(_mtime(config.TABLES_DIR / "test_metrics.csv"))
db_version = _mtime(config.DB_PATH)
dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
first_day, last_day = daily_all.index[0].date(), daily_all.index[-1].date()


# --- filters ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    preset = st.segmented_control("Period", PRESETS, default="Since 2017", required=True)
    if preset == "Custom":
        period = st.slider("Custom period", min_value=first_day, max_value=last_day,
                           value=(date.fromisoformat(config.MODEL_START), last_day), format="YYYY-MM-DD")
    else:
        offsets = {"1 year": pd.Timedelta(days=364), "3 years": pd.Timedelta(days=3 * 365 - 1)}
        if preset in offsets:
            period = ((pd.Timestamp(last_day) - offsets[preset]).date(), last_day)
        elif preset == "Since 2017":
            period = (date.fromisoformat(config.MODEL_START), last_day)
        else:
            period = (first_day, last_day)
    all_epochs = sorted(int(e) for e in daily_all["epoch"].unique())
    epochs = st.segmented_control(
        "Halving epochs", all_epochs, selection_mode="multi", default=all_epochs,
        help="Epoch k starts at the k-th halving: " + ", ".join(
            f"{h['epoch']} = {h['start_date'][:4]}" for h in config.HALVINGS))
    log_scale = st.segmented_control("Price scale", ["log", "linear"], default="log") == "log"
    window = st.select_slider("Rolling window, days", options=[7, 14, 30, 60, 90], value=30)
    chosen_models = st.multiselect("Models on the forecast charts", MODEL_ORDER,
                                   default=["LightGBM", "HAR-RV", "Naive"])
    st.caption("Filters apply to every tab. Times are UTC.")

start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
daily = daily_all.loc[start:end]
daily = daily[daily["epoch"].isin(epochs)]
if daily.empty:
    st.warning("No days match the filters.")
    st.stop()
start_s, end_s = daily.index[0].strftime("%Y-%m-%d"), daily.index[-1].strftime("%Y-%m-%d")


# --- header ----------------------------------------------------------------------------------------

title_col, glossary_col = st.columns([3, 1], vertical_alignment="bottom")
with title_col:
    st.title("Bitcoin Volatility Lab")
    st.caption(f"BTC/USD one-minute candles, {first_day:%d %b %Y} – {last_day:%d %b %Y} · "
               f"selected {start_s} – {end_s} ({len(daily):,} days)")
with glossary_col:
    with st.popover("Glossary", icon=":material/menu_book:", width="stretch"):
        st.markdown(GLOSSARY)

tabs = st.tabs([":material/home: Overview", ":material/candlestick_chart: Market", ":material/waves: Volatility",
                ":material/query_stats: Forecasts", ":material/leaderboard: Models",
                ":material/database: SQL", ":material/fact_check: Data"])


# --- 0. overview -------------------------------------------------------------------------------------

with tabs[0]:
    if not results:
        st.info("Train the models first: `python main.py --step 8`")
    else:
        summary = results["summary"]
        best = summary["best_model"]
        metrics = results["metrics"].set_index("model")
        upcoming = results["next_day"].set_index("model").loc[best]
        price = daily_all["close"].iloc[-1]
        vol = float(upcoming["forecast_vol"])
        next_date = pd.Timestamp(upcoming["date"])

        if not st.session_state.get("intro_closed"):
            with st.container(border=True):
                text_col, button_col = st.columns([7, 1], vertical_alignment="center")
                text_col.markdown(
                    ":material/waving_hand: **Start here.** This dashboard answers one question: "
                    "*how much will Bitcoin's price swing tomorrow?*  \n"
                    "① Read the forecast below · ② choose a period on the left — it applies to every tab · "
                    "③ explore the tabs from left to right · an unknown word? Open **Glossary** at the top right.")
                button_col.button("Got it", on_click=lambda: st.session_state.update(intro_closed=True),
                                  width="stretch")

        st.markdown(f"##### Tomorrow, {next_date:%A %d %B %Y}")
        # word joiner / no-break spaces keep "±$2,325" and "3.6 %" on one line
        st.markdown(f"## Expected move tomorrow: about **±⁠{vol:.1f} %**, "
                    f"≈ **±⁠${price * vol / 100:,.0f}**")
        interval = summary.get("lgbm_interval")
        if interval and pd.notna(upcoming.get("lower")):
            lo, hi = float(upcoming["lower"]), float(upcoming["upper"])
            st.caption(f"{best} forecast of tomorrow's realised volatility, i.e. the typical size of the day's "
                       f"price swing (one standard deviation), at a price of ${price:,.0f}. "
                       f"80 % range: {lo:.1f} – {hi:.1f} % (±${price * lo / 100:,.0f} – ${price * hi / 100:,.0f}); "
                       f"in the test period reality fell inside this range on {interval['coverage_pct']:.0f} % of days.")

        # where today's volatility sits in its own history
        annual = daily_all["realized_vol"].rolling(30).mean().loc[config.MODEL_START:].dropna() * np.sqrt(365)
        low_t, high_t = annual.quantile([1 / 3, 2 / 3])
        current = annual.iloc[-1]
        share_below = 100 * (annual < current).mean()
        mood = "calm" if current < low_t else "stormy" if current > high_t else "normal"
        gauge_col, mood_col, price_col = st.columns([2.2, 1, 1])
        with gauge_col:
            st.markdown("**Market mood: 30-day volatility, % a year**")
            st.plotly_chart(charts.regime_gauge(current, low_t, high_t, float(annual.quantile(0.95)), dark),
                            theme=None, config={"displayModeBar": False, "staticPlot": True})
            st.caption(f"Today: {current:.0f} % a year, higher than on {share_below:.0f} % of days since 2017. "
                       f"Bands are the lower, middle and upper thirds of that history.")
        mood_col.metric("Market mood", mood.capitalize(), border=True,
                        help="calm / normal / stormy = lower / middle / upper third of 30-day volatility since 2017")
        close_before = daily_all["close"].iloc[-31]
        price_col.metric("Last close", f"${price:,.0f}", f"{100 * (price / close_before - 1):+.1f} %",
                         help="BTC/USD close of the last full day; change over the last 30 days",
                         border=True, chart_data=daily_all["close"].iloc[-90:].round(0).tolist(), chart_type="area")

        st.markdown("#### What we found")
        cards = st.columns(4)
        test_days = summary["test_days"]
        cards[0].metric("Error vs naive", f"{metrics.loc[best, 'mae_vs_naive_pct']:.0f} %", border=True,
                        help=f"Mean absolute error of {best} compared with the forecast 'tomorrow = today'")
        cards[0].caption(f"{best}'s error is this much smaller than just repeating yesterday's "
                         f"volatility, on {test_days} days the model never saw.")
        stl = results.get("stl")
        if stl:
            saturday = 100 * (np.exp(stl["weekday_seasonal_effect_log"]["Sat"]) - 1)
            cards[1].metric("Saturday effect", f"{saturday:.0f} %", border=True)
            cards[1].caption("volatility on Saturdays vs an average day: weekends are calmer "
                             "(STL decomposition).")
        yearly = daily_all["realized_vol"].loc[config.MODEL_START:].groupby(lambda d: d.year).mean()
        cards[2].metric(f"{yearly.index[-1]} vs {yearly.index[0]}",
                        f"{100 * (yearly.iloc[-1] / yearly.iloc[0] - 1):.0f} %", border=True)
        cards[2].caption(f"average daily volatility fell from {yearly.iloc[0]:.1f} % in {yearly.index[0]} "
                         f"to {yearly.iloc[-1]:.1f} % in {yearly.index[-1]}.")
        profile = cached_query("intraday_profile", config.MODEL_START, end_s, db_version)
        weekday_hours = profile[profile["weekday"] < 5].groupby("hour")["avg_abs_return_pct"].mean()
        cards[3].metric("Busiest hour, UTC", f"{weekday_hours.idxmax():02d}:00", border=True)
        cards[3].caption("on weekdays prices move most around the U.S. stock-market open.")

        if results["intervals"] is not None:
            recent = results["predictions"].iloc[-60:]
            st.plotly_chart(charts.forecast_chart(
                recent, ["LightGBM"], dark, intervals=results["intervals"], show_days=60, selector=False,
                height=340, title="Last 60 days: forecast, 80 % range and what really happened"), theme=None)
            covered = summary.get("lgbm_interval", {}).get("coverage_pct")
            explain("the blue line is LightGBM's forecast made the day before; the band is its 80 % range; "
                    "the line with dots is the volatility that actually happened.",
                    (f"on {covered:.0f} % of test days reality fell inside the band; " if covered else "")
                    + "sudden shocks still surprise every model.")
        with st.expander("How this project works, in four steps", icon=":material/lightbulb:"):
            st.markdown(HOW_IT_WORKS)


# --- 1. market ---------------------------------------------------------------------------------

with tabs[1]:
    st.plotly_chart(charts.price_chart(daily, log_scale=log_scale, dark=dark), theme=None)
    first_close, last_close = daily["close"].iloc[0], daily["close"].iloc[-1]
    explain("on the log scale each grid step is a 10× change in price; numbered circles are market shocks "
            "(hover for details), vertical lines are halvings.",
            f"from {daily.index[0]:%b %Y} to {daily.index[-1]:%b %Y} the price went from ${first_close:,.0f} "
            f"to ${last_close:,.0f} ({100 * (last_close / first_close - 1):+,.0f} %); the deepest fall from a "
            f"previous peak in this period was {abs(max_drawdown_pct(daily['close'])):.0f} %.")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(charts.return_histogram(daily, dark=dark), theme=None)
        kurtosis = daily["log_return"].kurt()
        explain("bars show how often each daily return occurred; the orange curve is a normal distribution "
                "with the same mean and spread.",
                f"extreme days are far more common than a normal distribution predicts "
                f"(excess kurtosis {kurtosis:.1f}; a normal distribution has 0)." if kurtosis > 1 else
                f"in this period the tails are close to normal (excess kurtosis {kurtosis:.1f}).")
    with right:
        st.subheader("Year by year")
        nice_table(cached_query("yearly_summary", start_s, end_s, db_version))
        st.caption("From the SQL query `yearly_summary`.")
        with st.expander("Halving epochs: price change and volatility between halvings", icon=":material/table:"):
            nice_table(cached_query("halving_epochs", start_s, end_s, db_version))


# --- 2. volatility -------------------------------------------------------------------------------

with tabs[2]:
    st.plotly_chart(charts.volatility_chart(daily, window=window, dark=dark), theme=None)
    persistence = np.log(daily["realized_vol"]).autocorr(1)
    explain(f"gray: volatility of each day; blue: its {window}-day average; circles: market shocks.",
            f"volatility is persistent: each day's level is correlated with the previous day's "
            f"(r = {persistence:.2f} on a log scale), so calm and stormy periods last.")
    years = sorted(daily.index.year.unique())
    year = st.select_slider("Calendar year", options=years, value=years[-1])
    st.plotly_chart(charts.calendar_heatmap(daily, year, dark=dark), theme=None)
    explain("one square per day, weeks run left to right; hover a square for the date and the move.",
            "stormy days tend to come in groups (dark clusters) rather than alone.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.weekday_chart(daily, dark=dark), theme=None)
        weekend = daily.index.day_of_week >= 5
        weekend_diff = 100 * (daily["realized_vol"][weekend].mean() / daily["realized_vol"][~weekend].mean() - 1)
        explain("average volatility for each day of the week; weekends in orange.",
                f"in this period weekends are {abs(weekend_diff):.0f} % "
                f"{'calmer' if weekend_diff < 0 else 'more volatile'} than weekdays.")
    with right:
        profile = cached_query("intraday_profile", start_s, end_s, db_version)
        st.plotly_chart(charts.intraday_heatmap(profile, dark=dark), theme=None)
        busiest = profile[profile["weekday"] < 5].groupby("hour")["avg_abs_return_pct"].mean().idxmax()
        explain("rows are weekdays, columns are hours (UTC); the colour is the average hourly move.",
                f"on weekdays the biggest moves come at {busiest:02d}:00–{busiest + 1:02d}:00 UTC, around the "
                "U.S. stock-market open (13:30 UTC in summer, 14:30 UTC in winter).")
    market = load_market(_mtime(config.EXTERNAL_DIR / "VIX.csv"))
    left, right = st.columns(2)
    with left:
        corr = correlation_matrix(daily, market)
        st.plotly_chart(charts.correlation_heatmap(corr, dark=dark), theme=None)
        with_volume = corr.loc["BTC realised vol", "BTC volume (log)"]
        if "VIX" in corr:
            with_vix = corr.loc["BTC realised vol", "VIX"]
            takeaway = (f"Bitcoin's volatility goes with its own trading volume (ρ = {with_volume:.2f}) far "
                        f"more than with the stock-market VIX (ρ = {with_vix:.2f})."
                        if abs(with_volume) > abs(with_vix) + 0.2 else
                        f"volatility vs volume ρ = {with_volume:.2f}, vs VIX ρ = {with_vix:.2f} in this period.")
        else:
            takeaway = f"volatility vs trading volume: ρ = {with_volume:.2f} (no VIX data downloaded)."
        explain("red = move together, blue = move in opposite directions, gray = no link (Spearman ρ).",
                takeaway)
    with right:
        rolling = rolling_vix_correlation(daily, market)
        if rolling.empty:
            st.info("VIX data not available or the period is shorter than one year.")
        else:
            st.plotly_chart(charts.rolling_correlation_chart(rolling, dark=dark), theme=None)
            explain("correlation with the stock-market 'fear index' over a moving one-year window.",
                    f"the link is unstable: between {rolling.min():.2f} and {rolling.max():.2f} in this period. "
                    "The VIX is only analysed here; it is not a model input.")
    with st.expander("The three most volatile days of each year", icon=":material/table:"):
        nice_table(cached_query("top_volatile_days", start_s, end_s, db_version))


# --- 3. forecasts -----------------------------------------------------------------------------------

with tabs[3]:
    if not results:
        st.info("No model results yet. Run `python main.py --step 8`.")
    else:
        predictions = results["predictions"]
        window_pred = predictions.loc[max(start, predictions.index[0]): end]
        st.caption(f"Hold-out test {predictions.index[0]:%d %b %Y} – {predictions.index[-1]:%d %b %Y}: "
                   "every forecast uses only data up to the previous day.")
        if window_pred.empty:
            st.warning("The selected period does not overlap the test period (2025 onward).")
        else:
            st.plotly_chart(charts.forecast_chart(window_pred, chosen_models or ["LightGBM"], dark=dark,
                                                  intervals=results["intervals"]), theme=None)
            explain("pick models in the sidebar; use the buttons above the chart to zoom out.",
                    "no model can foresee a sudden shock: every forecast rises only after a stormy day.")

            st.subheader("Look at one day")
            day = st.date_input("Day", value=window_pred.index[-1].date(),
                                min_value=window_pred.index[0].date(), max_value=window_pred.index[-1].date())
            day = pd.Timestamp(day)
            history = daily_all["realized_vol"].loc[:day - pd.Timedelta(days=1)]
            inputs = st.columns(4)
            inputs[0].metric("What happened", f"{predictions.loc[day, 'actual']:.2f} %", border=True)
            inputs[1].metric("Yesterday", f"{history.iloc[-1]:.2f} %", border=True)
            inputs[2].metric("Last 5 days, average", f"{history.iloc[-5:].mean():.2f} %", border=True)
            inputs[3].metric("Last 22 days, average", f"{history.iloc[-22:].mean():.2f} %", border=True)
            one_day = pd.DataFrame({"Model": MODEL_ORDER,
                                    "Forecast, %": [predictions.loc[day, m] for m in MODEL_ORDER]})
            one_day["Error, pp"] = one_day["Forecast, %"] - predictions.loc[day, "actual"]
            one_day = one_day.reindex(one_day["Error, pp"].abs().sort_values().index)
            nice_table(one_day)
            explain("these are the kind of inputs the models use (HAR-RV takes yesterday's, the 5-day and the "
                    "22-day average on a log scale); the table ranks the models by how close they came.")

            scale = naive_scale(daily_all["realized_vol"].loc[config.MODEL_START: config.TEST_START])
            live = pd.DataFrame([{"model": m, **score(window_pred["actual"], window_pred[m], scale)}
                                 for m in MODEL_ORDER]).sort_values("mae")
            st.subheader(f"Accuracy on the selected window ({len(window_pred)} days)")
            nice_table(live)
            st.caption("Lower is better everywhere. MASE below 1 means better than the naive forecast.")
        st.subheader("Forecast for the next day")
        nice_table(cached_query("next_day_forecast", start_s, end_s, db_version))


# --- 4. models -------------------------------------------------------------------------------------------

with tabs[4]:
    if not results:
        st.info("No model results yet.")
    else:
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(charts.skill_chart(results["metrics"], dark=dark), theme=None)
            winners = results["metrics"].query("mae_vs_naive_pct < 0")["model"].tolist()
            note = (" Both learn from intraday (5-minute) volatility."
                    if set(winners) <= {"HAR-RV", "LightGBM"} and winners else "")
            explain("how much smaller each model's average error is than the naive forecast's; "
                    "bars to the right are better.",
                    f"models that beat 'tomorrow = today': {', '.join(winners) or 'none'}.{note}")
        with right:
            st.subheader("Leaderboard")
            nice_table(cached_query("model_leaderboard", start_s, end_s, db_version))
            st.caption("Hold-out test 2025–2026; lower is better. From the SQL query `model_leaderboard`.")
        st.plotly_chart(charts.validation_timeline(results["cv_folds"], daily_all.index[-1], dark=dark),
                        theme=None)
        explain("every row is one experiment: models learn on the light part and are scored on the dark part.",
                "the test period (orange) was never used for training or choosing settings.")
        metric = st.segmented_control("Metric", ["mae", "rmse", "mase", "qlike"], default="mae",
                                      format_func=str.upper) or "mae"
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.metric_bars(results["metrics"], metric, dark=dark), theme=None)
        with right:
            st.plotly_chart(charts.monthly_error_chart(results["by_month"], chosen_models or ["LightGBM"],
                                                       dark=dark), theme=None)
        st.subheader("Is the difference real? Significance tests")
        verdicts = with_verdict(results["dm"])
        nice_table(verdicts)
        better = verdicts[(verdicts["reference"] == "Naive")
                          & (verdicts["verdict"] == "significantly more accurate")]
        always_better = better.groupby("model").size().loc[lambda counts: counts == 3].index.tolist()
        explain("each row compares a model with a reference forecast (Diebold–Mariano test); "
                "'significantly' means p < 0.05.",
                f"significantly more accurate than the naive forecast on all three error measures: "
                f"{', '.join(always_better)}." if always_better else None)
        with st.expander("More details for experts: cross-validation, errors by regime, feature importance, "
                         "largest errors", icon=":material/science:"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Cross-validation (5 folds, 2017–2024)**")
                nice_table(results["cv"])
                st.markdown("**Average error by volatility regime (MAE)**")
                nice_table(results["by_regime"].rename_axis("regime").reset_index())
            with right:
                st.markdown("**What LightGBM relies on (share of total gain)**")
                st.dataframe(results["importance"].rename("Share of gain").to_frame().rename_axis("Feature"),
                             column_config={"Share of gain": st.column_config.ProgressColumn(
                                 "Share of gain", format="percent", min_value=0.0,
                                 max_value=float(results["importance"].max()))})
            st.markdown("**The ten largest errors of the best model**")
            nice_table(results["worst"])


# --- 5. SQL explorer ---------------------------------------------------------------------------------------

with tabs[5]:
    queries = load_queries()
    name = st.selectbox("Query", list(queries), format_func=lambda q: f"{q} - {queries[q]['description']}")
    st.code(queries[name]["sql"], language="sql")
    result = cached_query(name, start_s, end_s, db_version)
    st.caption(f"{len(result):,} rows · parameters :start = {start_s}, :end = {end_s}")
    st.dataframe(result, hide_index=True)
    st.download_button("Download CSV", result.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name}.csv", mime="text/csv", icon=":material/download:")
    st.divider()
    st.subheader("Your own query (read-only)")
    custom = st.text_area("SQL", "SELECT c.year, COUNT(*) AS days, ROUND(AVG(d.realized_vol), 2) AS avg_vol\n"
                                 "FROM candle_daily d JOIN calendar c ON c.date = d.date\n"
                                 "GROUP BY c.year ORDER BY c.year;", height=120)
    if st.button("Run query", type="primary", icon=":material/play_arrow:"):
        try:
            st.dataframe(run_sql(custom), hide_index=True)
        except Exception as error:      # show SQL errors to the user instead of crashing
            st.error(f"Query failed: {error}")


# --- 6. data & performance -----------------------------------------------------------------------------------

with tabs[6]:
    pipeline = config.FIGURES_DIR / "15_pipeline.png"
    if pipeline.exists():
        st.image(str(pipeline), width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Data quality check")
        quality_path = config.REPORTS_DIR / "quality_report.json"
        if quality_path.exists():
            q = load_json(quality_path)
            checks = pd.DataFrame({"Check": [QUALITY_LABELS.get(k, k) for k in q],
                                   "Result": [f"{v:,.2f}" if isinstance(v, float) else
                                              f"{v:,}" if isinstance(v, int) else str(v) for v in q.values()]})
            st.dataframe(checks, hide_index=True)
        quality_table = config.TABLES_DIR / "quality_by_year.csv"
        if quality_table.exists():
            with st.expander("Year by year", icon=":material/table:"):
                nice_table(pd.read_csv(quality_table)[["year", "rows", "coverage_pct", "zero_volume_share_pct",
                                                       "flat_candle_share_pct", "spikes"]])
    with right:
        st.subheader("Reading the data: CSV vs Parquet")
        bench = config.TABLES_DIR / "benchmark_storage.csv"
        if bench.exists():
            nice_table(pd.read_csv(bench)[["approach", "seconds", "peak_memory_mb", "storage_mb",
                                           "speedup_vs_baseline"]])
            st.caption("Same 7.8 M rows each time; best of 3 runs; memory measured in a separate process.")
        st.subheader("The same calculation in pandas and in SQLite")
        engines = config.TABLES_DIR / "benchmark_engines.csv"
        if engines.exists():
            nice_table(pd.read_csv(engines)[["task", "pandas_seconds", "sql_seconds", "sql_slower_x",
                                             "results_match"]])
    er = config.FIGURES_DIR / "00_er_diagram.png"
    if er.exists():
        st.subheader("Database schema")
        st.image(str(er))

st.caption("Research project for a university course. Not investment advice.")
