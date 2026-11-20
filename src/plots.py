"""Static figures for the report (matplotlib, PNG in reports/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")       # render to files, no window needed

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src import config
from src.style import BLUE_RAMP, DIVERGING, LIGHT, PALETTE, model_color, use_matplotlib_style

SOURCE = "Data: Bitstamp BTC/USD 1-minute candles (Kaggle, mczielinski); FRED for VIX / S&P 500."
C = LIGHT


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=C["text_secondary"],
            va="bottom", ha="left")


def _source(fig, text: str = SOURCE, y: float = -0.02) -> None:
    fig.text(0.0, y, text, fontsize=7.5, color=C["muted"], ha="left", va="top")


def _year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _halvings(ax, start: pd.Timestamp, end: pd.Timestamp, label: bool = True) -> None:
    for h in config.HALVINGS[1:]:
        day = pd.Timestamp(h["start_date"])
        if start <= day <= end:
            ax.axvline(day, color=C["axis"], linewidth=0.9, zorder=1)
            if label:
                ax.text(day, 1.0, f" halving {h['epoch']}", transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color=C["muted"], va="top", ha="left")


def _events(ax, series: pd.Series) -> str:
    """Numbered markers at well-known market shocks; returns the key for the caption."""
    keys = []
    for i, event in enumerate(config.MARKET_EVENTS, start=1):
        day = pd.Timestamp(event["date"])
        if day not in series.index or pd.isna(series.loc[day]):
            continue
        y = series.loc[day]
        ax.scatter(day, y, s=135, facecolor=C["surface"], edgecolor=C["text"], linewidth=1.1, zorder=6)
        ax.text(day, y, str(i), ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=C["text"], zorder=7)
        keys.append(f"{i} {event['label']} ({day:%d %b %Y})")
    return "Events:  " + "   ·   ".join(keys)


def _event_key(fig, text: str, y: float = -0.005) -> None:
    fig.text(0.0, y, text, fontsize=8, color=C["text_secondary"], ha="left", va="top")


# --- 1. price ------------------------------------------------------------------------

def price_history(daily: pd.DataFrame) -> Path:
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(daily.index, daily["close"], color=PALETTE[0], linewidth=1.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Bitcoin price, 2012–2026")
    _subtitle(ax, "Daily close in USD, log scale. Vertical lines: block-reward halvings; "
                  "numbered circles: market shocks.")
    _halvings(ax, daily.index[0], daily.index[-1])
    key = _events(ax, daily["close"])
    last = daily.iloc[-1]
    ax.annotate(f"${last['close']:,.0f}", (daily.index[-1], last["close"]), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=C["text"], va="center")

    monthly = daily["volume"].resample("MS").sum() / 1e3
    axv.bar(monthly.index, monthly.values, width=25, color=BLUE_RAMP[2], linewidth=0)
    axv.set_ylabel("Volume,\nthousand BTC / month")
    axv.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    _halvings(axv, daily.index[0], daily.index[-1], label=False)
    _year_axis(axv)
    _event_key(fig, key, y=-0.01)
    _source(fig, SOURCE, y=-0.045)
    return _save(fig, "01_price_history")


# --- 2. data quality ----------------------------------------------------------------------

def zero_volume_by_year(by_year: pd.DataFrame, threshold: float = 15.0) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    kept = by_year["year"] >= pd.Timestamp(config.MODEL_START).year
    colors = np.where(kept, PALETTE[0], C["neutral"])
    ax.bar(by_year["year"].astype(str), by_year["zero_volume_share_pct"], color=colors,
           width=0.72, linewidth=0)
    ax.axhline(threshold, color=C["text_secondary"], linewidth=0.8)
    ax.text(len(by_year) - 0.5, threshold + 1.5, f"threshold {threshold:.0f} %", ha="right",
            fontsize=8, color=C["text_secondary"])
    for i, row in by_year.reset_index(drop=True).iterrows():
        if row["year"] in (by_year["year"].min(), 2016, 2017, by_year["year"].max()):
            ax.text(i, row["zero_volume_share_pct"] + 1.2, f"{row['zero_volume_share_pct']:.1f} %",
                    ha="center", fontsize=8, color=C["text"])
    ax.set_ylabel("% of minutes with zero volume")
    ax.set_title("Minutes without any trade, by year")
    excluded = by_year.loc[~kept, "zero_volume_share_pct"]
    _subtitle(ax, f"Gray: {excluded.min():.0f}–{excluded.max():.0f} % of minutes without a trade (stale prices), "
                  "excluded from modelling. Blue: modelling period (2017 onward).")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    return _save(fig, "02_zero_volume_by_year")


# --- 3. returns -------------------------------------------------------------------------

def return_distribution(daily: pd.DataFrame) -> Path:
    r = daily["log_return"].dropna()
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    clipped = r.clip(-20, 20)
    ax.hist(clipped, bins=120, density=True, color=PALETTE[0], alpha=0.85, linewidth=0)
    grid = np.linspace(-20, 20, 400)
    ax.plot(grid, stats.norm.pdf(grid, r.mean(), r.std()), color=PALETTE[1], linewidth=1.6,
            label="normal distribution, same mean and std")
    ax.set_xlim(-20, 20)
    ax.set_xlabel("Daily log return, %")
    ax.set_ylabel("Density")
    ax.set_title("Daily returns have fat tails")
    _subtitle(ax, f"Excess kurtosis {r.kurt():.1f}, skewness {r.skew():.2f} (2017 onward).")
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)

    years = sorted(r.index.year.unique())
    data = [r[r.index.year == y].to_numpy() for y in years]
    axb.boxplot(data, tick_labels=[str(y) for y in years], widths=0.55, whis=1.5,
                patch_artist=True, showfliers=True,
                boxprops=dict(facecolor=BLUE_RAMP[0], edgecolor=PALETTE[0], linewidth=0.9),
                medianprops=dict(color=PALETTE[0], linewidth=1.6),
                whiskerprops=dict(color=C["axis"]), capprops=dict(color=C["axis"]),
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C["muted"],
                                markeredgewidth=0, alpha=0.6))
    axb.set_ylim(-25, 25)
    axb.set_ylabel("Daily log return, %")
    axb.set_title("Returns by year (box = IQR, whiskers = 1.5 × IQR)")
    _subtitle(axb, "Points beyond the whiskers are IQR outliers; they are kept, because bad ticks were "
                   "already removed at the minute level.")
    return _save(fig, "03_return_distribution")


# --- 4. volatility over time ------------------------------------------------------------

def rolling_volatility(daily: pd.DataFrame) -> Path:
    d = daily.loc[config.MODEL_START:]
    annual = d["realized_vol"].rolling(30).mean() * np.sqrt(365)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axvspan(pd.Timestamp(config.TEST_START), d.index[-1], color=C["grid"], alpha=0.45, linewidth=0)
    ax.text(pd.Timestamp(config.TEST_START), 0.98, "  hold-out test", transform=ax.get_xaxis_transform(),
            fontsize=8, color=C["text_secondary"], va="top")
    ax.plot(annual.index, annual, color=PALETTE[0], linewidth=1.4)
    _halvings(ax, d.index[0], d.index[-1])
    key = _events(ax, annual)
    _event_key(fig, key)
    yearly = d["realized_vol"].groupby(d.index.year).mean()
    first, last = yearly.index[0], yearly.index[-1]
    falling = yearly.iloc[-1] < yearly.iloc[0]
    ax.set_ylabel("Annualised volatility, %")
    ax.set_title(f"Bitcoin volatility has fallen since {first}" if falling
                 else "Bitcoin volatility over time")
    _subtitle(ax, f"30-day average of daily realised volatility (5-minute returns), × √365. "
                  f"Mean daily volatility: {yearly.iloc[0]:.1f} % in {first} vs {yearly.iloc[-1]:.1f} % in {last}.")
    ax.set_ylim(0, None)
    _year_axis(ax)
    return _save(fig, "04_rolling_volatility")


# --- 5. STL --------------------------------------------------------------------------------

def stl_figure(stl_result) -> Path:
    observed, trend = stl_result.observed, stl_result.trend
    seasonal, resid = stl_result.seasonal, stl_result.resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0, 0]
    ax.plot(observed.index, observed, color=C["neutral"], linewidth=0.6, label="log realised volatility")
    ax.plot(trend.index, trend, color=PALETTE[0], linewidth=1.6, label="trend")
    ax.set_title("STL decomposition of log volatility (weekly period)")
    ax.legend(loc="upper right")
    _year_axis(ax)

    ax = axes[1, 0]
    ax.plot(resid.index, resid, color=PALETTE[0], linewidth=0.5)
    ax.axhline(0, color=C["axis"], linewidth=0.8)
    ax.set_title("Remainder")
    _year_axis(ax)

    ax = axes[0, 1]
    last = seasonal.iloc[-56:]
    ax.plot(last.index, last, color=PALETTE[0], linewidth=1.4, marker="o", markersize=3)
    ax.axhline(0, color=C["axis"], linewidth=0.8)
    ax.set_title("Seasonal component, last 8 weeks")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SA, interval=2))

    ax = axes[1, 1]
    effect = 100 * (np.exp(seasonal.groupby(seasonal.index.day_of_week).mean()) - 1)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = [PALETTE[1] if v < 0 else PALETTE[0] for v in effect]
    ax.bar(names, effect, color=colors, width=0.65, linewidth=0)
    ax.axhline(0, color=C["axis"], linewidth=0.8)
    for i, v in enumerate(effect):
        ax.text(i, v + (1 if v >= 0 else -1), f"{v:+.0f} %", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=C["text"])
    ax.set_title("Weekday effect on volatility")
    ax.set_ylim(effect.min() - 6, effect.max() + 6)
    ax.grid(axis="y", visible=False)
    return _save(fig, "05_stl_decomposition")


# --- 6. autocorrelation ----------------------------------------------------------------------

def acf_figure(acf_table: pd.DataFrame, n_obs: int) -> Path:
    band = 1.96 / np.sqrt(n_obs)
    panels = [("acf_return", "ACF of daily returns",
               f"weak memory in the direction of moves (lag 1: {acf_table['acf_return'].iloc[1]:+.2f})"),
              ("acf_squared_return", "ACF of squared returns", "volatility clusters"),
              ("pacf_log_vol", "PACF of log realised volatility", "several lags matter, hence HAR")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    lags = acf_table.index[1:]
    for ax, (column, title, note) in zip(axes, panels):
        values = acf_table[column].iloc[1:]
        ax.axhspan(-band, band, color=C["grid"], alpha=0.6, linewidth=0)
        ax.vlines(lags, 0, values, color=PALETTE[0], linewidth=1.6)
        ax.scatter(lags, values, color=PALETTE[0], s=9, zorder=3)
        ax.axhline(0, color=C["axis"], linewidth=0.8)
        ax.set_title(title)
        _subtitle(ax, note)
        ax.set_xlabel("Lag, days")
    axes[0].set_ylabel("Correlation")
    fig.text(0.0, -0.04, "Shaded band: 95 % confidence interval for zero correlation.",
             fontsize=7.5, color=C["muted"])
    return _save(fig, "06_autocorrelation")


# --- 7. intraday pattern --------------------------------------------------------------------

def intraday_heatmap(profile: pd.DataFrame) -> Path:
    grid = profile.pivot(index="weekday", columns="hour", values="avg_abs_return_pct")
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(12, 3.9))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    peak = np.unravel_index(np.nanargmax(grid.to_numpy()), grid.shape)
    ax.scatter(peak[1], peak[0], s=60, facecolors="none", edgecolors=C["surface"], linewidths=1.6)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=C["text_secondary"])
    bar.set_label("mean |hourly return|, %", fontsize=8.5, color=C["text_secondary"])
    weekday_hours = grid.iloc[:5].mean()
    weekend_level = grid.iloc[5:].to_numpy().mean() / grid.iloc[:5].to_numpy().mean() - 1
    ax.set_title("When Bitcoin moves: mean absolute hourly return by weekday and hour (UTC)")
    peak_hour = weekday_hours.idxmax()
    weekend_text = (f"weekend hours are {abs(weekend_level):.0%} calmer" if weekend_level < 0
                    else f"weekend hours are {weekend_level:.0%} more volatile")
    _subtitle(ax, f"Weekday peak {peak_hour:02d}:00–{peak_hour + 1:02d}:00 UTC, around the U.S. stock-market "
                  f"open (13:30 UTC in summer, 14:30 in winter); {weekend_text}.")
    _source(fig, "Computed in SQL (query intraday_profile) from hourly candles; colour = mean absolute "
                 "hourly return, %.")
    return _save(fig, "07_intraday_heatmap")


# --- 8. correlations --------------------------------------------------------------------------

def correlation_figure(corr: pd.DataFrame, rolling: pd.Series) -> Path:
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = LinearSegmentedColormap.from_list("div", [c for _, c in DIVERGING])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)      # upper triangle and diagonal
    values = np.where(mask, np.nan, corr.to_numpy())
    ax.imshow(values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(corr)):
        for j in range(i):
            v = corr.iat[i, j]
            text = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color=C["surface"] if abs(v) > 0.6 else C["text"])
    ax.set_title("Spearman correlations, 2017 onward")
    _subtitle(ax, "n/a: the stock market has no weekend data.")

    if not rolling.empty:
        axr.axhline(0, color=C["axis"], linewidth=0.8)
        axr.plot(rolling.index, rolling, color=PALETTE[0], linewidth=1.5)
        axr.fill_between(rolling.index, 0, rolling, where=rolling > 0, color=PALETTE[0], alpha=0.12, linewidth=0)
        recent = rolling.loc["2020":]
        axr.set_title("Rolling 1-year correlation: BTC volatility vs VIX")
        _subtitle(axr, f"Unstable link: positive on {(recent > 0).mean():.0%} of days since 2020, "
                       f"peak {rolling.max():.2f} ({rolling.idxmax():%b %Y}), "
                       f"low {rolling.min():.2f} ({rolling.idxmin():%b %Y}).")
        axr.set_ylabel("Pearson correlation of logs")
        _year_axis(axr)
    else:
        axr.set_axis_off()
        axr.text(0.5, 0.5, "External data not downloaded", ha="center", color=C["muted"])
    return _save(fig, "08_correlations")


# --- 9-13. model results -----------------------------------------------------------------------

def forecast_vs_actual(predictions: pd.DataFrame, best: str) -> Path:
    fig, (ax, axz) = plt.subplots(2, 1, figsize=(11.5, 7))
    ax.plot(predictions.index, predictions["actual"], color=C["neutral"], linewidth=0.9, label="actual")
    ax.plot(predictions.index, predictions[best], color=model_color(best), linewidth=1.3, label=best)
    ax.set_title(f"Hold-out test: realised volatility vs {best} forecast")
    _subtitle(ax, "One-step-ahead forecasts for every day of the test period; the model never saw these days.")
    ax.set_ylabel("Daily volatility, %")
    ax.legend(loc="upper right", ncols=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    zoom = predictions.iloc[-90:]
    axz.plot(zoom.index, zoom["actual"], color=C["text"], linewidth=1.4, label="actual",
             marker="o", markersize=2.8)
    for name in [best, "HAR-RV", "Naive"]:
        if name in zoom:
            axz.plot(zoom.index, zoom[name], color=model_color(name), linewidth=1.4, label=name)
    axz.set_title("Last 90 days in detail")
    axz.set_ylabel("Daily volatility, %")
    axz.legend(loc="upper left", ncols=4)
    axz.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    return _save(fig, "09_forecast_vs_actual")


def model_comparison(metrics: pd.DataFrame, cv: pd.DataFrame) -> Path:
    """Test MAE, cross-validation MAE (mean +- std), test QLIKE and training time."""
    order = metrics.sort_values("mae", ascending=False).reset_index(drop=True)
    focus = order["model"].iloc[-1]
    cv_stats = cv.groupby("model")["mae"].agg(["mean", "std"]).reindex(order["model"])
    colors = [model_color(focus) if m == focus else C["neutral"] for m in order["model"]]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)

    def bars(ax, values, fmt, title):
        ax.barh(y, values, color=colors, height=0.62, linewidth=0)
        for yi, v in zip(y, values):
            ax.text(v, yi, "  " + fmt(v), va="center", fontsize=8, color=C["text"])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)

    bars(axes[0], order["mae"].to_numpy(), "{:.3f}".format, "Test MAE")
    axes[0].margins(x=0.22)

    ax = axes[1]
    ax.errorbar(cv_stats["mean"], y, xerr=cv_stats["std"], fmt="none", ecolor=C["axis"],
                elinewidth=1.2, capsize=3)
    ax.scatter(cv_stats["mean"], y, c=colors, s=46, zorder=3, edgecolors=C["surface"], linewidths=1.5)
    for yi, (mean, std) in enumerate(cv_stats.to_numpy()):
        ax.text(mean + std, yi, f"  {mean:.3f}", va="center", fontsize=8, color=C["text"])
    ax.set_title("Cross-validation MAE, mean ± std")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.25)

    bars(axes[2], order["qlike"].to_numpy(), "{:.3f}".format, "Test QLIKE")
    axes[2].margins(x=0.22)

    seconds = order["fit_seconds"].clip(lower=1e-4).to_numpy()
    bars(axes[3], seconds, lambda v: "< 1 ms" if v < 1e-3 else f"{v:.3g} s", "Training time (log scale)")
    axes[3].set_xscale("log")
    axes[3].set_xlim(1e-4, 60)

    axes[0].set_yticks(y, order["model"])
    fig.suptitle("Model comparison: lower is better everywhere", x=0.0, ha="left",
                 fontsize=13, fontweight="semibold")
    _source(fig, "Test: 2025-01-01 .. end of data, never used for fitting. "
                 "Cross-validation: 5 expanding TimeSeriesSplit folds inside 2017-2024.")
    return _save(fig, "10_model_comparison")


def error_by_month(by_month: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = pd.PeriodIndex(by_month.index, freq="M").to_timestamp()
    for name in ["Naive", "HAR-RV", "LightGBM"]:
        if name in by_month:
            ax.plot(x, by_month[name], color=model_color(name), linewidth=1.6, marker="o",
                    markersize=3.2, label=name)
    ax.set_ylabel("MAE, percentage points")
    winners = by_month.idxmin(axis=1).value_counts()
    ax.set_title("Forecast error by month (test period)")
    pairs = by_month.corr().to_numpy()[np.triu_indices(by_month.shape[1], 1)]
    _subtitle(ax, f"{winners.index[0]} had the lowest error in {winners.iloc[0]} of {len(by_month)} months. "
                  f"The models' monthly errors move together (median correlation {np.median(pairs):.2f}).")
    ax.legend(loc="upper right", ncols=3)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    return _save(fig, "11_error_by_month")


def error_by_regime(by_regime: pd.DataFrame, bias: pd.DataFrame, thresholds: pd.DataFrame) -> Path:
    models = [m for m in ["LightGBM", "HAR-RV", "GARCH(1,1)", "Rolling std (30d)", "Naive"] if m in by_regime]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    x = np.arange(len(by_regime.index))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, by_regime[name], width=width * 0.9, color=model_color(name),
               label=name, linewidth=0)
        axb.bar(x + offset, bias[name], width=width * 0.9, color=model_color(name), linewidth=0)
    low, high = thresholds.iloc[0]
    labels = [f"low\n< {low:.1f} %", f"medium\n{low:.1f}–{high:.1f} %", f"high\n> {high:.1f} %"]
    for a in (ax, axb):
        a.set_xticks(x, labels)
        a.grid(axis="x", visible=False)
    ax.set_title("Mean absolute error by volatility regime")
    ax.set_ylabel("MAE, percentage points")
    ax.legend(loc="upper left")
    axb.axhline(0, color=C["axis"], linewidth=0.8)
    axb.set_title("Bias (forecast − actual)")
    calm_worst = bias.loc["low", models].idxmax()
    best_high = by_regime.loc["high", models].idxmin()
    best_overall = by_regime[models].mean().idxmin()
    if best_high == best_overall:
        _subtitle(ax, f"{best_overall} has the lowest error in every regime.")
    else:
        _subtitle(ax, f"On high-volatility days {best_high} is as good as any model: "
                      f"the gain of {best_overall} comes from calm and normal days.")
    _subtitle(axb, f"Forecasts over-shoot calm days ({calm_worst} most) and under-shoot turbulent ones.")
    axb.set_ylabel("Mean error, percentage points")
    return _save(fig, "12_error_by_regime")


def feature_importance(importance: pd.Series, top: int = 12) -> Path:
    top_features = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.barh(top_features.index, 100 * top_features.values, color=PALETTE[0], height=0.6, linewidth=0)
    for i, v in enumerate(top_features.values):
        ax.text(100 * v, i, f"  {100 * v:.1f} %", va="center", fontsize=8)
    ax.set_xlabel("Share of total gain, %")
    ax.set_title("What LightGBM relies on")
    _subtitle(ax, f"{importance.index[0]} ({importance.iloc[0]:.0%}) and {importance.index[1]} "
                  f"({importance.iloc[1]:.0%}) carry most of the signal.")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.15)
    return _save(fig, "13_feature_importance")


# --- ER diagram -----------------------------------------------------------------------------

COLUMN_X = [0.02, 0.36, 0.70]
ROW_TOP = [1.0, 0.62, 0.25]
ER_TABLES = {   # name: (column, row, fields)
    "market_indicator": (0, 0, ["date  PK, FK", "series  PK", "value"]),
    "halving": (1, 0, ["epoch  PK", "start_date", "block_height", "block_reward"]),
    "candle_daily": (2, 0, ["date  PK, FK", "open, high, low, close", "volume, log_return",
                            "realized_vol, parkinson_vol", "zero_volume_share"]),
    "model": (0, 1, ["model_id  PK", "name", "family", "description"]),
    "calendar": (1, 1, ["date  PK", "year, quarter, month", "iso_week, weekday, is_weekend",
                        "epoch  FK"]),
    "candle_hourly": (2, 1, ["ts  PK", "date  FK", "hour", "open, high, low, close, volume"]),
    "experiment": (0, 2, ["run_id  PK", "model_id  FK", "stage, fold, period_start/end",
                          "mae, rmse, mase, smape, qlike", "fit_seconds, peak_memory_mb"]),
    "forecast": (1, 2, ["model_id  PK, FK", "date  PK, FK", "predicted_vol"]),
}
ER_LINKS = [  # (child with the foreign key, referenced parent)
    ("calendar", "halving"), ("candle_daily", "calendar"), ("candle_hourly", "calendar"),
    ("market_indicator", "calendar"), ("experiment", "model"), ("forecast", "model"),
    ("forecast", "calendar"),
]
ER_WIDTH, ER_ROW, ER_HEAD = 0.27, 0.036, 0.05


def _er_box(name: str) -> tuple[float, float, float, float]:
    column, row, fields = ER_TABLES[name]
    height = ER_HEAD + ER_ROW * len(fields) + 0.012
    return COLUMN_X[column], ROW_TOP[row] - height, ER_WIDTH, height


def _er_anchor(box, other) -> tuple[float, float]:
    """Point on the edge of `box` that faces the box `other`."""
    x, y, w, h = box
    ox, oy, ow, oh = other
    if abs(x - ox) < 1e-9:                       # same column: top or bottom edge
        return (x + w / 2, y + h) if oy > y else (x + w / 2, y)
    centre, other_centre = y + h / 2, oy + oh / 2
    share = 0.8 if other_centre > centre + 0.05 else 0.2 if other_centre < centre - 0.05 else 0.5
    return (x + w, y + h * share) if ox > x else (x, y + h * share)


def er_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(13, 8.4))
    ax.set_xlim(0, 0.99)
    ax.set_ylim(-0.08, 1.1)
    ax.set_axis_off()
    for child, parent in ER_LINKS:
        cb, pb = _er_box(child), _er_box(parent)
        start, end = _er_anchor(cb, pb), _er_anchor(pb, cb)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=C["muted"], linewidth=1.1, zorder=1)
        for (px, py), (qx, qy), mark in ((start, end, "N"), (end, start, "1")):
            dx, dy = qx - px, qy - py
            norm = np.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm             # along the line, then a small side step
            ax.text(px + 0.03 * ux - 0.014 * uy, py + 0.03 * uy + 0.014 * ux, mark,
                    fontsize=8.5, color=C["text_secondary"], fontweight="semibold",
                    ha="center", va="center", zorder=5)
    for name, (_, _, fields) in ER_TABLES.items():
        bx, by, bw, bh = _er_box(name)
        dimension = name in ("halving", "calendar", "model")
        header = PALETTE[6] if dimension else PALETTE[0]
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=C["surface"], edgecolor=C["axis"], linewidth=1, zorder=2))
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (bx, by + bh - ER_HEAD), bw, ER_HEAD, boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=header, edgecolor=header, linewidth=0, zorder=3))
        ax.text(bx + 0.014, by + bh - ER_HEAD / 2, name, color="white", fontsize=10.5,
                fontweight="semibold", va="center", zorder=4)
        for i, field in enumerate(fields):
            key = "PK" in field or "FK" in field
            ax.text(bx + 0.014, by + bh - ER_HEAD - ER_ROW * (i + 0.75), field, fontsize=8.8,
                    color=C["text"] if key else C["text_secondary"],
                    fontweight="semibold" if key else "normal", zorder=4)
    ax.text(0.0, 1.085, "ER diagram of the SQLite database (data/btc.db)", fontsize=13.5,
            fontweight="semibold", va="bottom")
    ax.text(0.0, 1.05, "Violet: dimension tables. Blue: fact tables. Lines join a foreign key (N) "
                       "to the primary key it references (1).", fontsize=9.5,
            color=C["text_secondary"], va="bottom")
    return _save(fig, "00_er_diagram")


# --- 14-17. explanatory figures ------------------------------------------------------------------

def validation_scheme(cv: pd.DataFrame, last_day: pd.Timestamp) -> Path:
    """Timeline of the expanding-window folds and the final hold-out test."""
    start, test_start = pd.Timestamp(config.MODEL_START), pd.Timestamp(config.TEST_START)
    folds = cv.groupby("fold")[["val_start", "val_end"]].first()
    rows = [(f"CV fold {k}", pd.Timestamp(r["val_start"]), pd.Timestamp(r["val_end"]), "validate")
            for k, r in folds.iterrows()]
    rows.append(("Final test", test_start, last_day, f"test: {(last_day - test_start).days + 1} unseen days"))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, (label, eval_start, eval_end, text) in enumerate(rows):
        y = len(rows) - 1 - i
        train_days = (eval_start - start).days
        ax.barh(y, train_days, left=mdates.date2num(start), height=0.58, color=BLUE_RAMP[1], linewidth=0)
        ax.text(mdates.date2num(start) + train_days / 2, y, "train", ha="center", va="center",
                fontsize=8, color=C["text_secondary"])
        width = (eval_end - eval_start).days + 1
        color = PALETTE[1] if label == "Final test" else PALETTE[0]
        ax.barh(y, width, left=mdates.date2num(eval_start), height=0.58, color=color, linewidth=0)
        ax.text(mdates.date2num(eval_start) + width / 2, y, text, ha="center", va="center",
                fontsize=8, color="white", fontweight="semibold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.xaxis_date()
    _year_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title("How the models were checked: always trained on the past, tested on the future")
    _subtitle(ax, "Five expanding-window folds inside 2017–2024 compare the models; "
                  "2025–2026 stays untouched until the final test. No shuffling anywhere.")
    return _save(fig, "14_validation_scheme")


def pipeline_diagram(facts: dict) -> Path:
    """The project as a flow of seven steps, with the key numbers of each step."""
    steps = [
        ("Raw data", ["Kaggle CSV, 1-min candles", f"{facts['rows'] / 1e6:.1f} M rows · {facts['csv_mb']:.0f} MB"]),
        ("Chunked loading", ["1 M rows per chunk", f"float32: −{facts['memory_saving']:.0f} % memory",
                             f"Parquet by year · {facts['parquet_mb']:.0f} MB"]),
        ("Quality audit", [f"{facts['missing']} gaps, {facts['duplicates']} duplicates",
                           f"{facts['spikes']} bad ticks repaired", "model from 2017 (liquidity)"]),
        ("Volatility", ["5-minute log returns", "→ daily realised volatility",
                        f"{facts['days']:,} days · {facts['hours'] / 1e3:.0f} k hours"]),
        ("SQLite", ["8 tables, ER diagram", f"{facts['queries']} analytical queries", "JOIN · CTE · windows"]),
        ("Forecasting", [f"{facts['baselines']} baselines", "HAR · GARCH · LightGBM", "5-fold CV + hold-out"]),
        ("Results", ["report figures, tables", "Streamlit dashboard", "next-day forecast"]),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 2.7))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    for i, (title, lines) in enumerate(steps):
        x, w = i + 0.07, 0.86
        color = PALETTE[0] if i < 4 else PALETTE[6]
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, 0.06), w, 0.78, boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=C["surface"], edgecolor=C["axis"], linewidth=1))
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, 0.66), w, 0.18, boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=color, edgecolor=color, linewidth=0))
        ax.text(x + 0.05, 0.75, f"{i + 1}  {title}", fontsize=9.5, fontweight="semibold",
                color="white", va="center")
        for j, line in enumerate(lines):
            ax.text(x + 0.05, 0.52 - 0.15 * j, line, fontsize=8.2, color=C["text_secondary"], va="center")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.07, 0.45), xytext=(x + w, 0.45),
                        arrowprops=dict(arrowstyle="-|>", color=C["muted"], linewidth=1.1,
                                        shrinkA=1, shrinkB=1, mutation_scale=11))
    ax.text(0.07, 0.99, "From 7.8 million one-minute candles to a next-day volatility forecast",
            fontsize=13, fontweight="semibold", va="top")
    ax.text(0.07, 0.905, "Blue: data engineering. Violet: database, modelling and results.",
            fontsize=9, color=C["text_secondary"], va="top")
    ax.set_ylim(0, 1.08)
    return _save(fig, "15_pipeline")


def volatility_calendar(daily: pd.DataFrame) -> Path:
    """Year × week heat-map of realised volatility: stormy weeks at a glance."""
    vol = daily.loc[config.MODEL_START:, "realized_vol"]
    years = sorted(vol.index.year.unique())
    grid = np.full((len(years), 53), np.nan)
    for r, year in enumerate(years):
        part = vol[vol.index.year == year]
        week = np.minimum((part.index.day_of_year - 1) // 7, 52)
        grid[r] = pd.Series(part.to_numpy()).groupby(week).mean().reindex(range(53)).to_numpy()
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad(C["surface"])
    fig, ax = plt.subplots(figsize=(12.5, 0.36 * len(years) + 1.4))
    image = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=1, vmax=7,
                      interpolation="nearest")
    ax.set_yticks(range(len(years)), [str(y) for y in years])
    starts = [(pd.Timestamp(2021, m, 1).day_of_year - 1) // 7 for m in range(1, 13)]
    ax.set_xticks(starts, [pd.Timestamp(2021, m, 1).strftime("%b") for m in range(1, 13)])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, 53, 1):
        ax.axvline(x, color=C["surface"], linewidth=1.2)
    for y in np.arange(-0.5, len(years), 1):
        ax.axhline(y, color=C["surface"], linewidth=2.5)
    for i, event in enumerate(config.MARKET_EVENTS, start=1):
        day = pd.Timestamp(event["date"])
        if day.year in years:
            r, c = years.index(day.year), min((day.day_of_year - 1) // 7, 52)
            ax.text(c, r, str(i), ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    bar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, extend="both")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=C["text_secondary"])
    bar.set_label("mean daily volatility, %", fontsize=8.5, color=C["text_secondary"])
    ax.set_title("Volatility calendar: every week since 2017")
    _subtitle(ax, "Each cell is one week; darker means stormier. Numbers mark the market shocks "
                  "listed under figure 1. Storms come in clusters and fade over the years.")
    return _save(fig, "16_volatility_calendar")


def forecast_interval(intervals: pd.DataFrame, predictions: pd.DataFrame, coverage: float,
                      days: int = 120) -> Path:
    last = intervals.iloc[-days:]
    inside = last["inside"].astype(bool)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(last.index, last["lower"], last["upper"], color=PALETTE[0], alpha=0.15,
                    linewidth=0, label="80 % forecast range")
    ax.plot(last.index, predictions.loc[last.index, "LightGBM"], color=PALETTE[0], linewidth=1.6,
            label="LightGBM forecast")
    ax.scatter(last.index[inside], last["actual"][inside], s=14, color=C["text"], zorder=4,
               label="actual, inside the range")
    ax.scatter(last.index[~inside], last["actual"][~inside], s=26, color=PALETTE[1], zorder=5,
               label="actual, outside the range")
    ax.set_ylabel("Daily volatility, %")
    top = max(last["actual"].max(), last["upper"].max())
    ax.set_ylim(0, top * 1.25)                      # head-room so the legend never covers data
    ax.legend(loc="upper left", ncols=4, frameon=True, facecolor=C["surface"], edgecolor="none",
              framealpha=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"Forecast with an 80 % range: reality fell inside on {coverage:.0f} % of test days")
    _subtitle(ax, f"LightGBM quantile models (10th and 90th percentiles); last {days} days of the "
                  "hold-out test shown. A well-calibrated range should hold about 80 % of days.")
    return _save(fig, "17_forecast_interval")


def _pipeline_facts(daily: pd.DataFrame) -> dict:
    """Numbers for the pipeline diagram, read from the saved reports."""
    from src.database import load_queries
    from src.models import default_models
    from src.utils import load_json
    facts = {"rows": 7.8e6, "csv_mb": 379, "parquet_mb": 183, "memory_saving": 42,
             "missing": 0, "duplicates": 0, "spikes": 0, "hours": 24 * len(daily)}
    try:
        conversion = load_json(config.REPORTS_DIR / "conversion.json")
        facts.update(rows=conversion["rows"], csv_mb=conversion["csv_size_mb"],
                     parquet_mb=conversion["parquet_size_mb"], memory_saving=conversion["memory_saving_pct"])
        quality = load_json(config.REPORTS_DIR / "quality_report.json")
        facts.update(missing=quality["missing_minutes"], duplicates=quality["duplicate_timestamps"],
                     spikes=quality["spikes"])
    except (FileNotFoundError, KeyError):
        pass
    if config.HOURLY_PATH.exists():
        facts["hours"] = len(pd.read_parquet(config.HOURLY_PATH, columns=["close"]))
    facts.update(days=len(daily), queries=len(load_queries()),
                 baselines=sum(m.is_baseline for m in default_models()))
    return facts


# --- all figures ----------------------------------------------------------------------------------

def make_all(verbose: bool = True) -> list[Path]:
    """Create every report figure that the available results allow."""
    from src.database import run_query
    from src.eda import autocorrelations, correlation_matrix, model_period, rolling_vix_correlation, \
        stl_decomposition
    from src.external import load_external
    from src.features import load_daily

    use_matplotlib_style()
    daily = load_daily()
    period = model_period(daily)
    external = load_external()
    paths = [er_diagram(), price_history(daily)]

    quality_path = config.TABLES_DIR / "quality_by_year.csv"
    if quality_path.exists():
        paths.append(zero_volume_by_year(pd.read_csv(quality_path)))
    paths += [return_distribution(period), rolling_volatility(daily),
              stl_figure(stl_decomposition(period)),
              acf_figure(autocorrelations(period), len(period)),
              correlation_figure(correlation_matrix(period, external),
                                 rolling_vix_correlation(period, external))]
    if config.DB_PATH.exists():
        paths.append(intraday_heatmap(run_query("intraday_profile")))
    paths += [pipeline_diagram(_pipeline_facts(daily)), volatility_calendar(daily)]

    metrics_path = config.TABLES_DIR / "test_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        cv = pd.read_csv(config.TABLES_DIR / "cv_folds.csv")
        predictions = pd.read_csv(config.TABLES_DIR / "test_predictions.csv",
                                  index_col="date", parse_dates=["date"])
        best = metrics.sort_values("mae").iloc[0]["model"]
        by_month = pd.read_csv(config.TABLES_DIR / "errors_by_month.csv", index_col=0)
        by_regime = pd.read_csv(config.TABLES_DIR / "errors_by_regime.csv", index_col=0)
        bias = pd.read_csv(config.TABLES_DIR / "errors_bias_by_regime.csv", index_col=0)
        thresholds = pd.read_csv(config.TABLES_DIR / "errors_regime_thresholds.csv")
        importance = pd.read_csv(config.TABLES_DIR / "lgbm_feature_importance.csv", index_col=0).iloc[:, 0]
        paths += [forecast_vs_actual(predictions, best), model_comparison(metrics, cv),
                  error_by_month(by_month), error_by_regime(by_regime, bias, thresholds),
                  feature_importance(importance), validation_scheme(cv, daily.index[-1])]
        intervals_path = config.TABLES_DIR / "test_intervals.csv"
        if intervals_path.exists():
            from src.utils import load_json
            coverage = load_json(config.REPORTS_DIR / "experiment.json")["lgbm_interval"]["coverage_pct"]
            intervals = pd.read_csv(intervals_path, index_col="date", parse_dates=["date"])
            paths.append(forecast_interval(intervals, predictions, coverage))
    if verbose:
        for path in paths:
            print(f"  {path.relative_to(config.ROOT)}")
    return paths
