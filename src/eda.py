"""Exploratory analysis: descriptive statistics, outliers, stationarity,
seasonal decomposition, autocorrelation, correlations and hypothesis tests."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src import config
from src.external import load_external
from src.features import load_daily
from src.utils import save_json, save_table

ANALYSIS_COLUMNS = ["log_return", "realized_vol", "parkinson_vol", "volume"]


def model_period(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.loc[config.MODEL_START:]


def describe(daily: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics plus skewness and excess kurtosis."""
    data = daily[ANALYSIS_COLUMNS]
    table = data.describe().T
    table["skew"] = data.skew()
    table["excess_kurtosis"] = data.kurt()
    return table


def iqr_outliers(series: pd.Series, k: float = 1.5) -> dict:
    """Tukey's rule: values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    outside = (series < lower) | (series > upper)
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper,
            "outliers": int(outside.sum()), "share_pct": 100 * float(outside.mean())}


def outlier_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = {column: iqr_outliers(daily[column].dropna()) for column in ANALYSIS_COLUMNS}
    return pd.DataFrame(rows).T


def stationarity(series: pd.Series) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) tests side by side."""
    series = series.dropna()
    with warnings.catch_warnings():
        # KPSS warns when p is outside its table; adfuller warns about a future return type
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    adf_rejects, kpss_rejects = adf_p < 0.05, kpss_p < 0.05
    if adf_rejects and not kpss_rejects:
        conclusion = "stationary"
    elif not adf_rejects and kpss_rejects:
        conclusion = "non-stationary (unit root)"
    elif adf_rejects and kpss_rejects:
        conclusion = "tests disagree: mean-reverting but persistent (long memory)"
    else:
        conclusion = "inconclusive"
    return {"adf_statistic": adf_stat, "adf_p_value": adf_p,
            "kpss_statistic": kpss_stat, "kpss_p_value": kpss_p, "conclusion": conclusion}


def stationarity_table(daily: pd.DataFrame) -> pd.DataFrame:
    series = {"log price": np.log(daily["close"]),
              "daily log return": daily["log_return"],
              "log realised volatility": np.log(daily["realized_vol"])}
    return pd.DataFrame({name: stationarity(s) for name, s in series.items()}).T


def stl_decomposition(daily: pd.DataFrame, period: int = config.SEASON_LENGTH):
    """Robust STL of log realised volatility with weekly seasonality."""
    log_vol = np.log(daily["realized_vol"]).asfreq("D")
    return STL(log_vol, period=period, robust=True).fit()


def seasonal_strength(result) -> float:
    """Hyndman's strength of seasonality, 0 (none) to 1 (strong)."""
    return float(max(0.0, 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)))


def autocorrelations(daily: pd.DataFrame, nlags: int = 30) -> pd.DataFrame:
    """ACF of returns and squared returns (volatility clustering), ACF/PACF of log volatility."""
    r = daily["log_return"].dropna()
    log_vol = np.log(daily["realized_vol"]).dropna()
    return pd.DataFrame({
        "acf_return": acf(r, nlags=nlags),
        "acf_squared_return": acf(r ** 2, nlags=nlags),
        "acf_log_vol": acf(log_vol, nlags=nlags),
        "pacf_log_vol": pacf(log_vol, nlags=nlags, method="ywm"),
    }).rename_axis("lag")


def market_frame(daily: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Daily Bitcoin measures joined with external series (weekdays with data only)."""
    frame = pd.DataFrame({
        "BTC realised vol": daily["realized_vol"],
        "BTC |return|": daily["log_return"].abs(),
        "BTC volume (log)": np.log1p(daily["volume"]),
        "BTC range vol": daily["parkinson_vol"],
        "weekend": (daily.index.day_of_week >= 5).astype(int),
    })
    if not external.empty:
        ext = external.reindex(daily.index)
        if "VIX" in ext:
            frame["VIX"] = ext["VIX"]
        if "SP500" in ext:
            frame["S&P 500 |return|"] = 100 * np.log(ext["SP500"]).diff().abs()
        if "USD_INDEX" in ext:
            frame["USD index |change|"] = 100 * np.log(ext["USD_INDEX"]).diff().abs()
    return frame


def correlation_matrix(daily: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations - rank based, so the fat tails do not dominate."""
    return market_frame(daily, external).corr(method="spearman")


def rolling_vix_correlation(daily: pd.DataFrame, external: pd.DataFrame,
                            window: int = 252) -> pd.Series:
    """Correlation of log BTC volatility and log VIX over a moving one-year window
    (252 trading days - the VIX has no weekend values)."""
    if external.empty or "VIX" not in external:
        return pd.Series(dtype="float64")
    both = pd.concat([np.log(daily["realized_vol"]),
                      np.log(external["VIX"].reindex(daily.index))], axis=1).dropna()
    return both.iloc[:, 0].rolling(window).corr(both.iloc[:, 1]).dropna().rename("rolling_corr")


def hypothesis_tests(daily: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Four hypotheses checked on the modelling period."""
    rows = []
    vol = daily["realized_vol"]
    weekend = daily.index.day_of_week >= 5

    u, p = stats.mannwhitneyu(vol[weekend], vol[~weekend], alternative="less")
    rows.append({"hypothesis": "H1: volatility is lower on weekends",
                 "test": "Mann-Whitney U (one-sided)", "statistic": u, "p_value": p,
                 "effect": f"median weekend {vol[weekend].median():.2f} % vs weekday {vol[~weekend].median():.2f} %"})

    lb = acorr_ljungbox(daily["log_return"].dropna() ** 2, lags=[10])
    lb_plain = acorr_ljungbox(daily["log_return"].dropna(), lags=[10])
    rows.append({"hypothesis": "H2: volatility clusters (squared returns are autocorrelated)",
                 "test": "Ljung-Box, 10 lags", "statistic": float(lb["lb_stat"].iloc[0]),
                 "p_value": float(lb["lb_pvalue"].iloc[0]),
                 "effect": f"for raw returns p = {float(lb_plain['lb_pvalue'].iloc[0]):.3f}"})

    next_vol = np.log(vol).shift(-1)
    rho, p = stats.spearmanr(daily["log_return"], next_vol, nan_policy="omit")
    rows.append({"hypothesis": "H3: a fall today raises tomorrow's volatility (leverage effect)",
                 "test": "Spearman, return(t) vs log vol(t+1)", "statistic": rho, "p_value": p,
                 "effect": f"rho = {rho:.3f}"})

    if not external.empty and "VIX" in external:
        vix = external["VIX"].reindex(daily.index)
        both = pd.concat([vol, vix], axis=1).dropna()
        rho, p = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        by_period = []
        for start, end in [("2017", "2019"), ("2020", "2022"), ("2023", "2026")]:
            part = both.loc[start:end]
            by_period.append(f"{start}-{end}: {stats.spearmanr(part.iloc[:, 0], part.iloc[:, 1])[0]:.2f}")
        rows.append({"hypothesis": "H4: Bitcoin volatility moves with stock-market fear (VIX)",
                     "test": "Spearman, BTC vol vs VIX", "statistic": rho, "p_value": p,
                     "effect": "rho by period - " + ", ".join(by_period)})
    table = pd.DataFrame(rows)
    table["reject_h0_at_5pct"] = table["p_value"] < 0.05
    return table


def run_eda(verbose: bool = True) -> dict:
    daily = model_period(load_daily())
    external = load_external()
    result = {
        "describe": describe(daily),
        "outliers": outlier_table(daily),
        "stationarity": stationarity_table(daily),
        "autocorrelations": autocorrelations(daily),
        "correlations": correlation_matrix(daily, external),
        "hypotheses": hypothesis_tests(daily, external),
    }
    stl = stl_decomposition(daily)
    weekday_effect = stl.seasonal.groupby(stl.seasonal.index.day_of_week).mean()
    result["stl_summary"] = {
        "seasonal_strength": seasonal_strength(stl),
        "weekday_seasonal_effect_log": dict(zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                                weekday_effect.round(4).tolist())),
    }
    for name in ("describe", "outliers", "stationarity", "correlations"):
        save_table(result[name], f"eda_{name}", index=True)
    save_table(result["autocorrelations"], "eda_autocorrelations", index=True)
    save_table(result["hypotheses"], "eda_hypotheses")
    save_json(result["stl_summary"], config.REPORTS_DIR / "stl_summary.json")
    if verbose:
        print(result["hypotheses"][["hypothesis", "p_value", "effect"]].to_string(index=False))
        print(f"\n  STL seasonal strength (weekly): {result['stl_summary']['seasonal_strength']:.3f}")
    return result
