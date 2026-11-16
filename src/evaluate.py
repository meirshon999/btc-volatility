"""Forecast metrics, time-series cross-validation, hold-out backtest and error analysis."""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from src import config
from src.models import Forecaster
from src.utils import measure

METRICS = ["mae", "rmse", "mase", "smape", "qlike"]


# --- metrics (inputs are volatilities in % per day) -----------------------

def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(100 * np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast))))


def mase(actual: np.ndarray, forecast: np.ndarray, scale: float) -> float:
    """MAE divided by the in-sample MAE of the naive forecast (< 1 beats naive)."""
    return mae(actual, forecast) / scale


def qlike(actual: np.ndarray, forecast: np.ndarray) -> float:
    """QLIKE loss on variances - robust to noise in the realised measure (Patton, 2011)."""
    ratio = actual ** 2 / forecast ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def naive_scale(volatility: pd.Series) -> float:
    """In-sample MAE of the one-step naive forecast, the denominator of MASE."""
    return float(volatility.diff().abs().mean())


def score(actual: pd.Series, forecast: pd.Series, scale: float) -> dict:
    """All metrics on the dates where both values exist."""
    both = pd.concat([actual, forecast], axis=1).dropna()
    a, f = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    return {"n": len(both), "mae": mae(a, f), "rmse": rmse(a, f), "mase": mase(a, f, scale),
            "smape": smape(a, f), "qlike": qlike(a, f)}


# --- running models ---------------------------------------------------------

def fit_and_predict(model: Forecaster, daily: pd.DataFrame, features: pd.DataFrame,
                    train_index: pd.DatetimeIndex, eval_index: pd.DatetimeIndex
                    ) -> tuple[Forecaster, pd.Series, dict]:
    """Fit a fresh copy of the model and forecast; also measure time and memory."""
    fitted = copy.deepcopy(model)
    with measure() as fit_cost:
        fitted.fit(daily, features, train_index)
    with measure() as predict_cost:
        forecast = fitted.predict(daily, features, eval_index)
    cost = {"fit_seconds": fit_cost.seconds, "predict_seconds": predict_cost.seconds,
            "peak_memory_mb": max(fit_cost.peak_mb, predict_cost.peak_mb),
            "model_size_kb": len(pickle.dumps(fitted)) / 1024}
    return fitted, forecast, cost


def cross_validate(models: list[Forecaster], daily: pd.DataFrame, features: pd.DataFrame,
                   train_index: pd.DatetimeIndex, n_splits: int = config.CV_SPLITS,
                   verbose: bool = True) -> pd.DataFrame:
    """Expanding-window TimeSeriesSplit inside the training period."""
    rows = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (fit_pos, val_pos) in enumerate(splitter.split(train_index), start=1):
        fit_idx, val_idx = train_index[fit_pos], train_index[val_pos]
        scale = naive_scale(daily["realized_vol"].reindex(fit_idx))
        actual = daily["realized_vol"].reindex(val_idx)
        for model in models:
            _, forecast, cost = fit_and_predict(model, daily, features, fit_idx, val_idx)
            rows.append({"model": model.name, "fold": fold,
                         "val_start": val_idx[0].date(), "val_end": val_idx[-1].date(),
                         **score(actual, forecast, scale), **cost})
        if verbose:
            best = min((r for r in rows if r["fold"] == fold), key=lambda r: r["mae"])
            print(f"  fold {fold}: {val_idx[0]:%Y-%m-%d} .. {val_idx[-1]:%Y-%m-%d}  "
                  f"best = {best['model']} (MAE {best['mae']:.3f})")
    return pd.DataFrame(rows)


def backtest(models: list[Forecaster], daily: pd.DataFrame, features: pd.DataFrame,
             train_index: pd.DatetimeIndex, test_index: pd.DatetimeIndex
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Forecaster]]:
    """Fit on the whole training period, forecast the untouched test period."""
    scale = naive_scale(daily["realized_vol"].reindex(train_index))
    actual = daily["realized_vol"].reindex(test_index).rename("actual")
    predictions, rows, fitted_models = [actual], [], {}
    for model in models:
        fitted, forecast, cost = fit_and_predict(model, daily, features, train_index, test_index)
        fitted_models[model.name] = fitted
        predictions.append(forecast.rename(model.name))
        rows.append({"model": model.name, "family": model.family,
                     **score(actual, forecast, scale), **cost})
    metrics = pd.DataFrame(rows)
    naive_mae = metrics.loc[metrics["model"] == "Naive", "mae"].iloc[0]
    metrics["mae_vs_naive_pct"] = 100 * (metrics["mae"] / naive_mae - 1)
    return pd.concat(predictions, axis=1), metrics, fitted_models


# --- statistical comparison ----------------------------------------------

def diebold_mariano(actual: pd.Series, forecast_a: pd.Series, forecast_b: pd.Series,
                    loss: str = "absolute") -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction (h = 1).

    Negative statistic: forecast A has the smaller loss. Returns (statistic, p-value).
    """
    both = pd.concat([actual, forecast_a, forecast_b], axis=1).dropna().to_numpy()
    y, a, b = both[:, 0], both[:, 1], both[:, 2]
    if loss == "absolute":
        d = np.abs(y - a) - np.abs(y - b)
    elif loss == "squared":
        d = (y - a) ** 2 - (y - b) ** 2
    elif loss == "qlike":                      # per-day QLIKE on variances
        ra, rb = y ** 2 / a ** 2, y ** 2 / b ** 2
        d = (ra - np.log(ra)) - (rb - np.log(rb))
    else:
        raise ValueError(f"unknown loss: {loss}")
    n = len(d)
    statistic = d.mean() / np.sqrt(d.var(ddof=0) / n)
    statistic *= np.sqrt((n - 1) / n)           # HLN small-sample correction for h = 1
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def dm_table(predictions: pd.DataFrame, reference: str = "Naive",
             models: list[str] | None = None) -> pd.DataFrame:
    """Each model (or the listed ones) against the reference forecast, absolute and squared loss."""
    rows = []
    for model in models or predictions.columns.drop(["actual", reference]):
        for loss in ("absolute", "squared", "qlike"):
            stat, p = diebold_mariano(predictions["actual"], predictions[model],
                                      predictions[reference], loss)
            rows.append({"model": model, "reference": reference, "loss": loss,
                         "dm_statistic": stat, "p_value": p,
                         "significantly_better": bool(stat < 0 and p < 0.05)})
    return pd.DataFrame(rows)


# --- error analysis -----------------------------------------------------------

def error_analysis(predictions: pd.DataFrame, daily: pd.DataFrame,
                   train_index: pd.DatetimeIndex, best_model: str) -> dict[str, pd.DataFrame]:
    """Where and when the forecasts fail: by month, volatility regime, weekday, worst days."""
    models = list(predictions.columns.drop("actual"))
    errors = predictions[models].sub(predictions["actual"], axis=0)   # forecast - actual

    by_month = errors.abs().groupby(errors.index.to_period("M")).mean()
    by_month.index = by_month.index.astype(str)

    # regimes are defined with training-period quantiles, so the test data do not set them
    low, high = daily["realized_vol"].reindex(train_index).quantile([1 / 3, 2 / 3])
    regime = pd.cut(predictions["actual"], [0, low, high, np.inf],
                    labels=["low", "medium", "high"])
    by_regime = errors.abs().groupby(regime, observed=True).mean()
    bias_by_regime = errors.groupby(regime, observed=True).mean()

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = errors.abs().groupby(errors.index.day_of_week).mean()
    by_weekday.index = [weekday_names[i] for i in by_weekday.index]

    worst = (predictions[["actual", best_model, "Naive"]]
             .assign(error=errors[best_model], abs_error=errors[best_model].abs(),
                     log_return=daily["log_return"].reindex(predictions.index))
             .nlargest(10, "abs_error"))
    return {"by_month": by_month, "by_regime": by_regime, "bias_by_regime": bias_by_regime,
            "by_weekday": by_weekday, "worst_days": worst,
            "regime_thresholds": pd.DataFrame({"low_below": [low], "high_above": [high]})}
