"""Forecasters of next-day realised volatility.

Every forecaster has the same interface:
    fit(daily, features, train_index)  - learn only from the training dates;
    predict(daily, features, index)    - one-step-ahead forecasts for `index`,
                                         each using information up to the day before.
Forecasts are on the volatility scale (% per day).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src import config

TARGET = "target_log_vol"
ONE_DAY = pd.Timedelta(days=1)


def _volatility_on(daily: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Realised volatility on the union of known dates and forecast dates."""
    return daily["realized_vol"].reindex(daily.index.union(index))


def _training_xy(features: pd.DataFrame, index: pd.DatetimeIndex,
                 columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows = features.loc[features.index.intersection(index), [*columns, TARGET]].dropna()
    return rows[columns], rows[TARGET]


def _predict_log_model(model, smearing: float, features: pd.DataFrame,
                       index: pd.DatetimeIndex, columns: list[str]) -> pd.Series:
    """Back-transform a log-volatility model with Duan's smearing correction."""
    X = features.reindex(index)[columns]
    valid = X.notna().all(axis=1)
    out = pd.Series(np.nan, index=index, dtype="float64")
    if valid.any():
        out[valid] = np.exp(model.predict(X[valid])) * smearing
    return out


class Forecaster:
    name = "Forecaster"
    family = "baseline"            # baseline / econometric / ml
    description = ""

    @property
    def is_baseline(self) -> bool:
        return self.family == "baseline"

    def fit(self, daily: pd.DataFrame, features: pd.DataFrame,
            train_index: pd.DatetimeIndex) -> "Forecaster":
        return self

    def predict(self, daily: pd.DataFrame, features: pd.DataFrame,
                index: pd.DatetimeIndex) -> pd.Series:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


# --- baselines ------------------------------------------------------------

class NaiveForecaster(Forecaster):
    name = "Naive"
    description = "tomorrow's volatility equals today's"

    def predict(self, daily, features, index):
        vol = _volatility_on(daily, index)
        return vol.shift(1, freq="D").reindex(index).rename(self.name)


class SeasonalNaiveForecaster(Forecaster):
    name = "Seasonal naive"
    description = "volatility of the same weekday one week earlier"

    def __init__(self, season: int = config.SEASON_LENGTH):
        self.season = season

    def predict(self, daily, features, index):
        vol = _volatility_on(daily, index)
        return vol.shift(self.season, freq="D").reindex(index).rename(self.name)


class MovingAverageForecaster(Forecaster):
    name = "Moving average (30d)"
    description = "average volatility of the previous 30 days"

    def __init__(self, window: int = 30):
        self.window = window
        self.name = f"Moving average ({window}d)"

    def predict(self, daily, features, index):
        vol = daily["realized_vol"]
        mean = vol.rolling(f"{self.window}D", min_periods=self.window // 2).mean()
        return mean.shift(1, freq="D").reindex(index).rename(self.name)


class RollingStdForecaster(Forecaster):
    name = "Rolling std (30d)"
    description = ("standard deviation of the previous 30 daily returns - the classic baseline "
                   "for GARCH, built from the same daily returns")

    def __init__(self, window: int = 30):
        self.window = window
        self.name = f"Rolling std ({window}d)"

    def predict(self, daily, features, index):
        returns = daily["log_return"]
        std = returns.rolling(f"{self.window}D", min_periods=self.window // 2).std()
        return std.shift(1, freq="D").reindex(index).rename(self.name)


# --- econometric models -----------------------------------------------------

class HARForecaster(Forecaster):
    name = "HAR-RV"
    family = "econometric"
    description = "linear regression on daily, weekly and monthly averages of log volatility (Corsi, 2009)"
    columns = ["log_vol_lag1", "log_vol_mean_5", "log_vol_mean_22"]

    def fit(self, daily, features, train_index):
        X, y = _training_xy(features, train_index, self.columns)
        self.model_ = LinearRegression().fit(X, y)
        self.smearing_ = float(np.mean(np.exp(y - self.model_.predict(X))))
        self.coefficients_ = {"intercept": float(self.model_.intercept_),
                              **dict(zip(self.columns, map(float, self.model_.coef_)))}
        return self

    def predict(self, daily, features, index):
        return _predict_log_model(self.model_, self.smearing_, features, index,
                                  self.columns).rename(self.name)


class GarchForecaster(Forecaster):
    name = "GARCH(1,1)"
    family = "econometric"
    description = "GARCH(1,1) with Student-t errors on daily close-to-close returns"

    def __init__(self, start: str = config.MODEL_START):
        self.start = start

    def _model(self, returns: pd.Series):
        return arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1,
                          dist="t", rescale=False)

    def fit(self, daily, features, train_index):
        returns = daily["log_return"].loc[self.start: train_index.max()].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._model(returns).fit(disp="off", show_warning=False)
        self.params_ = result.params
        return self

    def predict(self, daily, features, index):
        # filter the whole return history with the parameters fixed from training:
        # sigma_t depends only on returns up to t-1, so this is a one-step forecast.
        returns = daily["log_return"].loc[self.start: index.max()].dropna()
        fixed = self._model(returns).fix(self.params_)
        out = fixed.conditional_volatility.reindex(index)
        next_day = returns.index.max() + ONE_DAY
        if next_day in index:
            variance = fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
            out.loc[next_day] = np.sqrt(variance)
        return out.astype("float64").rename(self.name)


# --- machine learning -------------------------------------------------------

class LightGBMForecaster(Forecaster):
    name = "LightGBM"
    family = "ml"
    description = "gradient boosting on lagged volatility, returns, volume and calendar features"
    columns = ["log_vol_lag1", "log_vol_lag2", "log_vol_lag3", "log_vol_lag4", "log_vol_lag5",
               "log_vol_lag6", "log_vol_lag7", "log_vol_mean_5", "log_vol_mean_22",
               "log_vol_std_22", "return_lag1", "abs_return_lag1", "negative_return_lag1",
               "log_parkinson_lag1", "log_volume_lag1", "volume_change_lag1",
               "weekday", "is_weekend", "days_since_halving"]
    param_grid = {"num_leaves": [7, 15], "learning_rate": [0.03, 0.1], "n_estimators": [300, 600]}
    fixed_params = {"min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
                    "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1,
                    "random_state": config.RANDOM_STATE, "n_jobs": -1}

    def __init__(self, tune: bool = True, inner_splits: int = 3):
        self.tune = tune
        self.inner_splits = inner_splits

    def _candidates(self) -> list[dict]:
        keys = list(self.param_grid)
        return [dict(zip(keys, values)) for values in itertools.product(*self.param_grid.values())]

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame]:
        """Pick hyper-parameters with an inner TimeSeriesSplit (no shuffling)."""
        splitter = TimeSeriesSplit(n_splits=self.inner_splits)
        rows = []
        for params in self._candidates():
            scores = []
            for fit_idx, val_idx in splitter.split(X):
                model = LGBMRegressor(**self.fixed_params, **params)
                model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                pred = np.exp(model.predict(X.iloc[val_idx]))
                scores.append(np.mean(np.abs(np.exp(y.iloc[val_idx]) - pred)))
            rows.append({**params, "inner_cv_mae": float(np.mean(scores))})
        table = pd.DataFrame(rows).sort_values("inner_cv_mae", ignore_index=True)
        best = table.iloc[0][list(self.param_grid)].to_dict()
        best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
        return best, table

    def fit(self, daily, features, train_index):
        X, y = _training_xy(features, train_index, self.columns)
        if self.tune:
            self.best_params_, self.tuning_table_ = self._tune(X, y)
        else:
            self.best_params_ = {k: v[0] for k, v in self.param_grid.items()}
        self.model_ = LGBMRegressor(**self.fixed_params, **self.best_params_).fit(X, y)
        self.smearing_ = float(np.mean(np.exp(y - self.model_.predict(X))))
        # quantile models for the prediction interval (same tuned settings, pinball loss)
        self.quantile_models_ = {
            q: LGBMRegressor(**self.fixed_params, **self.best_params_, objective="quantile",
                             alpha=q).fit(X, y)
            for q in config.INTERVAL_QUANTILES}
        return self

    def predict(self, daily, features, index):
        return _predict_log_model(self.model_, self.smearing_, features, index,
                                  self.columns).rename(self.name)

    def predict_interval(self, daily, features, index) -> pd.DataFrame:
        """Lower and upper bounds of the prediction interval (volatility, %).

        Quantiles survive the exp() back-transform unchanged, so no correction is needed.
        """
        low_q, high_q = config.INTERVAL_QUANTILES
        low = _predict_log_model(self.quantile_models_[low_q], 1.0, features, index, self.columns)
        high = _predict_log_model(self.quantile_models_[high_q], 1.0, features, index, self.columns)
        return pd.DataFrame({"lower": np.minimum(low, high), "upper": np.maximum(low, high)})

    def feature_importance(self) -> pd.Series:
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.columns).sort_values(ascending=False)


def default_models() -> list[Forecaster]:
    """All compared approaches, baselines first. The order is fixed (it sets chart colours)."""
    return [NaiveForecaster(), SeasonalNaiveForecaster(), MovingAverageForecaster(30),
            RollingStdForecaster(30), HARForecaster(), GarchForecaster(), LightGBMForecaster()]
