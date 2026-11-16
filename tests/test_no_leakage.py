"""The most important property of a forecasting experiment: no look-ahead.

If the realised values from some day onward are changed, every forecast made
for that day or earlier must stay exactly the same.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import make_features
from src.models import (GarchForecaster, HARForecaster, LightGBMForecaster, MovingAverageForecaster,
                        NaiveForecaster, RollingStdForecaster, SeasonalNaiveForecaster)

START = "2020-01-01"


def synthetic_daily(n: int = 420, seed: int = 1) -> pd.DataFrame:
    """A persistent log-volatility process with matching returns, volume and range."""
    rng = np.random.default_rng(seed)
    log_vol = np.empty(n)
    log_vol[0] = 1.0
    for t in range(1, n):
        log_vol[t] = 0.2 + 0.8 * log_vol[t - 1] + rng.normal(scale=0.3)
    vol = np.exp(log_vol)
    returns = vol * rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(returns / 100))
    index = pd.date_range(START, periods=n, freq="D")
    return pd.DataFrame({"close": close, "log_return": returns, "realized_vol": vol,
                         "parkinson_vol": vol * rng.uniform(0.8, 1.2, n),
                         "volume": rng.uniform(1e3, 5e3, n)}, index=index)


MODELS = [NaiveForecaster(), SeasonalNaiveForecaster(), MovingAverageForecaster(30), RollingStdForecaster(30),
          HARForecaster(), GarchForecaster(start=START), LightGBMForecaster(tune=False)]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_forecasts_do_not_use_the_future(model):
    daily = synthetic_daily()
    train_index, test_index = daily.index[:300], daily.index[300:]
    cut = daily.index[360]

    model.fit(daily, make_features(daily), train_index)
    before = model.predict(daily, make_features(daily), test_index)

    changed = daily.copy()
    changed.loc[cut:, ["realized_vol", "parkinson_vol"]] *= 3
    changed.loc[cut:, "log_return"] *= -2
    changed.loc[cut:, "volume"] *= 10
    after = model.predict(changed, make_features(changed), test_index)

    pd.testing.assert_series_equal(before.loc[:cut], after.loc[:cut])
    assert not np.allclose(before.loc[cut + pd.Timedelta(days=2):], after.loc[cut + pd.Timedelta(days=2):])


def test_prediction_interval_is_ordered_and_does_not_use_the_future():
    daily = synthetic_daily()
    train_index, test_index = daily.index[:300], daily.index[300:]
    cut = daily.index[360]
    model = LightGBMForecaster(tune=False).fit(daily, make_features(daily), train_index)
    before = model.predict_interval(daily, make_features(daily), test_index)
    assert (before["lower"] <= before["upper"]).all()

    changed = daily.copy()
    changed.loc[cut:, ["realized_vol", "parkinson_vol"]] *= 3
    after = model.predict_interval(changed, make_features(changed), test_index)
    pd.testing.assert_frame_equal(before.loc[:cut], after.loc[:cut])


def test_features_of_a_day_use_only_earlier_days():
    daily = synthetic_daily()
    day = daily.index[200]
    changed = daily.copy()
    changed.loc[day:, ["realized_vol", "log_return", "volume", "parkinson_vol"]] *= 5
    original, modified = make_features(daily), make_features(changed)
    predictors = original.columns.drop("target_log_vol")
    pd.testing.assert_frame_equal(original.loc[:day, predictors], modified.loc[:day, predictors])
