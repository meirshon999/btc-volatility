import numpy as np
import pandas as pd
import pytest

from src.evaluate import diebold_mariano, mae, mase, qlike, rmse, smape


def test_basic_metrics():
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.0, 2.5, 2.0])
    assert mae(actual, forecast) == pytest.approx(0.5)
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(1.25 / 3))
    assert mase(actual, forecast, scale=0.5) == pytest.approx(1.0)
    assert smape(actual, actual) == pytest.approx(0.0)


def test_qlike_is_zero_for_a_perfect_forecast():
    actual = np.array([0.5, 2.0, 3.0])
    assert qlike(actual, actual) == pytest.approx(0.0)


def test_qlike_punishes_under_prediction_more_than_over_prediction():
    actual = np.array([2.0])
    assert qlike(actual, np.array([1.0])) > qlike(actual, np.array([4.0]))


def test_diebold_mariano_detects_the_better_forecast():
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(size=500))
    good = actual + rng.normal(scale=0.1, size=500)
    bad = actual + rng.normal(scale=1.0, size=500)
    statistic, p_value = diebold_mariano(actual, good, bad)
    assert statistic < 0 and p_value < 0.01
