import numpy as np
import pandas as pd
import pytest

from src import config
from src.database import load_queries, run_query
from src.features import realised_variance
from src.quality import clean_minute


def test_cleaning_rules():
    ts = pd.date_range("2020-01-01", periods=8, freq="min")
    df = pd.DataFrame({
        "ts": ts.insert(3, ts[2]),                               # duplicate timestamp
        "open":  [10, 10, 10, 10, 10, 10, 15, 10, 10],
        "high":  [10, 9, 10, 10, 10, 10, 15, 10, 10],            # high below open -> repaired
        "low":   [10, 10, 10, 10, 10, 10, 15, 10, 10],
        "close": [10, 10, 10, 10, np.nan, -1, 15, 10, 10],       # NaN, negative, spike at 15
        "volume": [1.0] * 9,
    }).astype({c: "float32" for c in ["open", "high", "low", "close"]})
    clean, log = clean_minute(df)
    assert log["dropped_nan"] == 1
    assert log["dropped_duplicates"] == 1
    assert log["dropped_non_positive"] == 1
    assert log["repaired_candles"] == 1
    assert log["replaced_spikes"] == 1
    assert clean["close"].max() == 10
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()


def test_realised_variance_of_a_constant_price_is_zero():
    index = pd.date_range("2021-01-01", periods=288 * 2, freq="5min")
    rv = realised_variance(pd.Series(100.0, index=index))
    assert np.allclose(rv["realized_var"].fillna(0), 0)


def test_realised_variance_matches_the_definition():
    index = pd.date_range("2021-01-01", periods=289, freq="5min")
    close = pd.Series(np.exp(np.cumsum(np.full(289, 0.001))), index=index)
    rv = realised_variance(close)
    # 287 returns of 0.1 % inside the first day -> 287 * 0.1^2 = 0.287 (%^2)
    assert rv["realized_var"].iloc[0] == pytest.approx(287 * 0.01, rel=1e-6)


def test_there_are_at_least_eight_documented_queries():
    queries = load_queries()
    assert len(queries) >= 8
    assert all(q["description"] and q["sql"].upper().lstrip().startswith(("SELECT", "WITH"))
               for q in queries.values())


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="database not built yet")
@pytest.mark.parametrize("name", list(load_queries()))
def test_every_query_runs(name):
    result = run_query(name)
    assert isinstance(result, pd.DataFrame)
