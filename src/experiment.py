"""The full modelling experiment: cross-validation, hold-out test, error analysis,
saved models and a forecast for the day after the data end."""
from __future__ import annotations

import re

import joblib
import pandas as pd

from src import config
from src.evaluate import backtest, cross_validate, dm_table, error_analysis
from src.features import load_daily, make_features, split_periods
from src.models import default_models
from src.utils import headline, load_json, print_table, save_json, save_table

EXPERIMENT_PATH = config.REPORTS_DIR / "experiment.json"
PREDICTIONS_PATH = config.TABLES_DIR / "test_predictions.csv"
INTERVALS_PATH = config.TABLES_DIR / "test_intervals.csv"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every metric across the folds."""
    summary = cv.groupby("model", sort=False)[["mae", "rmse", "mase", "qlike", "fit_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def forecast_next_day(daily: pd.DataFrame) -> pd.DataFrame:
    """Refit every model on all available days and forecast the next calendar day."""
    next_day = daily.index[-1] + pd.Timedelta(days=1)
    extended = daily.reindex(daily.index.append(pd.DatetimeIndex([next_day])))
    features = make_features(extended)
    history = features.loc[config.MODEL_START: daily.index[-1]].index
    target = pd.DatetimeIndex([next_day])
    rows = []
    for model in default_models():
        model.fit(extended, features, history)
        row = {"date": next_day.date(), "model": model.name,
               "forecast_vol": float(model.predict(extended, features, target).iloc[0]),
               "lower": None, "upper": None}
        if hasattr(model, "predict_interval"):
            bounds = model.predict_interval(extended, features, target).iloc[0]
            row.update(lower=float(bounds["lower"]), upper=float(bounds["upper"]))
        rows.append(row)
    return pd.DataFrame(rows)


def interval_check(fitted_model, daily: pd.DataFrame, features: pd.DataFrame,
                   index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Prediction interval on the test period and how often reality fell inside it."""
    bounds = fitted_model.predict_interval(daily, features, index)
    actual = daily["realized_vol"].reindex(index)
    inside = (actual >= bounds["lower"]) & (actual <= bounds["upper"])
    low_q, high_q = config.INTERVAL_QUANTILES
    summary = {"level_pct": round(100 * (high_q - low_q)),
               "coverage_pct": 100 * float(inside.mean()),
               "below_pct": 100 * float((actual < bounds["lower"]).mean()),
               "above_pct": 100 * float((actual > bounds["upper"]).mean()),
               "mean_width": float((bounds["upper"] - bounds["lower"]).mean())}
    return bounds.assign(actual=actual, inside=inside), summary


def run_experiment(verbose: bool = True) -> dict:
    daily = load_daily()
    features = make_features(daily)
    train, test = split_periods(features)
    models = default_models()
    if verbose:
        print(f"  train: {train.index[0]:%Y-%m-%d} .. {train.index[-1]:%Y-%m-%d} ({len(train)} days)")
        print(f"  test:  {test.index[0]:%Y-%m-%d} .. {test.index[-1]:%Y-%m-%d} ({len(test)} days)")
        headline("Time-series cross-validation (training period only)")
    cv = cross_validate(models, daily, features, train.index, verbose=verbose)

    if verbose:
        headline("Hold-out test")
    predictions, metrics, fitted = backtest(models, daily, features, train.index, test.index)
    best = metrics.sort_values("mae").iloc[0]["model"]
    # every model vs the naive forecast, and the topic's key pair: GARCH vs rolling std
    dm = pd.concat([dm_table(predictions),
                    dm_table(predictions, reference="Rolling std (30d)", models=["GARCH(1,1)"])],
                   ignore_index=True)
    errors = error_analysis(predictions, daily, train.index, best)
    intervals, interval_summary = interval_check(fitted["LightGBM"], daily, features, test.index)
    next_day = forecast_next_day(daily)

    # --- save everything -------------------------------------------------
    config.MODELS_DIR.mkdir(exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, config.MODELS_DIR / f"{slug(name)}.joblib")
    save_table(cv, "cv_folds")
    save_table(cv_summary(cv), "cv_summary")
    save_table(metrics, "test_metrics")
    predictions.to_csv(PREDICTIONS_PATH, index_label="date", float_format="%.6f")
    save_table(dm, "dm_tests")
    for key, table in errors.items():
        save_table(table, f"errors_{key}", index=key != "regime_thresholds")
    save_table(next_day, "next_day_forecast")
    intervals.to_csv(INTERVALS_PATH, index_label="date", float_format="%.6f")
    lgbm = fitted["LightGBM"]
    save_table(lgbm.feature_importance().rename("gain_share").to_frame(), "lgbm_feature_importance", index=True)
    save_table(lgbm.tuning_table_, "lgbm_tuning")

    summary = {
        "train_period": [str(train.index[0].date()), str(train.index[-1].date())],
        "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
        "train_days": len(train), "test_days": len(test),
        "best_model": best,
        "test_metrics": metrics.set_index("model")[["mae", "rmse", "mase", "smape", "qlike"]].to_dict("index"),
        "har_coefficients": fitted["HAR-RV"].coefficients_,
        "garch_params": fitted["GARCH(1,1)"].params_.to_dict(),
        "lgbm_best_params": lgbm.best_params_,
        "lgbm_interval": interval_summary,
        "next_day_forecast": next_day.to_dict("records"),
    }
    save_json(summary, EXPERIMENT_PATH)

    if verbose:
        print_table(metrics[["model", "mae", "rmse", "mase", "qlike", "mae_vs_naive_pct",
                             "fit_seconds", "peak_memory_mb"]].sort_values("mae"))
        print(f"\n  best model on the test period: {best}")
        print(f"  LightGBM {interval_summary['level_pct']} % interval covered "
              f"{interval_summary['coverage_pct']:.1f} % of test days")
    return {"cv": cv, "metrics": metrics, "predictions": predictions, "dm": dm,
            "errors": errors, "next_day": next_day, "summary": summary}


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("Model results not found. Run step: 8 - train and compare models")
    return pd.read_csv(PREDICTIONS_PATH, index_col="date", parse_dates=["date"])


def load_summary() -> dict:
    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError("Model results not found. Run step: 8 - train and compare models")
    return load_json(EXPERIMENT_PATH)
