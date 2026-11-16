"""Bitcoin Volatility Lab - console menu.

    python main.py            interactive menu
    python main.py --all      run the whole pipeline without questions
    python main.py --step 8   run one step (see the menu for numbers)

For a one-click start (set-up + dashboard) use start.bat / start.sh / start.py instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import date

import pandas as pd

from src import config
from src.utils import headline, load_state, mark_done, print_table


# --- pipeline steps ---------------------------------------------------------------

def step_convert() -> None:
    from src.loader import convert_csv_to_parquet
    stats = convert_csv_to_parquet()
    print(f"\n  {stats['rows']:,} rows in {stats['chunks']} chunks, {stats['seconds']:.1f} s")
    if stats["rows_after_data_end"]:
        print(f"  {stats['rows_after_data_end']:,} newer rows after {stats['data_end']} left out "
              f"(fixed report period, see DATA_END in src/config.py)")
    print(f"  memory per chunk: {stats['memory_default_mb']:.0f} MB (default dtypes) -> "
          f"{stats['memory_optimised_mb']:.0f} MB (compact), saving {stats['memory_saving_pct']:.1f} %")
    print(f"  CSV {stats['csv_size_mb']:.0f} MB -> Parquet {stats['parquet_size_mb']:.0f} MB; "
          f"float32 rounding error at most {stats['max_float32_relative_error']:.1e}")
    from src.utils import save_json
    save_json(stats, config.REPORTS_DIR / "conversion.json")
    mark_done("convert", rows=stats["rows"])


def step_audit() -> None:
    from src.quality import run_audit
    result = run_audit()
    s = result["summary"]
    print(f"\n  {s['rows']:,} minutes, coverage {s['coverage_pct']:.2f} %, duplicates {s['duplicate_timestamps']}, "
          f"invalid candles {s['invalid_candles']}, one-minute spikes {s['spikes']}")
    print(f"  recommended modelling start: {s['recommended_model_start']} "
          f"(configured: {s['configured_model_start']})")
    mark_done("audit")


def step_features() -> None:
    from src.features import build_tables
    result = build_tables()
    print(f"\n  cleaning: {result['cleaning']}")
    print(f"  hourly bars: {result['hourly_rows']:,}, days: {result['daily_rows']:,} "
          f"({result['first_day']:%Y-%m-%d} .. {result['last_day']:%Y-%m-%d})")
    mark_done("features", days=result["daily_rows"])


def step_external() -> None:
    from src.external import download_all
    status = download_all()
    if not any(s.startswith("ok") for s in status.values()):
        print("  No external data - the correlation analysis will use Bitcoin data only.")
    mark_done("external", **status)


def step_database() -> None:
    from src.database import build_database
    counts = build_database()
    print(f"\n  tables: {counts}")
    mark_done("database", **counts)


def ask_date(prompt: str, default: str) -> str:
    """Ask for a date until the answer is valid (empty answer keeps the default)."""
    while True:
        answer = input(f"  {prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return date.fromisoformat(answer).isoformat()
        except ValueError:
            print("  Please type a date as YYYY-MM-DD, e.g. 2024-03-15.")


def step_queries() -> None:
    from src.database import load_queries, run_query
    queries = load_queries()
    names = list(queries)
    while True:
        print()
        for i, name in enumerate(names, start=1):
            print(f"  {i:>2}  {name:<26} {queries[name]['description']}")
        choice = input("\n  Query number (a = all, Enter = back): ").strip().lower()
        if not choice:
            return
        if choice == "a":
            selected = names
        else:
            try:
                selected = [names[int(choice) - 1]]
            except (ValueError, IndexError):
                print("  No such query.")
                continue
        start = ask_date("period start", config.MODEL_START)
        end = ask_date("period end", "9999-12-31")
        for name in selected:
            headline(f"{name}: {queries[name]['description']}")
            started = time.perf_counter()
            result = run_query(name, {"start": start, "end": end})
            print_table(result, max_rows=15)
            print(f"  ({len(result)} rows, {1000 * (time.perf_counter() - started):.0f} ms)")


def step_eda() -> None:
    from src.eda import run_eda
    result = run_eda()
    print()
    print_table(result["stationarity"][["adf_p_value", "kpss_p_value", "conclusion"]])
    mark_done("eda")


def step_models() -> None:
    from src.database import store_results
    from src.experiment import run_experiment
    result = run_experiment()
    if config.DB_PATH.exists():
        store_results()
    best = result["summary"]["best_model"]
    print(f"\n  forecasts for {result['next_day']['date'].iloc[0]}:")
    print_table(result["next_day"][["model", "forecast_vol"]])
    mark_done("models", best_model=best)


def step_benchmarks() -> None:
    from src.benchmark import run_benchmarks
    run_benchmarks()
    mark_done("benchmarks")


def step_figures() -> None:
    from src import charts
    from src.experiment import load_predictions
    from src.features import load_daily
    from src.plots import make_all
    make_all()
    daily = load_daily()
    charts.price_chart(daily).write_html(config.FIGURES_DIR / "price_history.html", include_plotlyjs="cdn")
    try:
        predictions = load_predictions()
        intervals_path = config.TABLES_DIR / "test_intervals.csv"
        intervals = (pd.read_csv(intervals_path, index_col="date", parse_dates=["date"])
                     if intervals_path.exists() else None)
        charts.forecast_chart(predictions, ["LightGBM", "HAR-RV", "Naive"], intervals=intervals).write_html(
            config.FIGURES_DIR / "forecasts.html", include_plotlyjs="cdn")
        print("  reports/figures/price_history.html, reports/figures/forecasts.html (interactive)")
    except FileNotFoundError:
        print("  reports/figures/price_history.html (interactive); forecasts need step 8")
    mark_done("figures")


def step_lookup() -> None:
    """Show the actual volatility and every model's forecast for one date."""
    from src.database import run_sql
    from src.utils import require
    require(config.DB_PATH, "5 - build the SQLite database")
    span = run_sql("SELECT MIN(date) AS first, MAX(date) AS last FROM forecast")
    first, last = span.iloc[0]
    if first is None:
        print("  No forecasts yet. Run step 8 first.")
        return
    print(f"  Forecasts exist for {first} .. {last} (the last date is the forecast for tomorrow).")
    while True:
        day = input("  Date (YYYY-MM-DD, Enter = back): ").strip()
        if not day:
            return
        try:
            day = date.fromisoformat(day).isoformat()
        except ValueError:
            print("  Not a valid date.")
            continue
        if not first <= day <= last:
            print(f"  Choose a date between {first} and {last}.")
            continue
        result = run_sql("""
            SELECT m.name AS model, ROUND(f.predicted_vol, 3) AS forecast_pct,
                   ROUND(d.realized_vol, 3) AS actual_pct,
                   ROUND(ABS(f.predicted_vol - d.realized_vol), 3) AS abs_error
            FROM forecast f JOIN model m ON m.model_id = f.model_id
            LEFT JOIN candle_daily d ON d.date = f.date
            WHERE f.date = :day ORDER BY abs_error, f.predicted_vol""", {"day": day})
        print_table(result)
        if result["actual_pct"].isna().all():
            print("  No actual value yet: this is a forecast for the future.")


PIPELINE = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def step_all() -> None:
    started = time.perf_counter()
    for number, key in enumerate(PIPELINE, start=1):
        title, hint, func, _ = STEPS[key]
        headline(f"[{number}/{len(PIPELINE)}] {title}: {hint}")
        func()
    print(f"\n  Pipeline finished in {time.perf_counter() - started:.0f} s.")


def step_dashboard() -> None:
    from src.launcher import launch_dashboard
    from src.utils import require
    require(config.DAILY_PATH, "13 - run everything")
    require(config.DB_PATH, "13 - run everything")
    print("  Starting the dashboard; press Ctrl+C to come back to this menu.")
    launch_dashboard()


STEPS = {   # key: (title, what it does, function, state name for the "done" mark)
    "1": ("Load the raw CSV", "1 M-row chunks -> Parquet by year", step_convert, "convert"),
    "2": ("Check data quality", "gaps, duplicates, bad ticks", step_audit, "audit"),
    "3": ("Measure volatility", "daily realised volatility from 5-min returns", step_features, "features"),
    "4": ("Download market data", "VIX, S&P 500, dollar index (optional)", step_external, "external"),
    "5": ("Build the database", "SQLite, 8 tables", step_database, "database"),
    "6": ("Run SQL queries", "15 queries, choose a period", step_queries, None),
    "7": ("Explore the data", "statistics, seasonality, hypothesis tests", step_eda, "eda"),
    "8": ("Train and compare models", "7 models, cross-validation, hold-out test", step_models, "models"),
    "9": ("Speed and memory tests", "CSV vs Parquet, pandas vs SQLite", step_benchmarks, "benchmarks"),
    "10": ("Draw the figures", "18 report figures + 2 interactive charts", step_figures, "figures"),
    "11": ("Forecasts for a date", "what each model predicted vs reality", step_lookup, None),
    "12": ("Open the dashboard", "interactive charts in the browser", step_dashboard, None),
    "13": ("Run everything", "steps 1-5 and 7-10, about 1-2 minutes", step_all, None),
}
SECTIONS = [("Data", ["1", "2", "3", "4"]), ("Database", ["5", "6"]),
            ("Analysis and models", ["7", "8", "9"]), ("Results", ["10", "11", "12"]),
            ("All at once", ["13"])]

# --- a little colour when running in a real terminal (never in pipes or files) ----------------
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
GREEN, YELLOW, RED, DIM, SECTION = "32", "33", "31", "2", "1;36"


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def confirm(question: str) -> bool:
    return input(f"{question} [Y/n]: ").strip().lower() in ("", "y", "yes")


def status_line() -> str:
    """Is the dataset there, is the database built, are the models trained?"""
    from src.loader import find_raw_csv
    from src.utils import load_json
    try:
        dataset = paint("found", GREEN) + f" ({find_raw_csv().name})"
    except FileNotFoundError:
        dataset = paint("missing", RED) + " (see README: How to run)"
    database = paint("ready", GREEN) if config.DB_PATH.exists() else paint("not built", YELLOW)
    experiment = config.REPORTS_DIR / "experiment.json"
    models = (paint("trained", GREEN) + f" (best: {load_json(experiment)['best_model']})"
              if experiment.exists() else paint("not trained", YELLOW))
    return f"  Dataset: {dataset}   Database: {database}   Models: {models}"


def run_step(key: str) -> None:
    title, _, func, _ = STEPS[key]
    started = time.perf_counter()
    try:
        func()
        print(paint(f"\n  Done in {time.perf_counter() - started:.1f} s.", GREEN))
    except FileNotFoundError as error:
        print(paint(f"\n  Missing input: {error}", YELLOW))
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except Exception as error:          # keep the menu alive, show what went wrong
        print(paint(f"\n  Step '{title}' failed: {error}", RED))
        traceback.print_exc()


def menu() -> None:
    if USE_COLOR and os.name == "nt":
        os.system("")                   # switch on ANSI colours in the Windows console
    while True:
        state = load_state()
        headline("Bitcoin Volatility Lab")
        print(status_line())
        for section, keys in SECTIONS:
            print("\n  " + paint(section.upper(), SECTION))
            for key in keys:
                title, hint, _, state_key = STEPS[key]
                finished = state.get(state_key, {}).get("finished_at", "") if state_key else ""
                mark = paint(f"done {finished[11:16]}", GREEN) if finished else ""
                print(f"   {key:>2}  {title:<25} {paint(f'{hint:<46}', DIM)} {mark}")
        print("\n    0  Exit")
        choice = input("\n  Type a number and press Enter: ").strip().lstrip("﻿")   # ignore a stray BOM
        if not choice:
            continue
        if choice == "0":
            print("  Bye.")
            return
        if choice not in STEPS:
            print(paint("  Unknown option - type one of the numbers above.", YELLOW))
            continue
        if choice == "13" and not confirm("  This runs all steps and takes about 1-2 minutes. Continue?"):
            continue
        run_step(choice)
        if sys.stdin.isatty():
            input(paint("\n  Press Enter to return to the menu...", DIM))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.ensure_dirs()
    pd.set_option("display.width", 160)
    parser = argparse.ArgumentParser(description="Bitcoin volatility analysis and forecasting")
    parser.add_argument("--all", action="store_true", help="run the full pipeline")
    parser.add_argument("--step", choices=list(STEPS), help="run a single step")
    args = parser.parse_args()
    if args.all:
        step_all()
    elif args.step:
        run_step(args.step)
    else:
        menu()


if __name__ == "__main__":
    main()
