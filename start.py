"""One-click start: prepare whatever is missing, then open the dashboard.

    python start.py                  prepare data and models if needed, open the dashboard
    python start.py --rebuild        recompute everything from the raw CSV first
    python start.py --no-dashboard   only prepare, do not open the dashboard

On Windows, double-click start.bat; on macOS / Linux run ./start.sh
(they also create the Python environment on the first run).
"""
from __future__ import annotations

import argparse
import os
import sys

MIN_PYTHON = (3, 11)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        print(f"  Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed, this is "
              f"{sys.version.split()[0]}. Download: https://www.python.org/downloads/")
        sys.exit(1)


def explain_missing_data(raw_dir) -> None:
    print("\n  The dataset is missing. To add it:")
    print("    1. Download 'Bitcoin Historical Data' from Kaggle:")
    print("       https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data")
    print("    2. Unzip it and put btcusd_1-min_data.csv into this folder:")
    print(f"       {raw_dir}")
    print("    3. Start again.\n")
    if sys.platform == "win32" and sys.stdin.isatty():
        os.startfile(raw_dir)            # a person started it: open the folder in Explorer


def main() -> None:
    check_python()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Prepare the project and open the dashboard")
    parser.add_argument("--rebuild", action="store_true", help="recompute everything from the raw CSV")
    parser.add_argument("--no-dashboard", action="store_true", help="do not open the dashboard")
    args = parser.parse_args()

    from src import config
    from src.loader import find_raw_csv
    config.ensure_dirs()
    print("\n  Bitcoin Volatility Lab")
    try:
        csv = find_raw_csv()
    except FileNotFoundError:
        explain_missing_data(config.RAW_DIR)
        sys.exit(1)
    print(f"  Dataset found: {csv.name} ({csv.stat().st_size / 2**20:,.0f} MB)")

    results = [config.DAILY_PATH, config.DB_PATH, config.TABLES_DIR / "test_metrics.csv",
               config.FIGURES_DIR / "15_pipeline.png"]
    if args.rebuild or not all(path.exists() for path in results):
        print("  Preparing data, database, models and figures - about 1-2 minutes, only needed once.\n")
        from main import step_all
        step_all()
    else:
        print("  Data, database and models are ready (use --rebuild to recompute them).")

    if not args.no_dashboard:
        from src.launcher import launch_dashboard
        print("  Starting the dashboard...")
        launch_dashboard()


if __name__ == "__main__":
    main()
