#!/usr/bin/env bash
# One-click start for macOS / Linux: ./start.sh  (first run: chmod +x start.sh)
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First start: creating the Python environment and installing libraries (3-5 minutes)..."
  python3 -m venv .venv || { echo "Python 3.11 or newer is required: https://www.python.org/downloads/"; exit 1; }
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing the libraries failed."; exit 1; }
fi
exec .venv/bin/python start.py "$@"
