@echo off
rem One-click start for Windows: double-click this file.
cd /d "%~dp0"
title Bitcoin Volatility Lab
if exist ".venv\Scripts\python.exe" goto run
echo First start: creating the Python environment and installing libraries, 3-5 minutes...
where py >nul 2>nul
if %errorlevel%==0 (py -3 -m venv .venv) else (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" goto nopython
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto installfailed
:run
".venv\Scripts\python.exe" start.py %*
if errorlevel 1 pause
exit /b
:nopython
echo Python 3.11 or newer is required: https://www.python.org/downloads/
pause
exit /b 1
:installfailed
echo Installing the libraries failed - check the internet connection and try again.
pause
exit /b 1
