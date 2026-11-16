@echo off
rem Rebuilds everything from scratch: tests -> full study -> figures/report/dashboard -> opens dashboard.
rem Run from the project folder with the virtual environment activated.
set PYTHONUTF8=1
python -m pytest -q || exit /b 1
python run_experiments.py || exit /b 1
python -m src.reporting.build_all || exit /b 1
start "" "%~dp0dashboard\index.html"
