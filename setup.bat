@echo off
cd /d "%~dp0"
echo === Raqeeb setup (run once) ===
set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" where python >nul 2>nul && set PY=python
if "%PY%"=="" (
  echo Python was not found. Install Python 3.11+ from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
%PY% --version
if not exist .venv %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Installing libraries failed. Check your internet connection and try again.
  pause
  exit /b 1
)
python train.py
echo.
echo Setup finished. Double-click run.bat to open Raqeeb.
pause
