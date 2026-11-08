@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Opening Raqeeb in your browser... (close this window to stop it)
streamlit run app.py
