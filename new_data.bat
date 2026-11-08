@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set SEED=42
set /p SEED=Random seed for the new practice data (press Enter for 42): 
python generate_data.py --out data --seed %SEED%
python train.py
echo.
echo New data made and models retrained. Refresh the browser page (or click Retrain models).
pause
