@echo off
setlocal
set ROOT=C:\Users\stuff\nfl_export
set CONDA=C:\Users\stuff\miniconda3\condabin\conda.bat
set LOG=%ROOT%\weekly_update.log

echo [%date% %time%] === Weekly update start === > "%LOG%"

echo [1/3] Fetch latest weekly data... | tee -a "%LOG%"
call "%CONDA%" activate dfsfetch
python "%ROOT%\update_weekly_data.py" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] Fetch failed. See weekly_update.log. & echo Fetch failed >> "%LOG%"
  call "%CONDA%" deactivate
  pause
  exit /b 1
)
call "%CONDA%" deactivate

echo [2/3] Refit correlation table... | tee -a "%LOG%"
call "%CONDA%" activate dfsopt
python "%ROOT%\correlation_fitter.py" --data-root "%ROOT%" --out "%ROOT%\correlation_table.json" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] Correlation fit failed. See weekly_update.log. & echo Corr fit failed >> "%LOG%"
  call "%CONDA%" deactivate
  pause
  exit /b 1
)

echo [3/3] Retrain / recalibrate models... | tee -a "%LOG%"
python "%ROOT%\train_models.py" --data-root "%ROOT%" --out-dir "%ROOT%\models" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [WARN] Training failed; keeping last good models. See weekly_update.log. >> "%LOG%"
  rem We continue so you can still run with previous models + fresh correlations.
) else (
  echo Training complete. >> "%LOG%"
)

call "%CONDA%" deactivate

echo Done. Open Streamlit, run Simulator/Optimizer. | tee -a "%LOG%"
echo Log saved to %LOG%
pause
endlocal
