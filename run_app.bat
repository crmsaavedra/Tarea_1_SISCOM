@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.11 or 3.12 was not found. Install 64-bit Python with Tcl/Tk.
    pause
    exit /b 1
  )
  set "PYTHON_CMD=python"
) else (
  set "PYTHON_CMD=py"
)
if not exist ".venv\Scripts\python.exe" (
  if "%PYTHON_CMD%"=="py" (
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
start "" ".venv\Scripts\pythonw.exe" app.py
