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
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed --name "SisCom Morse" --add-data "app.py;source" --add-data "requirements.txt;source" --add-data "requirements-build.txt;source" --add-data "README.md;source" --add-data "run_app.bat;source" --add-data "build_exe.bat;source" --add-data "arduino_morse_leds.ino;source" --add-data "siscom;source/siscom" --hidden-import matplotlib.backends.backend_tkagg --exclude-module pytest --exclude-module torch --exclude-module torchmetrics --exclude-module IPython --exclude-module jupyter --exclude-module pyarrow --exclude-module numba --exclude-module llvmlite --exclude-module openpyxl --exclude-module lxml --exclude-module cryptography --exclude-module fsspec app.py
if errorlevel 1 exit /b 1
echo.
echo EXE created at: dist\SisCom Morse.exe
echo Copy that one file to the computer where you will use the app.
