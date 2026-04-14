@echo off
title Surveillance System Dashboard
echo ========================================================
echo Startup Sequence Activated
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found!
    echo Please install Python 3.9+ from python.org
    echo IMPORTANT: Make sure to check the box "Add Python to PATH" during installation!
    pause
    exit /b
)

if not exist "venv_cuda" (
    echo [INFO] Virtual environment missing. Creating a fresh one...
    python -m venv venv_cuda
)

echo [INFO] Activating environment...
call venv_cuda\Scripts\activate.bat

echo [INFO] Ensuring all AI dependencies are installed... (This might take a minute on the first run)
pip install -r requirements.txt

echo.
echo ========================================================
echo Booting Application...
echo ========================================================
python app.py

pause
