@echo off
setlocal enabledelayedexpansion
title PoultryScan AI — 5-Class Diagnostic & Multi-Chicken Screening Server

echo =====================================================================
echo       PoultryScan AI — Deep Learning Poultry Health System
echo =====================================================================
echo.

:: 1. Kill any stale process holding port 5000 (from a previous abrupt close)
echo [INFO] Checking for stale processes on port 5000...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    echo [INFO] Found stale process PID %%p on port 5000. Releasing it...
    taskkill /PID %%p /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)
timeout /t 2 /nobreak >nul

:: 2. Resolve Python Executable
set "PYTHON_EXE="
if exist "C:\Users\sneha\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\Users\sneha\anaconda3\python.exe"
    echo [ENV] Found Anaconda Python: C:\Users\sneha\anaconda3\python.exe
) else if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        echo [ENV] Found Active Conda Python: %CONDA_PREFIX%\python.exe
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('where python') do (
            if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
        )
        echo [ENV] Found System Python on PATH: !PYTHON_EXE!
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] No Python interpreter found!
    echo Please install Python 3.10+ or Anaconda, and ensure it is added to your PATH.
    echo Download: https://www.python.org/downloads/ or https://www.anaconda.com/
    echo.
    pause
    exit /b 1
)

:: 3. Verify Required Core Dependencies
echo [CHECK] Checking core libraries (Flask, TensorFlow, Keras, PyTorch, Torchvision)...
"!PYTHON_EXE!" -c "import flask, tensorflow, keras, torch, torchvision, PIL; print('[CHECK] All core dependencies verified successfully.')" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] One or more required packages may be missing or incompatible.
    echo Attempting to launch anyway... If it fails, run:
    echo   pip install -r requirements.txt
    echo.
)

:: 4. Launch browser after a brief delay in the background
echo [INFO] Starting PoultryScan AI Web Server...
echo [INFO] Your browser will open automatically at http://localhost:5000
echo [INFO] To stop the server at any time, press Ctrl+C in this window.
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:5000"

:: 5. Set OpenMP guard and run Flask server
set "KMP_DUPLICATE_LIB_OK=TRUE"
"!PYTHON_EXE!" app.py

echo.
echo Server stopped.
pause
