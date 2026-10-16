@echo off
title PoultryScan AI - Web Server
echo ========================================================
echo   Starting PoultryScan AI Deep Learning Web Server
echo ========================================================
echo.

:: Prevent OpenMP duplicate library crash
set KMP_DUPLICATE_LIB_OK=TRUE

:: Check if port 5000 is actively LISTENING and kill the process
echo [INFO] Checking for stale processes on port 5000...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    echo [INFO] Found stale process PID %%p holding port 5000. Killing it...
    taskkill /PID %%p /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

:: Small wait for port to be fully released
timeout /t 2 /nobreak >nul

:: Automatically launch browser after 5 seconds
echo [INFO] Launching browser at http://localhost:5000 in 5 seconds...
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:5000"

IF EXIST "C:\Users\sneha\anaconda3\python.exe" (
    echo Using Anaconda Python Environment [TensorFlow + Keras 3 + PyTorch]
    "C:\Users\sneha\anaconda3\python.exe" app.py
) ELSE (
    echo Using System Python Environment
    python app.py
)

pause
