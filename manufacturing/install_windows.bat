@echo off
setlocal
title ACT-Africa 2026 Bearing Challenge Setup

cd /d "%~dp0"

echo ============================================================
echo ACT-Africa 2026 Bearing Challenge - Windows setup
echo ============================================================
echo.

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY=py"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PY=python"
    ) else (
        echo ERROR: Python was not found.
        echo Install Python 3.11 from python.org, then run this file again.
        echo During installation, enable "Add python.exe to PATH" if offered.
        pause
        exit /b 1
    )
)

echo Using Python:
%PY% --version
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo Installing required packages from requirements.txt ...
%PY% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo Running the environment check ...
%PY% check_environment.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup is not READY yet. Read the FAIL line above or ask a facilitator.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo READY
 echo Launch the workshop notebook with:
 echo     %PY% -m notebook starter.ipynb
 echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo Setup failed. Ask a facilitator if you cannot resolve the message above.
pause
exit /b 1
