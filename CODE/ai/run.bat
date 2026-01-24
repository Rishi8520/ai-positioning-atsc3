@echo off
REM ============================================================================
REM PPaaS AI System - Quick Start Batch Script
REM ============================================================================
REM This script sets up the environment and runs the complete workflow

echo.
echo ============================================================================
echo PPaaS AI SYSTEM - QUICK START
echo ============================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please create it first with: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call .venv\Scripts\Activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Check dependencies
echo.
echo [2/3] Checking dependencies...
python -c "import numpy, torch, sklearn, transformers; print('[OK] All dependencies installed')" 2>nul
if errorlevel 1 (
    echo [WARNING] Missing dependencies - installing...
    pip install --quiet numpy torch scikit-learn transformers
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

REM Run the main workflow
echo.
echo [3/3] Running PPaaS AI System...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] System execution failed - check ppaas_system.log for details
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo EXECUTION COMPLETE - CHECK ppaas_system.log FOR DETAILS
echo ============================================================================
echo.
pause
