#!/usr/bin/env pwsh
<#
============================================================================
PPaaS AI System - Quick Start PowerShell Script
============================================================================
This script sets up the environment and runs the complete workflow
#>

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PPaaS AI SYSTEM - QUICK START" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "[ERROR] Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Cyan
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "[ERROR] Activation script not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    & $venvPath
} catch {
    Write-Host "[ERROR] Failed to activate virtual environment: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green

# Check dependencies
Write-Host ""
Write-Host "[2/3] Checking dependencies..." -ForegroundColor Cyan

$depCheck = python -c "import numpy, torch, sklearn, transformers; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Missing dependencies - installing..." -ForegroundColor Yellow
    pip install --quiet numpy torch scikit-learn transformers
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "[OK] All dependencies installed" -ForegroundColor Green
}

# Run the main workflow
Write-Host ""
Write-Host "[3/3] Running PPaaS AI System..." -ForegroundColor Cyan
Write-Host ""

python main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] System execution failed - check ppaas_system.log for details" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "EXECUTION COMPLETE - CHECK ppaas_system.log FOR DETAILS" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
