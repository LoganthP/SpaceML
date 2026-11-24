# One-time setup on Windows (PowerShell):  .\scripts\setup.ps1
# If scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.10-3.12 is required (python.org, tick 'Add to PATH')" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "Node.js 18+ is required (nodejs.org)" }
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Push-Location backend
..\.venv\Scripts\python.exe -m spaceml.telemetry.data download
..\.venv\Scripts\python.exe -m spaceml.workbench.samples telemetry
Pop-Location
Push-Location frontend; npm install; Pop-Location
Push-Location backend; ..\.venv\Scripts\python.exe -m spaceml.doctor; Pop-Location
Write-Host "`nSetup complete. Start everything with: .\scripts\dev.ps1   then open http://localhost:5173"
