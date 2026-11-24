# Starts the API (port 8000) in a new window and the dashboard (port 5173) here.
Set-Location (Join-Path $PSScriptRoot "..")
$root = (Get-Location).Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --reload-dir spaceml --port 8000"
Set-Location frontend
npm run dev
