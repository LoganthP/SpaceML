#!/usr/bin/env bash
# Starts the API (port 8000) and the dashboard (port 5173). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
(cd backend && uvicorn app.main:app --reload --reload-dir app --reload-dir spaceml --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
cd frontend && npm run dev
