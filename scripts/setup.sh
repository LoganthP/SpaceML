#!/usr/bin/env bash
# One-time setup on macOS/Linux: venv, backend + frontend deps, telemetry data, sample files, checks.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v python3 >/dev/null || { echo "Python 3.10-3.12 is required"; exit 1; }
command -v npm >/dev/null || { echo "Node.js 18+ (npm) is required: https://nodejs.org"; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ "$(uname)" == "Linux" ]] && ! command -v nvidia-smi >/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu   # avoids a multi-GB CUDA download
fi
pip install -r backend/requirements.txt
(cd backend && python -m spaceml.telemetry.data download && python -m spaceml.workbench.samples telemetry)
(cd frontend && npm install)
(cd backend && python -m spaceml.doctor) || true
echo
echo "Setup complete. Start everything with: scripts/dev.sh   then open http://localhost:5173"
