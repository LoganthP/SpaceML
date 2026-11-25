"""Shared paths and settings for every pipeline and the API.

All locations can be overridden with environment variables, which is how the
Docker image points them at mounted volumes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODE = os.environ.get("SMC_MODE", "local").strip().lower()
if MODE not in ("local", "vercel"):
    raise ValueError("SMC_MODE must be local or vercel")
IS_VERCEL = MODE == "vercel"

DATA_DIR = Path(os.environ.get("SPACEML_DATA_DIR", REPO_ROOT / "data")).resolve()
MODELS_DIR = Path(os.environ.get("SPACEML_MODELS_DIR", REPO_ROOT / "models")).resolve()
if IS_VERCEL:
    # Ignore persistent local paths in serverless mode. These directories are
    # created only if a live cache is used, never at import/cold start.
    _tmp = Path(tempfile.gettempdir()).resolve()
    STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", _tmp / "smc")).resolve()
    if STORAGE_DIR != _tmp and _tmp not in STORAGE_DIR.parents:
        raise ValueError("STORAGE_DIR must be under the temporary directory in Vercel mode")
else:
    STORAGE_DIR = Path(os.environ.get("SPACEML_STORAGE_DIR", REPO_ROOT / "storage")).resolve()

DB_PATH = STORAGE_DIR / "mission_control.db"
RUNS_DIR = STORAGE_DIR / "runs"
UPLOADS_DIR = STORAGE_DIR / "uploads"

TELEMETRY_DIR = DATA_DIR / "telemetry"
EXOPLANET_DIR = DATA_DIR / "exoplanet"
MARS_DIR = DATA_DIR / "mars"

SEED = 42


def ensure_dirs() -> None:
    if IS_VERCEL:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        return
    for p in (DATA_DIR, MODELS_DIR, STORAGE_DIR, RUNS_DIR, UPLOADS_DIR,
              TELEMETRY_DIR, EXOPLANET_DIR, MARS_DIR):
        p.mkdir(parents=True, exist_ok=True)
