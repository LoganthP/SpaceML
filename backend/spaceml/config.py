"""Shared paths and settings for every pipeline and the API.

All locations can be overridden with environment variables, which is how the
Docker image points them at mounted volumes.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("SPACEML_DATA_DIR", REPO_ROOT / "data")).resolve()
MODELS_DIR = Path(os.environ.get("SPACEML_MODELS_DIR", REPO_ROOT / "models")).resolve()
STORAGE_DIR = Path(os.environ.get("SPACEML_STORAGE_DIR", REPO_ROOT / "storage")).resolve()

DB_PATH = STORAGE_DIR / "mission_control.db"
RUNS_DIR = STORAGE_DIR / "runs"
UPLOADS_DIR = STORAGE_DIR / "uploads"

TELEMETRY_DIR = DATA_DIR / "telemetry"
EXOPLANET_DIR = DATA_DIR / "exoplanet"
MARS_DIR = DATA_DIR / "mars"

SEED = 42


def ensure_dirs() -> None:
    for p in (DATA_DIR, MODELS_DIR, STORAGE_DIR, RUNS_DIR, UPLOADS_DIR,
              TELEMETRY_DIR, EXOPLANET_DIR, MARS_DIR):
        p.mkdir(parents=True, exist_ok=True)
