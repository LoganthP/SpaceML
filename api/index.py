"""The one Python Function. Local/Docker uvicorn still imports app.main directly."""
import os
import sys
from pathlib import Path

os.environ.setdefault("SMC_MODE", "vercel")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.main import app  # noqa: E402,F401
