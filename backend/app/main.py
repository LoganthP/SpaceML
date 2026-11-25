"""FastAPI entrypoint shared by local/Docker and Vercel."""
from spaceml.config import IS_VERCEL

if IS_VERCEL:
    from .vercel_app import app
else:
    from .local_main import *  # noqa: F401,F403 - preserve the local entrypoint
