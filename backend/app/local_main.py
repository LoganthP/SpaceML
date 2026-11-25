"""Space ML Mission Control - FastAPI entrypoint.

    uvicorn app.main:app --reload --port 8000      (run from backend/)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from spaceml.config import REPO_ROOT, ensure_dirs, IS_MANAGED_WORKER

from . import handlers  # noqa: F401  (registers module handlers with the worker)
from . import jobs, registry
from .db import init_db
from .routers import jobs as jobs_router
from .routers import catalog, live, missions, modules, overview, reviews, runs, setup, spaceweather, traverse, workbench

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_dirs()
    init_db()
    jobs.init()
    try:
        from . import starter
        result = starter.install()
        if IS_MANAGED_WORKER and result.get("errors"):
            raise RuntimeError("Bundled starter assets failed verification")
    except Exception:  # noqa: BLE001 - a broken pack must never stop the API
        logging.getLogger("starter").exception("Starter pack install failed")
        if IS_MANAGED_WORKER:
            raise
    if IS_MANAGED_WORKER:
        from .managed_worker import initialize
        initialize()
    registry.scan()
    if IS_MANAGED_WORKER:
        from .managed_worker import recover
        recover()
    jobs.bootstrap()

    def _warm() -> None:  # pre-load data summaries so the first page view is instant
        try:
            overview.datasets()
            catalog.catalog()
        except Exception:  # noqa: BLE001 - warming is best-effort
            logging.getLogger("warmup").debug("warm-up skipped", exc_info=True)
        try:
            from . import autoinstall
            autoinstall.run()                 # queue downloads of missing mission data when online
        except Exception:  # noqa: BLE001
            logging.getLogger("autoinstall").warning("auto-install failed", exc_info=True)
    import threading
    threading.Thread(target=_warm, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="Space ML Mission Control", version="3.1.10", lifespan=lifespan)
from .auth import install as install_auth
install_auth(app)
if IS_MANAGED_WORKER:
    from .managed_security import install as install_private_runtime
    install_private_runtime(app)

origins = os.environ.get("SPACEML_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(overview.router)
app.include_router(runs.router)
app.include_router(missions.router)
app.include_router(modules.router)
app.include_router(jobs_router.router)
app.include_router(workbench.router)
app.include_router(spaceweather.router)
app.include_router(reviews.router)
app.include_router(traverse.router)
app.include_router(setup.router)
app.include_router(catalog.router)
app.include_router(live.router)


@app.get("/api/starter", tags=["system"])
def starter_status():
    from . import starter
    return starter.status()

# Optional: serve the built frontend from the same port (frontend/dist).
DIST = Path(os.environ.get("SPACEML_FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))
class _ImmutableAssets(StaticFiles):
    """Hashed build assets never change: let the browser keep them for a year."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


if (DIST / "index.html").exists():
    app.mount("/assets", _ImmutableAssets(directory=DIST / "assets"), name="assets")
    NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        from fastapi import HTTPException
        if path.startswith("api/") or path == "api":
            raise HTTPException(404, "Not found")
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        # index.html must never be cached: after an update it has to point at the new asset files,
        # otherwise the browser asks for old ones that no longer exist (blank page)
        return FileResponse(DIST / "index.html", headers=NO_CACHE)
