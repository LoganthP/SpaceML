"""Automatic setup status/control and model uploads (no terminal needed)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from spaceml.config import MODELS_DIR

from .. import autosetup, registry

router = APIRouter(prefix="/api", tags=["setup"])


class SetupRun(BaseModel):
    steps: list[str] = Field(default_factory=list, max_length=20)
    force: bool = True


@router.get("/setup")
def setup_status():
    return autosetup.status()


@router.post("/setup/run")
def setup_run(body: SetupRun | None = None):
    body = body or SetupRun()
    known = {s.id for s in autosetup.STEPS}
    bad = [s for s in body.steps if s not in known]
    if bad:
        raise HTTPException(422, f"Unknown setup step(s): {', '.join(bad)}")
    return autosetup.start(reason="manual", only=body.steps or None, force=body.force)


@router.post("/models/mars/upload")
async def upload_mars_model(file: UploadFile = File(...)):
    """Install the mars_model.zip produced by notebooks/mars_training.ipynb (Colab/Kaggle GPU)."""
    from spaceml.mars import install
    (MODELS_DIR / "mars").mkdir(parents=True, exist_ok=True)
    tmp = MODELS_DIR / "mars" / f".upload-{uuid.uuid4().hex}.zip"
    size = 0
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > install.MAX_ZIP:
                    raise HTTPException(413, "The zip is larger than 2 GB")
                fh.write(chunk)
        try:
            res = await run_in_threadpool(install.install_zip, tmp, file.filename or "upload")
        except install.InstallError as e:
            raise HTTPException(422, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)
    registry.scan()
    return res
