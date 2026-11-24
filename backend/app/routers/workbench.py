"""Workbench: upload any supported dataset, preview it, pick a compatible analysis."""
from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from spaceml.common.utils import clean_floats
from spaceml.config import UPLOADS_DIR

router = APIRouter(prefix="/api/workbench", tags=["workbench"])
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB


def _store():
    from spaceml.workbench import store
    return store


@router.get("/analyses")
def analyses():
    from spaceml.workbench import analyses as a
    return a.catalog()


@router.get("/datasets")
def datasets():
    return _store().list_all()


@router.post("/datasets")
async def upload(file: UploadFile = File(...), kind: str = Form("auto")):
    store = _store()
    if kind not in ("auto", "timeseries", "images", "lightcurve", "samples"):
        raise HTTPException(422, "kind must be auto, timeseries, images or lightcurve")
    name = (file.filename or "upload").replace("\\", "/").split("/")[-1][:120] or "upload"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS_DIR / f"wb_{uuid.uuid4().hex}_{name}"
    size = 0
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Upload is larger than 1 GB - split it into several archives")
                fh.write(chunk)
        if size == 0:
            raise HTTPException(422, "The uploaded file is empty")
        try:
            return await run_in_threadpool(store.create, tmp, name, kind)
        except store.WorkbenchError as e:
            raise HTTPException(422, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)


@router.get("/datasets/{dataset_id}")
def dataset(dataset_id: str):
    from spaceml.workbench import analyses as a
    store = _store()
    try:
        m = store.get(dataset_id)
    except (FileNotFoundError, store.WorkbenchError) as e:
        raise HTTPException(404, "Dataset not found") from e
    from spaceml.workbench import profile
    try:
        prof = profile.classify(m)
    except Exception as e:  # noqa: BLE001 - classification is a convenience, never a blocker
        prof = {"category": "unknown", "label": m.get("kind", "dataset"), "confidence": 0, "reasons": [str(e)],
                "facts": {}, "recommended": None}
    from .. import registry
    prof["mars_model"] = registry.active("mars") is not None
    return {**m, "compatible": a.compatible(m), "profile": prof}


@router.delete("/datasets/{dataset_id}")
def delete(dataset_id: str):
    store = _store()
    try:
        store.get(dataset_id)
    except (FileNotFoundError, store.WorkbenchError) as e:
        raise HTTPException(404, "Dataset not found") from e
    store.delete(dataset_id)
    return {"deleted": dataset_id}


@router.get("/datasets/{dataset_id}/class-profile")
def class_profile(dataset_id: str, param: str = Query(..., max_length=120)):
    """Per-class median / IQR of one parameter over the window (labeled-sample datasets)."""
    from spaceml.workbench import mvts
    store = _store()
    try:
        return mvts.class_profile(dataset_id, param)
    except FileNotFoundError as e:
        raise HTTPException(404, "Dataset not found") from e
    except store.WorkbenchError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/datasets/{dataset_id}/series")
def series(dataset_id: str, table: str = Query(..., pattern=r"^t\d+$"), columns: str = Query(""),
           points: int = Query(1500, ge=100, le=5000)):
    """Downsampled preview of up to 8 numeric columns (for the column-mapping step)."""
    store = _store()
    try:
        entry, arrays = store.load_table(dataset_id, table)
    except (FileNotFoundError, store.WorkbenchError) as e:
        raise HTTPException(404, str(e)) from e
    names = [c for c in columns.split(",") if c][:8] or [c["name"] for c in entry["columns"]
                                                          if c["type"] == "numeric"][:4]
    step = max(1, entry["rows"] // points)
    out = {}
    for c in names:
        a = arrays.get(f"n::{c}")
        if a is not None:
            out[c] = clean_floats(a[::step], 6)
    return {"step": step, "rows": entry["rows"], "series": out}


@router.get("/datasets/{dataset_id}/images/{image_id}/{kind}")
def image_file(dataset_id: str, image_id: str, kind: str):
    store = _store()
    if kind not in ("thumb", "mask"):
        raise HTTPException(404, "Unknown file")
    try:
        e, _ = store.image_entry(dataset_id, image_id)
    except (FileNotFoundError, store.WorkbenchError) as ex:
        raise HTTPException(404, "Image not found") from ex
    name = f"{image_id}_thumb.png" if kind == "thumb" else e.get("mask")
    if not name:
        raise HTTPException(404, "No mask for this image")
    return FileResponse(store.dataset_dir(dataset_id) / name, media_type="image/png")


@router.get("/datasets/{dataset_id}/image-stats/{image_id}")
def image_stats(dataset_id: str, image_id: str):
    store = _store()
    try:
        e, arr, geo = store.load_image(dataset_id, image_id)
    except (FileNotFoundError, store.WorkbenchError) as ex:
        raise HTTPException(404, str(ex)) from ex
    bands = []
    for i, b in enumerate(arr[:12]):
        v = b[np.isfinite(b)]
        bands.append({"band": i + 1, "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
                      "mean": float(v.mean()) if v.size else None})
    return {**e, "geo": geo, "band_stats": bands}
