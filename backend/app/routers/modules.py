"""Input catalogs for the three modules: channels, Kepler TCEs, Mars samples and uploads."""
from __future__ import annotations

import io
import uuid

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from spaceml.common.utils import clean_floats, read_json, write_json
from spaceml.config import UPLOADS_DIR

from .. import registry

router = APIRouter(prefix="/api", tags=["modules"])


# ------------------------------------------------------------------ telemetry
@router.get("/telemetry/channels")
def telemetry_channels(mission: str | None = Query(None, pattern="^(SMAP|MSL)$")):
    from spaceml.telemetry import data
    if not data.LABELS_PATH.exists():
        raise HTTPException(409, "Telemetry data not downloaded yet. Run: task:telemetry.download")
    from pathlib import Path
    trained = {f.stem for v in registry.versions("telemetry") for f in Path(v["path"]).glob("*.pt")}
    out = []
    for c in sorted(data.load_channels().values(), key=lambda c: (c.spacecraft, c.chan_id)):
        if mission and c.spacecraft != mission:
            continue
        out.append({**c.to_dict(), "has_model": c.chan_id in trained,
                    "downloaded": (data.TEST_DIR / f"{c.chan_id}.npy").exists()})
    return out


# ------------------------------------------------------------------ exoplanet
@router.get("/exoplanet/tces")
def exoplanet_tces(q: str = "", label: str | None = Query(None, pattern="^(PC|AFP|NTP|UNK)$"),
                   cached_only: bool = False, limit: int = Query(50, ge=1, le=500)):
    from spaceml.exoplanet import data
    try:
        df = data.tce_table()
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    if q.strip():
        digits = "".join(ch for ch in q if ch.isdigit())
        if not digits:
            return []
        df = df[df.kepid.astype(str).str.startswith(digits)]
    if label:
        df = df[df.av_training_set == label]
    cached = data.cached_view_ids()
    if cached_only:
        df = df[[(int(k), int(p)) in cached for k, p in zip(df.kepid, df.tce_plnt_num)]]
    df = df.sort_values(["kepid", "tce_plnt_num"]).head(limit)
    return [{"kepid": int(r.kepid), "tce_plnt_num": int(r.tce_plnt_num), "period_days": float(r.tce_period),
             "duration_hours": float(r.tce_duration), "depth_ppm": float(r.tce_depth),
             "label": r.av_training_set, "cached": (int(r.kepid), int(r.tce_plnt_num)) in cached,
             "name": data.FAMOUS.get(int(r.kepid))} for r in df.itertuples()]


@router.get("/exoplanet/featured")
def exoplanet_featured():
    """Kepler-90 plus held-out test examples: confident planets, caught and missed false positives."""
    from spaceml.exoplanet import data
    act = registry.active("exoplanet")
    preds = read_json(registry.active_dir("exoplanet") / "test_predictions.json", []) if act else []
    pc = sorted([p for p in preds if p["planet"] == 1], key=lambda p: -p["cnn"])[:3]
    fp_caught = sorted([p for p in preds if p["av_training_set"] == "AFP"], key=lambda p: p["cnn"])[:3]
    fp_missed = sorted([p for p in preds if p["planet"] == 0 and p["cnn"] >= 0.5], key=lambda p: -p["cnn"])[:3]
    k90 = []
    if data.TCE_PATH.exists():
        df = data.tce_table()
        k90 = [{"kepid": int(r.kepid), "tce_plnt_num": int(r.tce_plnt_num), "label": r.av_training_set,
                "period_days": float(r.tce_period)} for r in df[df.kepid == data.KEPLER_90].itertuples()]
    return {"kepler90": k90, "test_planets": pc, "false_positives_caught": fp_caught,
            "false_positives_missed": fp_missed}


@router.get("/exoplanet/views/{kepid}/{plnt}")
def exoplanet_view(kepid: int, plnt: int):
    """Cached local view only (for thumbnails) - never triggers a download."""
    from spaceml.exoplanet import data
    if not data.view_path(kepid, plnt).exists():
        raise HTTPException(404, "View not cached")
    v = data.get_views(kepid, plnt, allow_download=False)
    return {"local": clean_floats(v["local"], 3), "global": clean_floats(v["global"][::4], 3)}


# ------------------------------------------------------------------ mars
@router.get("/mars/samples")
def mars_samples():
    from spaceml.mars import data
    return data.list_samples()


@router.get("/mars/samples/{sample_id}/{kind}")
def mars_sample_file(sample_id: str, kind: str):
    from spaceml.mars import data
    if kind not in ("image", "label", "ignore", "thumb"):
        raise HTTPException(404, "Unknown file kind")
    for s in data.list_samples():
        if s["id"] == sample_id:
            p = data.SAMPLES_DIR / s[kind]
            media = "image/jpeg" if p.suffix == ".jpg" else "image/png"
            return FileResponse(p, media_type=media, headers={"Cache-Control": "max-age=86400"})
    raise HTTPException(404, "Sample not found")


MAX_UPLOAD = 15 * 1024 * 1024


@router.post("/mars/uploads")
async def mars_upload(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 15 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, "Unsupported image - upload a PNG or JPEG rover image") from e
    if min(img.size) < 64:
        raise HTTPException(422, "Image is too small (minimum 64 px)")
    uid = uuid.uuid4().hex[:16]
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    gray = img.convert("L")
    if max(gray.size) > 1536:
        gray.thumbnail((1536, 1536))
    gray.save(UPLOADS_DIR / f"{uid}.png")
    arr = np.asarray(gray)
    meta = {"upload_id": uid, "filename": (file.filename or "image")[:80], "width": gray.width,
            "height": gray.height, "mean_brightness": float(arr.mean())}
    write_json(UPLOADS_DIR / f"{uid}.json", meta)
    return meta


@router.get("/mars/uploads/{upload_id}/image")
def mars_upload_image(upload_id: str):
    if not upload_id.isalnum():
        raise HTTPException(404, "Not found")
    p = UPLOADS_DIR / f"{upload_id}.png"
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")


# ------------------------------------------------------------------ exoplanet: sky + uploaded datasets
@router.get("/exoplanet/catalog")
def exoplanet_catalog_info():
    from spaceml.exoplanet import data
    return data.status()


@router.get("/exoplanet/sky")
def exoplanet_sky(limit: int = Query(16000, ge=100, le=20000)):
    from spaceml.exoplanet import data, infer
    if not data.TCE_PATH.exists():
        return {"available": False, "stars": [], "reason": "catalog not downloaded"}
    out = infer.sky_catalog(limit)
    if not out["available"]:
        out["reason"] = "catalog has no sky positions - re-download it (Data & Training → Kepler catalog)"
    return out


MAX_DATASET = 1024 * 1024 * 1024  # 1 GB


@router.post("/exoplanet/datasets")
async def exoplanet_dataset_upload(file: UploadFile = File(...), combine: bool = False):
    """Upload a light curve file or an archive of files; parsed immediately into targets."""
    from starlette.concurrency import run_in_threadpool

    from spaceml.exoplanet import ingest, uploads
    name = (file.filename or "upload").replace("\\", "/").split("/")[-1][:120] or "upload"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS_DIR / f"ds_{uuid.uuid4().hex}_{name}"
    size = 0
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_DATASET:
                    raise HTTPException(413, "Upload is larger than 1 GB - split it into several archives")
                fh.write(chunk)
        if size == 0:
            raise HTTPException(422, "The uploaded file is empty")
        try:
            return await run_in_threadpool(uploads.create, tmp, name, combine)
        except ingest.IngestError as e:
            raise HTTPException(422, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)


@router.get("/exoplanet/datasets")
def exoplanet_datasets():
    from spaceml.exoplanet import uploads
    return uploads.list_all()


@router.get("/exoplanet/datasets/{dataset_id}")
def exoplanet_dataset(dataset_id: str):
    from spaceml.exoplanet import uploads
    try:
        return uploads.get(dataset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, "Dataset not found") from e


@router.delete("/exoplanet/datasets/{dataset_id}")
def exoplanet_dataset_delete(dataset_id: str):
    from spaceml.exoplanet import uploads
    try:
        uploads.get(dataset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, "Dataset not found") from e
    uploads.delete(dataset_id)
    return {"deleted": dataset_id}


# ------------------------------------------------------------------ telemetry preview for the overview
@router.get("/telemetry/preview")
def telemetry_preview(chan_id: str | None = Query(None, pattern=r"^[A-Z]-\d{1,2}$"), points: int = Query(600, ge=100, le=3000)):
    """Real test telemetry of one channel (downsampled) with its expert-labeled anomalies."""
    from spaceml.telemetry import data
    if not data.is_usable():
        raise HTTPException(409, "Telemetry data not downloaded")
    chans = data.load_channels()
    if chan_id is None:
        act = registry.active("telemetry")
        trained = sorted(p.stem for p in registry.active_dir("telemetry").glob("*.pt")) if act else []
        trained = [c for c in trained if data.has_channel(c)]
        pick = [c for c in ("T-8", "C-1", "M-6") if c in trained] or trained or sorted(c for c in chans if data.has_channel(c))
        chan_id = pick[0]
    if chan_id not in chans:
        raise HTTPException(404, "Unknown channel")
    if not data.has_channel(chan_id):
        raise HTTPException(409, f"Telemetry for {chan_id} is not downloaded")
    _, test = data.load_arrays(chan_id)
    y = test[:, 0]
    step = max(1, len(y) // points)
    return {"chan_id": chan_id, "spacecraft": chans[chan_id].spacecraft, "step": step, "n_steps": len(y),
            "values": clean_floats(y[::step], 4),
            "labeled": [{"start": a, "end": b, "class": c} for (a, b), c in zip(chans[chan_id].sequences, chans[chan_id].classes)]}
