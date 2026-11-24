"""Space Weather Explorer: OMNIWeb series, storm catalog, user annotations, Dst forecasts."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from spaceml.common.utils import now_iso

from .. import registry
from ..db import tx

router = APIRouter(prefix="/api/spaceweather", tags=["spaceweather"])
MAX_POINTS = 6000

SCHEMA = """CREATE TABLE IF NOT EXISTS sw_annotations (
    id TEXT PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL, label TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL)"""


def _ensure():
    with tx() as c:
        c.execute(SCHEMA)


@router.get("/status")
def status():
    from spaceml.spaceweather import data
    years = data.cached_years()
    act = registry.active("spaceweather")
    return {"cached_years": years, "default_years": data.DEFAULT_YEARS, "variables": data.META,
            "model": act["manifest"] if act else None}


@router.get("/series")
def series(start: str, end: str, vars: str = Query("V,Bz,N,Pdyn,Dst,Kp")):
    from spaceml.spaceweather import data
    names = [v for v in vars.split(",") if v in data.META][:10]
    try:
        s, e = data.ts(start), data.ts(end)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(422, f"Invalid date: {ex}") from ex
    if e <= s:
        raise HTTPException(422, "End must be after start")
    if (e - s).days > 366 * 3:
        raise HTTPException(422, "Choose a window of at most 3 years")
    try:
        df = data.load(s.isoformat(), e.isoformat())
    except FileNotFoundError as ex:
        raise HTTPException(409, str(ex)) from ex
    step = max(1, len(df) // MAX_POINTS)
    d = df.iloc[::step] if step > 1 else df
    if step > 1:  # keep the extremes of Dst / Bz visible when decimating
        agg = {c: ("min" if c in ("Dst", "Bz") else "max" if c in ("V", "Kp", "Pdyn", "N", "AE") else "mean") for c in names}
        d = df[names].resample(f"{step}h").agg(agg)
    return {"time": [t.isoformat() for t in d.index], "step_hours": step,
            "series": {c: [None if not np.isfinite(v) else round(float(v), 2) for v in d[c].to_numpy(float)] for c in names},
            "meta": {c: data.META[c] for c in names},
            "summary": {"min_dst": float(df["Dst"].min()) if df["Dst"].notna().any() else None,
                        "max_kp": float(df["Kp"].max()) if df["Kp"].notna().any() else None,
                        "max_v": float(df["V"].max()) if df["V"].notna().any() else None,
                        "min_bz": float(df["Bz"].min()) if df["Bz"].notna().any() else None,
                        "coverage": float(df["V"].notna().mean())}}


@router.get("/events")
def events():
    from spaceml.spaceweather import data
    _ensure()
    with tx() as c:
        notes = [dict(r) for r in c.execute("SELECT * FROM sw_annotations ORDER BY start")]
    return {"catalog": data.events_with_stats(), "annotations": notes}


class AnnotationIn(BaseModel):
    start: str
    end: str
    label: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


@router.post("/annotations")
def add_annotation(a: AnnotationIn):
    from spaceml.spaceweather import data
    try:
        s, e = data.ts(a.start), data.ts(a.end)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(422, f"Invalid date: {ex}") from ex
    if e < s:
        raise HTTPException(422, "End is before start")
    _ensure()
    rec = {"id": uuid.uuid4().hex[:10], "start": s.isoformat(), "end": e.isoformat(), "label": a.label,
           "note": a.note, "created_at": now_iso()}
    with tx() as c:
        c.execute("INSERT INTO sw_annotations VALUES (:id,:start,:end,:label,:note,:created_at)", rec)
    return rec


@router.delete("/annotations/{ann_id}")
def delete_annotation(ann_id: str):
    _ensure()
    with tx() as c:
        c.execute("DELETE FROM sw_annotations WHERE id=?", (ann_id,))
    return {"deleted": ann_id}


@router.get("/forecast")
def forecast(start: str, end: str, horizon: int = Query(6, ge=1, le=48)):
    from spaceml.spaceweather import data, forecast as fc
    try:
        vdir = registry.active_dir("spaceweather")
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    if (data.ts(end) - data.ts(start)) > pd.Timedelta(days=60):
        raise HTTPException(422, "Forecast windows are limited to 60 days")
    try:
        return fc.predict_window(vdir, start, end, horizon)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/quality")
def quality(start: str, end: str):
    """Data-quality check of a window before replaying or forecasting it."""
    from spaceml.common.quality import series_quality
    from spaceml.spaceweather import data
    try:
        s, e = data.ts(start), data.ts(end)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(422, f"Invalid date: {ex}") from ex
    if (e - s).days > 366 * 3:
        raise HTTPException(422, "Choose a window of at most 3 years")
    need = sorted({s.year, e.year})
    missing = [y for y in need if y not in data.cached_years()]
    if missing:
        return {"status": "source_unavailable" if _bootstrap_failed() else "not_downloaded", "missing_years": missing,
                "findings": [f"OMNI2 {', '.join(map(str, missing))} not downloaded yet"], "columns": [], "samples": 0,
                "provenance": data.provenance([y for y in need if y not in missing])}
    df = data.load(s.isoformat(), e.isoformat())
    q = series_quality(df, ["Dst", "V", "Bz", "N", "Pdyn", "Kp"], required=["Dst", "V", "Bz"],
                       continuous={"V", "N", "Pdyn"})
    q["provenance"] = data.provenance(need)
    q["dst_status"] = {y: data.dst_status(y) for y in need}
    return q


def _bootstrap_failed() -> bool:
    from .. import jobs
    last = next((j for j in jobs.list_jobs(50) if j["task"] in ("spaceweather.download", "spaceweather.setup")), None)
    return bool(last and (last["status"] == "source_unavailable"
                          or (last["status"] == "failed" and "source_unavailable" in (last.get("error") or ""))))


@router.get("/replays")
def replays(event_id: str | None = None, limit: int = Query(20, ge=1, le=100)):
    """Replay runs (bundled reference runs and runs computed here), newest first."""
    from ..worker import list_runs
    out = []
    for r in list_runs("spaceweather", 300):
        if event_id and (r.get("input_ref") or {}).get("event_id") != event_id:
            continue
        out.append({k: r.get(k) for k in ("id", "status", "error", "input_label", "input_ref", "model_version",
                                          "created_at", "finished_at", "summary", "origin", "repeat_of")})
        if len(out) >= limit:
            break
    return out


@router.post("/import")
async def import_omni(files: list[UploadFile] = File(...)):
    """Offline machines: upload omni2_YYYY.dat files downloaded elsewhere (no terminal needed)."""
    import tempfile
    from pathlib import Path

    from spaceml.spaceweather import data
    out = []
    for f in files[:40]:
        raw = await f.read(60 * 1024 * 1024)
        name = Path(f.filename or "omni2.dat").name
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_bytes(raw)
            try:
                n = data.import_file(p)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"{name}: not a readable OMNI2 hourly file ({e})") from e
        out.append({"file": name, "records": n})
    return {"imported": out, "cached_years": data.cached_years()}
