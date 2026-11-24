"""Multi-mission telemetry API: catalog of sources, items, series (time-range + downsampled), models."""
from __future__ import annotations

import re
import shutil
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from spaceml.common.utils import read_json
from spaceml.missions import catalog, models, store

router = APIRouter(prefix="/api/missions", tags=["missions"])


def repair_needed(mid: str) -> dict | None:
    """A previously imported store that is known to be wrong (and how to fix it with one click)."""
    if mid != "webb" or not store.installed(mid):
        return None
    from spaceml.missions.pointing import webb_time_problem
    files = webb_time_problem(store.items(mid))
    if not files:
        return None
    return {"reason": "These Webb visits were imported by an older release that misread the guider clock, so every "
                      "visit is squeezed into a few microseconds and the time axis is wrong. Re-download the same "
                      f"{len(files)} MAST files once to fix it (needs internet; about 1-2 MB each).",
            "label": "Re-download Webb data (fix timestamps)", "task": "missions.webb.install",
            "params": {"files": files}}


def _status(mid: str) -> dict:
    meta = catalog.get(mid)
    man = store.manifest(mid)
    act = models.active(mid)
    inst = store.installed(mid)
    return {"id": mid, "name": meta["name"], "agency": meta["agency"], "family": meta["family"],
            "summary": meta["summary"], "item_label": meta["item_label"], "time_basis": meta["time_basis"],
            "trainable": bool(meta.get("train")), "train_label": (meta.get("train") or {}).get("label"),
            "detect": meta.get("detect"), "evaluate": meta.get("evaluate"), "source": meta["source"],
            "dataset": {"installed": inst,
                        "state": "installed" if inst else "importer available - no local data",
                        "scope": (man or {}).get("scope"), "counts": (man or {}).get("counts"),
                        "date_range": (man or {}).get("date_range"), "items": (man or {}).get("items"),
                        "events": (man or {}).get("events"), "bytes": (man or {}).get("prepared_bytes"),
                        "source_version": ((man or {}).get("source") or {}).get("version"),
                        "next_action": None if inst else f"Prepare once on a machine with the source data: {meta['prepare']}"},
            "repair": repair_needed(mid),
            "install": __import__("spaceml.missions.install", fromlist=["SPECS"]).SPECS.get(mid),
            "versions": store.versions(mid),
            "auto_install": __import__("spaceml.missions.install", fromlist=["auto_source"]).auto_source(mid),
            "model": {"ready": act is not None, "stale": bool(act and act.get("stale")),
                      "version": act and act["version"], "run_kind": act and act["run_kind"],
                      "headline": act and (act.get("metrics") or {}).get("headline")}}


@router.post("/{mission}/versions/{vid}/activate")
def activate_version(mission: str, vid: str):
    """Switch the installed data to an earlier/later version (models follow the data fingerprint)."""
    if mission not in catalog.MISSIONS:
        raise HTTPException(404, f"Unknown mission {mission}")
    from .. import jobs
    if any(j["status"] in ("queued", "running") and j["task"].startswith(f"missions.{mission}.")
           for j in jobs.list_jobs(50)):
        raise HTTPException(409, "A job for this mission is running - switch versions when it has finished")
    try:
        man = store.activate(mission, vid)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"active": store.versions(mission)[0], "label": store.version_label(man)}


MAX_UPLOAD = 8 * 1024**3          # ESA-AD mission zips are a few GB


@router.post("/{mission}/upload")
async def upload(mission: str, files: list[UploadFile] = File(...)):
    """Receive source files for the Install data action. Files go to data/missions-source/<mission>/<folder>;
    the install job validates and imports them (and deletes the upload afterwards)."""
    from spaceml.missions.install import SRC_ROOT, SPECS
    try:
        catalog.get(mission)
    except KeyError as e:
        raise HTTPException(404, str(e))
    accept = SPECS[mission]["upload"]["accept"].split(",")
    folder = f"upload-{time.strftime('%Y%m%d-%H%M%S')}"
    dest = SRC_ROOT / mission / folder
    dest.mkdir(parents=True, exist_ok=True)
    total, names = 0, []
    try:
        for f in files:
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", (f.filename or "file").split("/")[-1].split("\\")[-1])[:160]
            if not any(name.lower().endswith(a) for a in accept):
                raise HTTPException(422, f"{name}: expected {', '.join(accept)} ({SPECS[mission]['upload']['hint']})")
            with open(dest / name, "wb") as out:
                while chunk := await f.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "upload larger than 8 GB")
                    out.write(chunk)
            names.append(name)
    except HTTPException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return {"folder": folder, "files": names, "bytes": total}


@router.get("")
def list_missions():
    return {"missions": [_status(m) for m in catalog.ORDER]}


@router.get("/{mission}")
def mission(mission: str):
    try:
        st = _status(mission)
    except KeyError as e:
        raise HTTPException(404, str(e))
    st["manifest"] = store.manifest(mission)
    st["models"] = [{k: v for k, v in m.items() if k != "path"} for m in models.versions(mission)]
    return st


@router.get("/{mission}/items")
def items(mission: str, q: str | None = None, split: str | None = None, channel: str | None = None,
          label: str | None = Query(None, pattern="^(anomaly|nominal)$"), limit: int = Query(200, le=5000), offset: int = 0):
    try:
        catalog.get(mission)
    except KeyError as e:
        raise HTTPException(404, str(e))
    its = store.items(mission)
    if q:
        ql = q.lower()
        its = [i for i in its if ql in str(i["id"]).lower() or ql in str(i.get("label", "")).lower()]
    if split:
        its = [i for i in its if i.get("split") == split]
    if channel:
        its = [i for i in its if i.get("channel") == channel]
    if label:
        its = [i for i in its if ("anomaly" if i.get("anomaly") else "nominal") == label]
    return {"total": len(its), "items": its[offset:offset + limit]}


@router.get("/{mission}/series")
def series(mission: str, item: str, columns: str | None = None, start: float | None = None, end: float | None = None,
           max_points: int = Query(2000, ge=50, le=20000)):
    try:
        out = store.query(mission, item, columns.split(",") if columns else None, start, end, max_points)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    out["events"] = [e for e in store.events(mission) if e.get("channel") == item]
    return out


@router.get("/{mission}/predictions")
def predictions(mission: str, version: str | None = None):
    v = models.version_dir(mission, version) if version else (models.active(mission) or {}).get("path")
    if not v:
        raise HTTPException(404, "no trained model")
    from pathlib import Path
    return {"version": Path(v).name, "metrics": read_json(Path(v) / "metrics.json"),
            "predictions": read_json(Path(v) / "predictions.json")}
