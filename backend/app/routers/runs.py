from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import ValidationError

from spaceml.common.utils import read_json
from spaceml.config import RUNS_DIR, UPLOADS_DIR

from .. import registry, worker
from ..schemas import ExoplanetInput, MarsInput, ReviewIn, RunCreate, SpaceWeatherInput, TelemetryInput, WorkbenchInput

router = APIRouter(prefix="/api/runs", tags=["runs"])
RUN_FILES = {"input.png", "mask.png", "label.png", "before.png", "after.png", "change.png", "source.png", "corrected_mask.png"}


def _require_torch(what: str) -> None:
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise HTTPException(409, f"{what} needs PyTorch, which this installation does not include (lite image built "
                                 "with TORCH_INDEX=none). Rebuild with the default: docker compose build --no-cache, "
                                 "or locally: pip install -r backend/requirements.txt")


def _precheck(module: str, raw: dict) -> tuple[dict, str]:
    """Validate the input and fail fast (before queueing) when data or model is missing."""
    if module in ("telemetry", "mars"):
        _require_torch("Telemetry anomaly detection" if module == "telemetry" else "Mars terrain segmentation")
    try:
        if module == "missions":
            from spaceml.missions import catalog as mcat
            from spaceml.missions import models as mmodels
            from spaceml.missions import store as mstore
            from ..schemas import MissionInput
            inp = MissionInput(**raw)
            meta = mcat.get(inp.mission)
            if not mstore.installed(inp.mission):
                raise HTTPException(409, f"{meta['name']} data is not installed. {meta['prepare']}")
            if inp.action == "evaluate" and not mmodels.active(inp.mission):
                raise HTTPException(409, f"No trained {meta['name']} model yet - press Quick train first.")
            if inp.action == "score":
                if not inp.item:
                    raise HTTPException(422, f"Choose a {meta['item_label']} to score.")
                mstore.item(inp.mission, inp.item)
            what = "held-out evaluation" if inp.action == "evaluate" else f"{meta['item_label']} {inp.item}"
            return inp.model_dump(), f"{meta['name']} · {what}"
        if module == "workbench":
            from spaceml.workbench import analyses
            from spaceml.workbench.store import WorkbenchError
            inp = WorkbenchInput(**raw)
            try:
                meta, params = analyses.validate(inp.analysis, inp.dataset_id, inp.params)
            except WorkbenchError as e:
                raise HTTPException(422, str(e)) from e
            need = analyses.ANALYSES[inp.analysis]["requires_model"]
            if need == "mars":
                registry.active_dir("mars")
            title = analyses.ANALYSES[inp.analysis]["title"]
            return ({"dataset_id": inp.dataset_id, "analysis": inp.analysis, "params": params,
                     "dataset_sha256": meta.get("sha256")}, f"{title} · {meta['name'][:40]}")
        if module == "telemetry":
            inp = TelemetryInput(**raw)
            from spaceml.telemetry import data
            if not data.is_usable():
                raise HTTPException(409, "Telemetry data not downloaded. Run: python -m spaceml.telemetry.data download")
            chans = data.load_channels()
            if inp.chan_id not in chans:
                raise HTTPException(404, f"Unknown channel {inp.chan_id}")
            if not data.has_channel(inp.chan_id):
                raise HTTPException(409, f"Telemetry for {inp.chan_id} is not in the starter pack. Download the full "
                                         "NASA set: python -m spaceml.telemetry.data download")
            vdir = registry.active_dir("telemetry")
            if not (vdir / f"{inp.chan_id}.pt").exists():
                raise HTTPException(409, f"No trained model for {inp.chan_id} in {vdir.name}. "
                                         f"Run: python -m spaceml.telemetry.train --channels {inp.chan_id} "
                                         f"--version {vdir.name}")
            return inp.model_dump(), f"{inp.chan_id} · {chans[inp.chan_id].spacecraft}"
        if module == "exoplanet":
            inp = ExoplanetInput(**raw)
            if inp.dataset_id:
                from spaceml.exoplanet import uploads
                m = uploads.get(inp.dataset_id)
                tg = next((t for t in m["targets"] if t["id"] == inp.target_id), None)
                if tg is None:
                    raise HTTPException(404, "Target not found in this dataset")
                if inp.period is None and not tg.get("searchable", True):
                    raise HTTPException(422, tg.get("note") or "Light curve too short to search")
                return inp.model_dump(exclude_none=True), f"{tg['name']} · upload"
            from spaceml.exoplanet import data
            data.tce_row(inp.kepid, inp.tce_plnt_num)
            return inp.model_dump(exclude_none=True), f"KIC {inp.kepid} · TCE {inp.tce_plnt_num}"
        if module == "spaceweather":
            inp = SpaceWeatherInput(**raw)
            from spaceml.spaceweather import data as swd
            if inp.event_id:
                ev = next((e for e in swd.EVENTS if e["id"] == inp.event_id), None)
                if not ev:
                    raise HTTPException(404, f"Unknown event {inp.event_id}")
                start, end, label = ev["start"], ev["end"], f"Replay · {ev['name']}"
            else:
                try:
                    start, end = swd.ts(inp.start).isoformat(), swd.ts(inp.end).isoformat()
                except Exception as ex:  # noqa: BLE001
                    raise HTTPException(422, f"Invalid date: {ex}") from ex
                label = f"Replay · {start[:16]} → {end[:16]}"
            missing = sorted({swd.ts(start).year, swd.ts(end).year} - set(swd.cached_years()))
            if missing:
                raise HTTPException(409, f"OMNI2 data for {missing} is not downloaded yet. Run: python -m "
                                         f"spaceml.spaceweather.data download --years {' '.join(map(str, missing))}")
            registry.active_dir("spaceweather")
            return inp.model_dump(exclude_none=True), label
        inp = MarsInput(**raw)
        registry.active_dir("mars")
        if inp.raw_image_id:
            from spaceml.mars import traverse
            meta = traverse._find_meta(inp.raw_image_id)
            if not meta:
                raise HTTPException(404, "Unknown raw image - open its waypoint first")
            return inp.model_dump(exclude_none=True), f"Sol {meta.get('sol')} · {inp.raw_image_id[:24]}"
        if inp.sample_id:
            from spaceml.mars import data
            data.sample_paths(inp.sample_id)
            return inp.model_dump(), f"Sample {inp.sample_id}"
        if not (UPLOADS_DIR / f"{inp.upload_id}.png").exists():
            raise HTTPException(404, "Upload not found")
        meta = read_json(UPLOADS_DIR / f"{inp.upload_id}.json", {}) or {}
        return inp.model_dump(), f"Upload {meta.get('filename', inp.upload_id)}"
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False, include_input=False)) from e
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e


@router.post("")
def create_run(body: RunCreate):
    inp, label = _precheck(body.module, body.input)
    try:
        return worker.submit(body.module, inp, label, version=body.model_version)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e


@router.get("")
def list_runs(module: str | None = Query(None, pattern="^(telemetry|exoplanet|mars|workbench|spaceweather|missions)$"),
              limit: int = Query(100, ge=1, le=500)):
    return worker.list_runs(module, limit)


@router.post("/{run_id}/repeat")
def repeat_run(run_id: str, same_model: bool = True):
    """Run the same input again - by default with the same model version - to check reproducibility."""
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    inp = dict(run["input_ref"] or {})
    raw = {k: v for k, v in inp.items() if k != "dataset_sha256"}
    if run["module"] == "workbench":
        from spaceml.workbench.store import get as wb_get
        try:
            meta = wb_get(inp["dataset_id"])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(409, "The dataset of this run was deleted - upload it again to repeat") from e
        if inp.get("dataset_sha256") and meta.get("sha256") != inp["dataset_sha256"]:
            raise HTTPException(409, "The dataset content changed since this run (different hash)")
    checked, label = _precheck(run["module"], raw)
    version = run.get("model_version") if same_model and run["module"] in registry.MODULES else None
    try:
        return worker.submit(run["module"], checked, label, version=version, repeat_of=run_id)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/{run_id}/manifest")
def run_manifest(run_id: str):
    from ..manifest import build_manifest
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return build_manifest(run)


@router.get("/{run_id}/reproducibility")
def reproducibility(run_id: str):
    """Compare a repeated run with the run it repeats."""
    from ..manifest import compare_runs
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if not run.get("repeat_of"):
        raise HTTPException(422, "This run is not a repeat of another run")
    orig = worker.get(run["repeat_of"])
    if not orig:
        raise HTTPException(404, "The original run was deleted")
    return compare_runs(orig, run)


@router.get("/{run_id}/evidence.zip")
def evidence(run_id: str):
    from fastapi.responses import Response

    from ..manifest import evidence_zip
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    blob = evidence_zip(run)
    return Response(blob, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="run-{run_id}-evidence.zip"'})


@router.get("/{run_id}")
def get_run(run_id: str, include_result: bool = True):
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if include_result and run["status"] == "succeeded" and run.get("result_path"):
        run["result"] = read_json(RUNS_DIR / run_id / "result.json")
    return run


@router.get("/{run_id}/files/img/{name}")
def run_image_file(run_id: str, name: str):
    """Per-image thumbnails/overlays of a workbench image-insights run."""
    import re
    if not run_id.isalnum() or not re.fullmatch(r"i\d{1,4}(_terrain\.png|_regions\.png|_texture\.png|\.jpg)", name):
        raise HTTPException(404, "File not found")
    p = RUNS_DIR / run_id / "img" / name
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="image/jpeg" if name.endswith(".jpg") else "image/png",
                        headers={"Cache-Control": "max-age=86400"})


@router.get("/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    if not run_id.isalnum() or name not in RUN_FILES:
        raise HTTPException(404, "File not found")
    p = RUNS_DIR / run_id / name
    if not p.exists():
        raise HTTPException(404, "File not found")
    cache = "no-cache" if name == "corrected_mask.png" else "max-age=86400"
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": cache})


@router.delete("/{run_id}")
def delete_run(run_id: str):
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if run["status"] in ("queued", "running"):
        raise HTTPException(409, "Run is still in progress")
    worker.delete(run_id)
    return {"deleted": run_id}


# ------------------------------------------------------------------ reviews
@router.get("/{run_id}/reviews")
def list_reviews(run_id: str):
    from ..db import tx
    with tx() as c:
        return [dict(r) for r in c.execute("SELECT * FROM reviews WHERE run_id=? ORDER BY item", (run_id,))]


@router.put("/{run_id}/reviews/{item}")
def put_review(run_id: str, item: str, body: ReviewIn):
    import re

    from spaceml.common.utils import now_iso

    from ..db import tx
    if not worker.get(run_id):
        raise HTTPException(404, "Run not found")
    if not re.fullmatch(r"[\w:.\-]{1,80}", item):
        raise HTTPException(422, "Invalid review item")
    with tx() as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?)", (run_id, item, body.verdict, body.note, now_iso()))
    return {"run_id": run_id, "item": item, "verdict": body.verdict, "note": body.note}


@router.delete("/{run_id}/reviews/{item}")
def delete_review(run_id: str, item: str):
    from ..db import tx
    with tx() as c:
        c.execute("DELETE FROM reviews WHERE run_id=? AND item=?", (run_id, item))
    return {"deleted": item}


# ------------------------------------------------------------------ reports
@router.get("/{run_id}/report")
def report(run_id: str, format: str = Query("md", pattern="^(md|json)$")):
    from fastapi.responses import JSONResponse, PlainTextResponse

    from ..reports import build
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    data, md = build(run)
    name = f"run-{run_id}-report"
    if format == "json":
        return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    return PlainTextResponse(md, media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
