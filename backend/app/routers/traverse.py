"""Rover Traverse Explorer API (Curiosity waypoints, raw images, analysed terrain along the drive)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from spaceml.common.fetch import SourceUnavailable

from ..db import tx
from ..worker import list_runs

router = APIRouter(prefix="/api/mars/traverse", tags=["traverse"])
MAX_POINTS = 4000


def _analysed() -> list[dict]:
    """Succeeded Mars runs that carry a location, with their review state."""
    runs = [r for r in list_runs("mars", 500) if r["status"] == "succeeded"]
    with tx() as c:
        rev = {}
        for r in c.execute("SELECT run_id, item, verdict FROM reviews"):
            rev.setdefault(r["run_id"], []).append((r["item"], r["verdict"]))
    from spaceml.common.utils import read_json
    from spaceml.config import RUNS_DIR
    out = []
    for r in runs:
        res = read_json(RUNS_DIR / r["id"] / "result.json", {}) or {}
        loc = res.get("location") or {}
        reviews = rev.get(r["id"], [])
        out.append({"run_id": r["id"], "label": r["input_label"], "origin": r.get("origin"),
                    "created_at": r["created_at"], "class_shares": res.get("class_shares"),
                    "sol": loc.get("sol"), "site": loc.get("site"), "drive": loc.get("drive"),
                    "lat": loc.get("lat"), "lon": loc.get("lon"), "dist_total_m": loc.get("dist_total_m"),
                    "precision": loc.get("precision") or "unlocated",
                    "reviewed": bool(reviews), "corrected": any(v == "corrected" for _, v in reviews),
                    "top_class": max(res["class_shares"], key=res["class_shares"].get) if res.get("class_shares") else None})
    return out


@router.get("")
def traverse_overview():
    from spaceml.mars import traverse
    st = traverse.status()
    body = {"status": st, "analysed": _analysed(), "waypoints": None}
    if st["downloaded"]:
        wps = traverse.waypoints()
        step = max(1, len(wps) // MAX_POINTS)
        keep = wps[::step] + ([wps[-1]] if (len(wps) - 1) % step else [])
        body["waypoints"] = {k: [w[k] for w in keep] for k in ("sol", "site", "drive", "lon", "lat", "dist_total_m")}
        body["waypoints"]["step"] = step
    return body


@router.post("/download")
def download_waypoints():
    from spaceml.mars import traverse
    try:
        return traverse.fetch_waypoints()
    except SourceUnavailable as e:
        raise HTTPException(503, {"status": "source_unavailable", "message": str(e)}) from e
    except traverse.TraverseError as e:
        raise HTTPException(502, str(e)) from e


@router.get("/waypoint")
def waypoint(sol: int = Query(..., ge=0, le=20000), site: int | None = Query(None, ge=0, le=9999),
             drive: int | None = Query(None, ge=0, le=99999)):
    from spaceml.mars import data, traverse
    local = [a for a in _analysed() if a["sol"] == sol or (site is not None and a["site"] == site)]
    samples = []
    for s in data.list_samples():
        pid = traverse.parse_product_id(s.get("name", ""))
        if pid and (pid["site"] == site or pid["sol_estimate"] == sol):
            samples.append({"id": s["id"], "name": s["name"], "site": pid["site"], "drive": pid["drive"],
                            "sol_estimate": pid["sol_estimate"]})
    try:
        raw = traverse.raw_images(sol, site, drive)
        raw_status = "ready" if raw["images"] else "no_images"
    except SourceUnavailable as e:
        raw, raw_status = {"sol": sol, "images": [], "error": str(e)}, "source_unavailable"
    return {"sol": sol, "site": site, "drive": drive, "raw": raw, "raw_status": raw_status,
            "analysed": local, "samples": samples}


@router.get("/raw/{imageid}")
def raw_image(imageid: str, thumb: bool = True):
    from PIL import Image

    from spaceml.mars import traverse
    try:
        path, _ = traverse.raw_image(imageid)
    except SourceUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except traverse.TraverseError as e:
        raise HTTPException(404, str(e)) from e
    if not thumb:
        return FileResponse(path, headers={"Cache-Control": "max-age=604800"})
    t = path.with_name(path.stem + "_thumb.jpg")
    if not t.exists():
        im = Image.open(path)
        im.thumbnail((320, 320))
        im.convert("L").save(t, quality=82)
    return FileResponse(t, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800"})


@router.get("/sections")
def sections(section_m: float = Query(1000, ge=100, le=10000), sol_min: int = Query(0, ge=0),
             sol_max: int = Query(20000, le=20000), terrain: str | None = Query(None, pattern="^(soil|bedrock|sand|big rock)$"),
             reviewed_only: bool = False):
    from spaceml.mars import traverse
    pts = [a for a in _analysed() if a["dist_total_m"] is not None
           and (a["sol"] is None or sol_min <= a["sol"] <= sol_max)
           and (not reviewed_only or a["reviewed"]) and (terrain is None or a["top_class"] == terrain)]
    return {"section_m": section_m, "points": len(pts), "sections": traverse.sections(pts, section_m)}
