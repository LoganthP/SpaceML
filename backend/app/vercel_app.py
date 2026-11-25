"""Read-only API: frozen results plus a bounded live proxy, no SQLite or ML imports."""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from spaceml.config import REPO_ROOT, STORAGE_DIR

VERSION = "3.1.10"
HINT = "Connect Vercel compute to run this operation"
ROOT = REPO_ROOT / "backend" / "static_data"
MAX_BODY = 4_000_000
app = FastAPI(title="Space ML Mission Control", version=VERSION)


@lru_cache(maxsize=1)
def index() -> dict:
    return json.loads((ROOT / "index.json").read_text(encoding="utf-8"))


def key(path: str, params: dict | None = None) -> str:
    return path + ("?" + urlencode(sorted(params.items())) if params else "")


def stored(path: str, params: dict | None = None, default=None):
    record = index()["responses"].get(key(path, params))
    if record is None:
        return copy.deepcopy(default)
    body = json.loads((ROOT / "responses" / record["file"]).read_text(encoding="utf-8"))
    if path == "/api/catalog":
        runs = all_runs()
        for entry in body["datasets"]:
            status = entry.get("status", {})
            mission = entry.get("mission")
            run = next((r for r in runs if r.get("status") == "succeeded" and
                        r.get("module") == entry.get("module") and
                        (not mission or r.get("input_ref", {}).get("mission") == mission)), None)
            entry["capabilities"] = {"write": False, "train": False}
            if run and status.get("data") != "needs_repair":
                status["next"] = {"kind": "link", "label": "View sample result",
                               "to": status.get("page", "/").split("?")[0] + "?" +
                               urlencode({**({"mission": mission} if mission else {}), "run": run["id"]})}
    return body


def all_runs() -> list:
    return stored("/api/runs", {"limit": 500}, stored("/api/runs", default=[]))


def limit(query, default=100, maximum=5000) -> int:
    try:
        value = int(query.get("limit", default))
    except ValueError:
        raise HTTPException(422, "limit must be an integer") from None
    return max(0, min(value, maximum))


def bounded(body, status=200):
    response = JSONResponse(body, status_code=status, headers={"Cache-Control": "public, s-maxage=300, stale-while-revalidate=86400"})
    if len(response.body) > MAX_BODY:
        # Prebuilt JSON is also hosted statically. Sending a redirect keeps the
        # Function response below 4.5 MB; the static response has no such limit.
        import hashlib
        name = hashlib.sha256(response.body).hexdigest() + ".json"
        if (REPO_ROOT / "frontend" / "public" / "data" / name).is_file():
            return RedirectResponse("/data/" + name, status_code=307)
        raise HTTPException(413, "Choose a smaller page or use the static result download")
    return response


@app.get("/api/health")
@app.get("/api/config")
def health():
    return {"status": "ok", "mode": "vercel", "version": VERSION,
            "precomputed_modules": sorted({r["module"] for r in all_runs()}),
            "stored_inputs": [{"module": r["module"], "input": r.get("input_ref", {})} for r in all_runs() if r.get("status") == "succeeded"],
            "read_only": ["training", "install", "download", "uploads", "reviews", "annotations", "model_activation"],
            "capabilities": {"write": False, "train": False, "live": True, "stored_scoring": True},
            "hint": HINT, "storage": "ephemeral", "static_responses": len(index()["responses"])}


@app.get("/api/live/solar-wind")
def solar_wind():
    from spaceml.common.solarwind import latest
    return JSONResponse(latest(), headers={"Cache-Control": "no-store"})


@app.get("/api/live/{telescope}")
def live(telescope: str, refresh: bool = False):
    from spaceml.common import stlive
    from spaceml.common.fetch import SourceUnavailable
    if telescope not in stlive.TELESCOPES:
        raise HTTPException(404, "unknown telescope")
    headers = {"Cache-Control": "no-store" if refresh else f"public, s-maxage={int(stlive.MAX_AGE)}"}
    try:
        response = JSONResponse(stlive.current(telescope, refresh), headers=headers)
        if len(response.body) > MAX_BODY:
            raise SourceUnavailable("Space Telescope Live", stlive.SITE, "response exceeds the function body limit")
        return response
    except (SourceUnavailable, OSError):
        return JSONResponse({"telescope": telescope, "name": stlive.TELESCOPES[telescope], "observation": None,
                             "fetched_at": datetime.now(timezone.utc).isoformat(), "source": "cached", "history": [],
                             "notice": "Space Telescope Live is offline; no last-known observation is cached on this instance."},
                            headers=headers)


@app.get("/api/live/{telescope}/sky/{obs_id}")
def sky(telescope: str, obs_id: str, fov: float = 0.5):
    from spaceml.common import stlive
    from spaceml.common.fetch import SourceUnavailable
    if telescope not in stlive.TELESCOPES or fov not in stlive.FOVS:
        raise HTTPException(404, "Unknown observation or field of view")
    try:
        path = stlive.sky(telescope, obs_id, fov)
        if path.stat().st_size > MAX_BODY:
            raise SourceUnavailable("CDS sky survey", stlive.SKY, "image exceeds the response limit")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, s-maxage=86400, stale-while-revalidate=604800"})
    except (SourceUnavailable, KeyError):
        # A cold instance may have no cached observation yet; provide a calm
        # unavailable image instead of a broken image / browser console error.
        return Response('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480"><rect width="100%" height="100%" fill="#020617"/><text x="50%" y="50%" text-anchor="middle" fill="#94a3b8">Sky image unavailable while offline</text></svg>',
                        media_type="image/svg+xml", headers={"Cache-Control": "public, s-maxage=60"})


@app.post("/api/runs")
async def score(request: Request):
    try:
        body = await request.json()
    except (ValueError, UnicodeError):
        raise HTTPException(422, "Expected a JSON run request") from None
    if not isinstance(body, dict) or not isinstance(body.get("input"), dict) or not body["input"]:
        raise HTTPException(422, "module and input are required")
    for run in all_runs():
        inp = run.get("input_ref") or {}
        supplied = {k: v for k, v in body["input"].items() if v is not None}
        if (run["module"] == body.get("module") and all(inp.get(k) == v for k, v in supplied.items())
                and (not body.get("model_version") or run.get("model_version") == body["model_version"])):
            return {**run, "execution": "precomputed"}
    raise HTTPException(409, f"{HINT}. This input has no precomputed result.")


@app.post("/api/runs/{run_id}/repeat")
def repeat(run_id: str):
    run = next((r for r in all_runs() if r["id"] == run_id), None)
    if not run:
        raise HTTPException(404, "Run not found")
    return {**run, "execution": "precomputed"}


@app.api_route("/api/{path:path}", methods=["POST", "PUT", "PATCH", "DELETE"])
def unavailable_write(path: str):
    raise HTTPException(409, HINT)


@app.get("/api/{path:path}")
def frozen(path: str, request: Request):
    route = "/api/" + path
    q = dict(request.query_params)
    if route == "/api/runs":
        runs = all_runs()
        return bounded([r for r in runs if not q.get("module") or r["module"] == q["module"]][:limit(q)])
    if route == "/api/telemetry/channels":
        channels = stored(route, default=[])
        return bounded([c for c in channels if not q.get("mission") or c["spacecraft"] == q["mission"]])
    if route == "/api/telemetry/preview":
        body = stored(route, {"chan_id": q["chan_id"]}) if q.get("chan_id") else stored(route)
        if body is None:
            raise HTTPException(404, "Stored channel not found")
        try:
            points = max(100, min(3000, int(q.get("points", 600))))
        except ValueError:
            raise HTTPException(422, "points must be an integer") from None
        run = next((r for r in all_runs() if r["module"] == "telemetry" and r["input_ref"].get("chan_id") == body["chan_id"]), None)
        if run:
            result = stored(f"/api/runs/{run['id']}")["result"]
            step = max(1, len(result["actual"]) // points)
            body["values"], body["step"] = result["actual"][::step], step
        return bounded(body)
    if route.startswith("/api/runs/") and route.endswith("/reproducibility"):
        rid = route.split("/")[3]
        if not any(r["id"] == rid for r in all_runs()):
            raise HTTPException(404, "Run not found")
        return bounded({"status": "not_a_repeat", "reason": "This is a precomputed reference. Repeat opens the same immutable result."})
    if route.endswith("/items") and route.startswith("/api/missions/"):
        body = stored(route, {"limit": 5000}, {"total": 0, "items": []})
        items = body["items"]
        for field in ("split", "channel"):
            if q.get(field):
                items = [i for i in items if i.get(field) == q[field]]
        if q.get("q"):
            items = [i for i in items if q["q"].lower() in str(i).lower()]
        if q.get("label"):
            items = [i for i in items if bool(i.get("anomaly")) == (q["label"] == "anomaly")]
        try:
            offset = max(0, int(q.get("offset", 0)))
        except ValueError:
            raise HTTPException(422, "offset must be an integer") from None
        return bounded({"total": len(items), "items": items[offset:offset + limit(q, 200)]})
    if route.endswith("/series") and route.startswith("/api/missions/"):
        body = stored(route, {"item": q.get("item", "")})
        if body is None:
            raise HTTPException(404, "Stored series not found")
        try:
            start = float(q.get("start", "-inf"))
            end = float(q.get("end", "inf"))
            points = max(50, min(20000, int(q.get("max_points", 2000))))
        except ValueError:
            raise HTTPException(422, "Invalid series range or max_points") from None
        selected = [i for i, t in enumerate(body.get("t", [])) if start <= t <= end]
        stride = max(1, (len(selected) + points - 1) // points)
        selected = selected[::stride]
        body["t"] = [body["t"][i] for i in selected]
        body["cols"] = {k: [values[i] for i in selected] for k, values in body.get("cols", {}).items()
                        if not q.get("columns") or k in q["columns"].split(",")}
        for label_key in ("label", "labeled_samples"):
            if isinstance(body.get(label_key), list):
                original = body[label_key]
                body[label_key] = [original[i] for i in selected if i < len(original)]
        body["n_returned"] = len(selected)
        body["downsampled"] = body.get("downsampled", False) or stride > 1
        return bounded(body)
    if route == "/api/mars/traverse/waypoint":
        try:
            sol = int(q.get("sol", 0))
        except ValueError:
            raise HTTPException(422, "sol must be an integer") from None
        return bounded({"sol": sol, "site": q.get("site"), "drive": q.get("drive"),
                        "raw": {"sol": sol, "images": [], "error": HINT}, "raw_status": "source_unavailable",
                        "analysed": [], "samples": []})
    if route == "/api/exoplanet/tces":
        items = stored(route, default=[])
        if q.get("q"):
            items = [i for i in items if str(i["kepid"]).startswith(q["q"])]
        if q.get("label"):
            items = [i for i in items if i["label"] == q["label"]]
        if q.get("cached_only") == "true":
            items = [i for i in items if i.get("cached")]
        return bounded(items[:limit(q, 50, 500)])
    if route == "/api/spaceweather/series":
        return bounded({"time": [], "series": {}, "meta": stored("/api/spaceweather/status", default={}).get("variables", {}),
                        "step_hours": 1, "summary": {"min_dst": None, "max_kp": None, "max_v": None, "min_bz": None, "coverage": 0}})
    if route == "/api/spaceweather/quality":
        return bounded({"status": "not_downloaded", "findings": [HINT], "columns": [], "samples": 0,
                        "missing_years": [], "provenance": []})
    if route == "/api/reviews/queue":
        body = stored(route, {"limit": 1000}, stored(route, default={"items": [], "pending": 0, "runs_scanned": 0, "by_module": {}}))
        if q.get("module"):
            body["items"] = [i for i in body["items"] if i["module"] == q["module"]]
            body["pending"] = body["by_module"].get(q["module"], 0)
        body["items"] = body["items"][:limit(q, 60, 1000)]
        return bounded(body)
    if route == "/api/reviews/evidence":
        body = stored(route, {"run_id": q.get("run_id", ""), "item": q.get("item", "")})
        if body is None:
            raise HTTPException(404, "Stored evidence not found")
        return bounded(body)
    if route == "/api/catalog":
        body = stored(route)
        body["free_bytes"] = None
        body["capacity_source"] = "unavailable"
        return bounded(body)
    if route.startswith("/api/runs/") and route.endswith("/report"):
        q.setdefault("format", "md")
    exact = key(route, {k: v for k, v in q.items() if k == "format"})
    file = index()["files"].get(exact)
    if file is None and exact not in index()["responses"]:
        file = index()["files"].get(route)
    if file:
        return RedirectResponse(file["url"], status_code=307)
    record = index()["responses"].get(exact) or index()["responses"].get(route)
    if record:
        source = ROOT / "responses" / record["file"]
        if source.stat().st_size > MAX_BODY:
            return RedirectResponse("/data/" + record["file"], status_code=307)
        filename = f"run-{route.split('/')[3]}-report.json" if route.startswith("/api/runs/") and route.endswith("/report") else None
        return FileResponse(source, media_type="application/json", filename=filename,
                            headers={"Cache-Control": "public, s-maxage=300, stale-while-revalidate=86400"})
    raise HTTPException(404, "No precomputed resource at this path")
