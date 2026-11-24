"""Label & Review Studio: every human verdict across modules, corrected terrain masks, and
exports of reviewed labels in formats a future training run can read."""
from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image

from spaceml.common.utils import now_iso, read_json
from spaceml.config import RUNS_DIR

from .. import worker
from ..db import tx

router = APIRouter(tags=["reviews"])


def _describe(run: dict, item: str, result: dict | None) -> dict:
    """Turn a review item id into something a training script can use."""
    inp = run.get("input_ref") or {}
    d: dict = {"module": run["module"], "analysis": inp.get("analysis"), "input": run["input_label"]}
    r = result or {}
    try:
        if item.startswith("flag:") and run["module"] == "telemetry":
            a, b = item[5:].split("-")
            d.update(kind="telemetry_interval", chan_id=inp.get("chan_id"), start=int(a), end=int(b))
        elif item.split(":")[0] in ("robust", "zscore") and r.get("viewer") == "timeline":
            det, i = item.split(":")
            iv = r[det]["intervals"][int(i)]
            d.update(kind="series_interval", detector=det, dataset_id=inp.get("dataset_id"), file=r.get("file"),
                     start=iv["start"], end=iv["end"], x0=iv["x0"], x1=iv["x1"])
        elif item.startswith("region:") and r.get("viewer") == "change":
            g = r["regions"][int(item[7:])]
            d.update(kind="change_region", dataset_id=inp.get("dataset_id"), bbox=g["bbox"], pixels=g["pixels"],
                     area_km2=g.get("area_km2"), before=r["before"]["file"], after=r["after"]["file"])
        elif item.startswith("sample:") and r.get("viewer") == "classification":
            sid = item[7:]
            row = next((x for x in r.get("samples", []) if x["id"] == sid), {})
            d.update(kind="labeled_sample", dataset_id=inp.get("dataset_id"), sample_id=sid, label=row.get("label"),
                     prob=row.get("prob"), outcome=row.get("outcome"), target=r.get("target"))
        elif item == "candidate":
            d.update(kind="transit_candidate", kepid=inp.get("kepid"), tce_plnt_num=inp.get("tce_plnt_num"),
                     dataset_id=inp.get("dataset_id"), target_id=inp.get("target_id"),
                     period_days=(r.get("tce") or {}).get("period_days"), score=r.get("candidate_score"))
        elif item == "mask":
            d.update(kind="terrain_mask", file="corrected_mask.png")
        elif run["module"] == "missions":
            it = next((x for x in r.get("review_items", []) if x["item"] == item), None) or {}
            d.update(kind=it.get("kind", "mission_item"), mission=inp.get("mission"), target=it.get("target"),
                     **{k: it[k] for k in ("label", "pred", "prob", "outcome", "start", "end", "detector") if k in it})
    except (KeyError, IndexError, ValueError):
        d["kind"] = "unknown"
    return d


def _all_reviews(module: str | None = None, verdict: str | None = None) -> list[dict]:
    q = ("SELECT rv.*, r.module, r.input_label, r.input_ref, r.created_at AS run_created, r.model_version "
         "FROM reviews rv JOIN runs r ON r.id = rv.run_id")
    cond, args = [], []
    if module:
        cond.append("r.module = ?"); args.append(module)
    if verdict:
        cond.append("rv.verdict = ?"); args.append(verdict)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    with tx() as c:
        rows = [dict(x) for x in c.execute(q + " ORDER BY rv.updated_at DESC", args)]
    import json
    cache: dict[str, dict | None] = {}
    out = []
    for row in rows:
        row["input_ref"] = json.loads(row["input_ref"]) if row.get("input_ref") else {}
        if row["run_id"] not in cache:
            cache[row["run_id"]] = read_json(RUNS_DIR / row["run_id"] / "result.json")
        desc = _describe({"module": row["module"], "input_ref": row["input_ref"], "input_label": row["input_label"]},
                         row["item"], cache[row["run_id"]])
        out.append({**row, **{k: v for k, v in desc.items() if k not in row}})
    return out


@router.get("/api/reviews")
def list_reviews(module: str | None = Query(None, pattern="^(telemetry|exoplanet|mars|workbench|missions)$"),
                 verdict: str | None = Query(None, pattern="^(confirmed|dismissed|unsure|corrected)$")):
    rows = _all_reviews(module, verdict)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"reviews": rows, "counts": counts}


def _queue_items(run: dict, res: dict) -> list[dict]:
    """Reviewable items of one finished run, most important first."""
    mod, items = run["module"], []
    if mod == "telemetry":
        lab = res.get("labeled") or []
        for f in res.get("flagged") or []:
            hit = any(not (f["end"] < l["start"] or f["start"] > l["end"]) for l in lab)
            items.append({"item": f"flag:{f['start']}-{f['end']}", "kind": "telemetry_interval",
                          "title": f"{res.get('chan_id')} · steps {f['start']}–{f['end']}",
                          "detail": f"peak error {f.get('peak_error', 0):.3g}" + (" · overlaps an expert label" if hit
                                                                                 else " · no expert label here"),
                          "hint": "confirmed" if hit else "dismissed"})
    elif mod == "workbench" and res.get("viewer") == "timeline":
        for i, iv in enumerate((res.get("robust") or {}).get("intervals") or []):
            items.append({"item": f"robust:{i}", "kind": "series_interval",
                          "title": f"{str(iv.get('x0'))[:19]} → {str(iv.get('x1'))[:19]}",
                          "detail": f"{iv.get('length', 0)} steps"
                                    + (f" · {', '.join(iv['channels'][:3])}" if iv.get("channels") else "")})
    elif mod == "workbench" and res.get("viewer") == "change":
        for i, g in enumerate((res.get("regions") or [])[:30]):
            area = g.get("area_km2")
            items.append({"item": f"region:{i}", "kind": "change_region",
                          "title": f"Region {i + 1} · " + (f"{area:.3g} km²" if area is not None else f"{g.get('pixels')} px"),
                          "detail": f"bbox {g.get('bbox')}"})
    elif mod == "workbench" and res.get("viewer") == "insights":
        for im in res.get("images") or []:
            if im.get("error"):
                continue
            reg = im.get("regions") or {}
            items.append({"item": f"image:{im['id']}", "kind": "image_class",
                          "title": f"{im.get('scene_label', im.get('scene'))} · {im['file'][:48]}",
                          "detail": (f"terrain {reg.get('terrain', 0):.0%} · hardware {reg.get('hardware', 0):.0%} · "
                                     if reg else "") + f"{im.get('rocks', 0)} rocks · is this classification right?",
                          "thumb": f"/api/runs/{run['id']}/files/{im.get('region_map') or im.get('thumb')}"})
    elif mod == "workbench" and res.get("viewer") == "classification":
        rank = {"missed": 0, "false alarm": 1, "hit": 2}
        for r in sorted([x for x in res.get("samples") or [] if x["outcome"] in rank],
                        key=lambda x: (rank[x["outcome"]], -x["prob"]))[:60]:
            items.append({"item": f"sample:{r['id']}", "kind": "labeled_sample",
                          "title": f"Sample {r['id']} · labeled {r['label']} · {r['outcome']}",
                          "detail": f"model probability {r['prob']:.2f} · {r['file']}",
                          "hint": "confirmed" if r["outcome"] == "hit" else None})
    elif mod == "exoplanet":
        sc = res.get("candidate_score")
        items.append({"item": "candidate", "kind": "transit_candidate",
                      "title": f"{run['input_label']}", "detail": f"candidate score {sc:.2f}" if sc is not None else
                      "no CNN score", "hint": None if sc is None else ("confirmed" if sc >= 0.5 else "dismissed")})
    elif mod == "missions":
        for it in res.get("review_items") or []:
            items.append({k: v for k, v in it.items()})
    elif mod == "mars" or (mod == "workbench" and res.get("viewer") == "segmentation"):
        sh = res.get("class_shares") or {}
        top = max(sh, key=sh.get) if sh else "?"
        items.append({"item": "mask", "kind": "terrain_mask", "title": f"Terrain mask · mostly {top}",
                      "detail": " · ".join(f"{k} {v:.0%}" for k, v in sh.items()), "mask": True})
    return items


@router.get("/api/reviews/queue")
def review_queue(module: str | None = Query(None, pattern="^(telemetry|exoplanet|mars|workbench|missions)$"),
                 limit: int = Query(60, ge=1, le=1000)):
    """Everything that still waits for a human verdict, across the latest finished runs."""
    runs = [r for r in worker.list_runs(module, 200) if r["status"] == "succeeded" and r["module"] != "spaceweather"]
    with tx() as c:
        done = {(r["run_id"], r["item"]) for r in c.execute("SELECT run_id, item FROM reviews")}
    out, total, seen, by_module = [], 0, {}, {}
    for r in runs:                                       # newest run first
        res = read_json(RUNS_DIR / r["id"] / "result.json") or {}
        for it in _queue_items(r, res):
            if (r["id"], it["item"]) in done:
                continue
            key = (r["module"], it.get("kind"), it.get("title"))
            if key in seen:                              # the same flag from an older re-run: count it once
                seen[key]["repeats"] = seen[key].get("repeats", 0) + 1
                continue
            total += 1
            by_module[r["module"]] = by_module.get(r["module"], 0) + 1
            row = {**it, "run_id": r["id"], "module": r["module"], "input_label": r["input_label"],
                   "origin": r.get("origin"), "created_at": r["created_at"]}
            seen[key] = row
            if len(out) < limit:
                out.append(row)
    return {"items": out, "pending": total, "runs_scanned": len(runs), "by_module": by_module}


def _window(n: int, a: int, b: int, pad_factor: float = 3.0, min_pad: int = 50) -> tuple[int, int]:
    pad = max(min_pad, int((b - a + 1) * pad_factor))
    return max(0, a - pad), min(n, b + pad + 1)


def _thin(lo: int, hi: int, max_points: int = 1500) -> range:
    return range(lo, hi, max(1, (hi - lo) // max_points))


@router.get("/api/reviews/evidence")
def evidence(run_id: str = Query(..., pattern=r"^[A-Za-z0-9_-]{4,40}$"), item: str = Query(..., max_length=200)):
    """What a reviewer needs to judge one item: the signal around a flag, the images of a region, the folded
    light curve of a candidate, or the window of a labeled sample - in one uniform shape."""
    run = worker.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    res = read_json(RUNS_DIR / run_id / "result.json") or {}
    files = f"/api/runs/{run_id}/files"
    mod = run["module"]
    def num(v):
        try:
            f = float(v)
            return round(f, 6) if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    try:
        if mod == "telemetry" and item.startswith("flag:"):
            a, b = (int(x) for x in item[5:].split("-"))
            act, pred, err = res.get("actual") or [], res.get("predicted") or [], res.get("smoothed_error") or []
            lo, hi = _window(len(act), a, b)
            idx = list(_thin(lo, hi))
            return {"type": "series", "x_label": "step", "x": idx, "shade": [[a, b]], "panels": [
                {"name": f"{res.get('chan_id')} telemetry", "lines": [{"name": "actual", "y": [num(act[i]) for i in idx]},
                                                                      {"name": "predicted", "y": [num(pred[i]) for i in idx if i < len(pred)]}]},
                {"name": "prediction error", "lines": [{"name": "smoothed error", "y": [num(err[i]) for i in idx if i < len(err)]}],
                 "hline": num(res.get("threshold"))}],
                "facts": {"channel": res.get("chan_id"), "spacecraft": res.get("spacecraft"), "steps": f"{a}–{b}",
                          "expert label overlaps": any(not (b < l["start"] or a > l["end"]) for l in res.get("labeled") or [])}}
        if mod == "workbench" and res.get("viewer") == "timeline" and item.split(":")[0] in ("robust", "zscore"):
            det, i = item.split(":")
            iv = res[det]["intervals"][int(i)]
            xs = res.get("x") or []
            n = len(xs)
            lo, hi = _window(n, int(iv["start"]), int(iv["end"]))
            idx = list(_thin(lo, hi))
            return {"type": "series", "x_label": res.get("x_label") or "step", "x": [xs[j] for j in idx],
                    "shade": [[iv["x0"], iv["x1"]]], "panels": [
                        {"name": ch["name"] + (f" [{ch['unit']}]" if ch.get("unit") else ""),
                         "lines": [{"name": ch["name"], "y": [num(ch["values"][j]) if j < len(ch["values"]) else None for j in idx]}]}
                        for ch in (res.get("channels") or [])[:4]],
                    "facts": {"file": res.get("file"), "length": f"{iv.get('length')} steps", "channels": ", ".join(iv.get("channels") or []) or "-"}}
        if mod == "workbench" and res.get("viewer") == "change" and item.startswith("region:"):
            g = res["regions"][int(item[7:])]
            return {"type": "images", "images": [{"src": f"{files}/before.png", "label": "before"}, {"src": f"{files}/after.png", "label": "after"},
                                                 {"src": f"{files}/change.png", "label": "change map"}],
                    "bbox": g.get("bbox"), "facts": {"area": f"{g['area_km2']:.3g} km²" if g.get("area_km2") is not None else f"{g.get('pixels')} px",
                                                     "bounding box (px)": str(g.get("bbox"))}}
        if mod == "workbench" and res.get("viewer") == "insights" and item.startswith("image:"):
            im = next((x for x in res.get("images") or [] if f"image:{x['id']}" == item), {})
            src = [im.get(k) for k in ("region_map", "thumb") if im.get(k)]
            return {"type": "images", "images": [{"src": f"{files}/{p}", "label": p} for p in src],
                    "facts": {"scene": im.get("scene_label"), "rocks": im.get("rocks"), "file": im.get("file")}}
        if mod == "workbench" and res.get("viewer") == "classification" and item.startswith("sample:"):
            from spaceml.workbench import mvts
            sid = item[7:]
            row = next((x for x in res.get("samples") or [] if x["id"] == sid), {})
            params = [p["param"] for p in res.get("profiles") or []][:3]
            return {**mvts.sample_evidence(res["dataset_id"], sid, params), "facts": {
                "label": row.get("label"), "model probability": row.get("prob"), "outcome": row.get("outcome"),
                "file": row.get("file"), "target": res.get("target")}}
        if mod == "exoplanet" and item == "candidate":
            gv = res.get("global_view") or {}
            lv = res.get("local_view") or {}
            panels = [{"name": "phase-folded light curve (global view)", "lines": [{"name": "flux", "y": gv.get("y") or []}]}]
            if lv.get("y"):
                panels.append({"name": "zoom on the transit (local view)", "lines": [{"name": "flux", "y": lv["y"]}], "x": lv.get("x")})
            return {"type": "series", "x_label": "phase", "x": gv.get("x") or [], "panels": panels, "markers": True,
                    "facts": {"candidate score": res.get("candidate_score"), "baseline score": res.get("baseline_score"),
                              "period (days)": (res.get("tce") or {}).get("period_days")}}
        if item == "mask":
            return {"type": "images", "images": [{"src": f"{files}/input.png", "label": "image"}],
                    "facts": {k: f"{v:.0%}" for k, v in (res.get("class_shares") or {}).items()}, "mask": True}
        if mod == "missions":
            it = next((x for x in res.get("review_items") or [] if x["item"] == item), None)
            if it:
                return {"type": "mission", "mission": res.get("mission") or (run.get("input_ref") or {}).get("mission"),
                        "target": it.get("target"), "start": it.get("start"), "end": it.get("end"),
                        "facts": {k: it[k] for k in ("label", "pred", "prob", "outcome", "detector") if k in it}}
    except (KeyError, IndexError, ValueError, FileNotFoundError) as e:
        raise HTTPException(404, f"Evidence not available: {e}") from e
    raise HTTPException(404, "No evidence view for this item")


@router.get("/api/reviews/export")
def export(format: str = Query("csv", pattern="^(csv|json|zip)$"),
           module: str | None = Query(None, pattern="^(telemetry|exoplanet|mars|workbench|missions)$")):
    rows = _all_reviews(module)
    stamp = now_iso()[:19].replace(":", "")
    if format == "json":
        import json
        return Response(json.dumps({"exported_at": now_iso(), "reviews": rows}, indent=2, default=str),
                        media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="reviewed-labels-{stamp}.json"'})
    cols = ["run_id", "module", "analysis", "kind", "item", "verdict", "note", "updated_at", "input", "model_version",
            "chan_id", "kepid", "tce_plnt_num", "dataset_id", "target_id", "file", "detector", "start", "end", "x0", "x1",
            "bbox", "pixels", "area_km2", "period_days", "score"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in cols})
    if format == "csv":
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="reviewed-labels-{stamp}.csv"'})
    # zip: CSV + every corrected terrain mask with its image (AI4Mars ids: 0 soil,1 bedrock,2 sand,3 big rock,255 none)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("reviewed_labels.csv", buf.getvalue())
        n = 0
        for r in rows:
            if r["item"] != "mask":
                continue
            d = RUNS_DIR / r["run_id"]
            img = d / ("input.png" if (d / "input.png").exists() else "source.png")
            if (d / "corrected_mask.png").exists() and img.exists():
                z.write(img, f"terrain/images/{r['run_id']}.png")
                z.write(d / "corrected_mask.png", f"terrain/labels/{r['run_id']}.png")
                n += 1
        z.writestr("README.txt", f"Exported {now_iso()} from Space ML Mission Control.\n"
                   f"{len(rows)} review verdicts in reviewed_labels.csv; {n} corrected terrain masks in terrain/.\n"
                   "Mask values: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 unlabeled (AI4Mars convention).\n")
    mem.seek(0)
    return StreamingResponse(mem, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="reviewed-labels-{stamp}.zip"'})


@router.put("/api/runs/{run_id}/mask")
async def save_corrected_mask(run_id: str, file: UploadFile = File(...), note: str = ""):
    run = worker.get(run_id)
    if not run or run["status"] != "succeeded":
        raise HTTPException(404, "Run not found")
    d = RUNS_DIR / run_id
    pred = d / "mask.png"
    if not pred.exists():
        raise HTTPException(409, "This run has no terrain mask to correct")
    raw = await file.read(20 * 1024 * 1024)
    try:
        m = np.array(Image.open(io.BytesIO(raw)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, "Upload a PNG mask") from e
    m = m[..., 0] if m.ndim == 3 else m
    ref = np.array(Image.open(pred))
    if m.shape != ref.shape:
        raise HTTPException(422, f"Mask is {m.shape[1]}x{m.shape[0]}, expected {ref.shape[1]}x{ref.shape[0]}")
    if not set(np.unique(m).tolist()).issubset({0, 1, 2, 3, 255}):
        raise HTTPException(422, "Mask values must be 0-3 (terrain classes) or 255 (unlabeled)")
    Image.fromarray(m.astype(np.uint8)).save(d / "corrected_mask.png")
    changed = float((m != ref).mean())
    with tx() as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?)",
                  (run_id, "mask", "corrected", note or f"{changed:.1%} of pixels relabeled", now_iso()))
    return {"saved": True, "changed_fraction": changed}
