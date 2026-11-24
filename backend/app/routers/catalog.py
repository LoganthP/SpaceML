"""Dataset catalog: dataset readiness and model status, shown separately, with the actions for each.

Nothing here downloads or trains by itself - the catalog only describes and offers buttons.
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from spaceml.common.utils import read_json
from spaceml.config import DATA_DIR, MARS_DIR, MODELS_DIR, UPLOADS_DIR

from .. import jobs, registry

router = APIRouter(prefix="/api", tags=["catalog"])
_size_cache: dict[str, tuple[float, int, int]] = {}


def _size(rel: str) -> tuple[int, int]:
    """(files, bytes) under data/<rel>, cached for 60 s."""
    now = time.time()
    hit = _size_cache.get(rel)
    if hit and now - hit[0] < 60:
        return hit[1], hit[2]
    p = DATA_DIR / rel
    files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else ([p] if p.is_file() else [])
    n, b = len(files), sum(f.stat().st_size for f in files)
    _size_cache[rel] = (now, n, b)
    return n, b


def _model(module: str) -> dict:
    act = registry.active(module)
    if not act:
        return {"ready": False, "version": None}
    man = read_json(Path(act["path"]) / "manifest.json", {}) or {}
    ev = next((e for e in registry.evaluations(module) if e["version"] == act["version"]), None)
    head = (read_json(Path(act["path"]) / "metrics.json", {}) or {}).get("headline") if ev else None
    quick = man.get("run_kind") == "quick" or str(act["version"]).endswith("-quick")
    return {"ready": True, "version": act["version"], "created_at": act.get("created_at"),
            "name": man.get("model_name"), "run_kind": "quick" if quick else man.get("run_kind", "full"),
            "evaluated": ev is not None, "headline": head,
            "origin": "release" if man.get("installed_from") == "release" else man.get("installed_from")}


def _jobs_for(module: str, tasks: list[str]) -> dict:
    """The job running now, and the most recent failure that still matters.

    A failure is not shown when a later job of the same dataset succeeded, when the user dismissed it,
    or when it came from an older app version (its error describes code that no longer exists).
    """
    from ..manifest import APP_VERSION
    act, last_fail, settled = None, None, False
    for j in jobs.list_jobs(80):
        if j["task"] not in tasks:
            continue
        st = j["status"]
        if st in ("queued", "running"):
            act = act or j
            continue
        if settled or last_fail:
            continue
        if st in ("succeeded", "cancelled"):
            settled = True
        elif st in ("failed", "source_unavailable", "interrupted"):
            if j.get("dismissed") or j.get("app_version") != APP_VERSION:
                settled = True       # older failures are superseded too
            else:
                last_fail = j
    if act:
        last_fail = None
    keep = ("id", "task", "title", "status", "progress", "message", "step", "error", "started_at",
            "last_progress_at", "waiting_for", "created_at", "finished_at")
    return {"active_job": {k: act.get(k) for k in keep} if act else None,
            "last_failure": {k: last_fail.get(k) for k in keep} if last_fail else None}


def _entry_telemetry() -> dict:
    from spaceml.telemetry import data
    ready = data.is_downloaded()
    det = {}
    if ready:
        ch = data.load_channels()
        det = {"channels": len(ch), "anomalies": sum(len(c.sequences) for c in ch.values())}
    n, b = _size("telemetry")
    return {"id": "telemetry", "module": "telemetry", "title": "SMAP & MSL spacecraft telemetry",
            "dataset": {"ready": ready, "files": n, "bytes": b, "details": det,
                        "summary": f"{det.get('channels', 0)} channels · {det.get('anomalies', 0)} labeled anomalies" if ready else "not installed"},
            "model": _model("telemetry"),
            "train": [{"task": "telemetry.train", "label": "Quick run", "params": {"mode": "fast"},
                       "note": "fast mode (stride 4, early stopping) on channels without a model; ~1-2 min per channel per core"},
                      {"task": "telemetry.retrain", "label": "Full retrain", "params": {},
                       "note": "every channel into a new version; the current version stays active until it is complete"}],
            "download": None if ready else {"task": "telemetry.download", "label": "Download (needs internet)"},
            **_jobs_for("telemetry", ["telemetry.train", "telemetry.retrain", "telemetry.evaluate", "telemetry.download"])}


def _entry_kepler() -> dict:
    from spaceml.exoplanet import data
    has_cat = data.TCE_PATH.exists()
    views = data.cached_view_ids()
    det = {"tces_with_views": len(views), "stars": len({k for k, _ in views})}
    trainable, why = False, "needs the Kepler catalog and folded light curves"
    if has_cat and views:
        try:
            sel = data.cached_training_rows()
            det.update(planet_candidates=int(sel.planet.sum()), false_positives=int((sel.planet == 0).sum()))
            from spaceml.exoplanet.data import MIN_PER_CLASS
            trainable = det["planet_candidates"] >= MIN_PER_CLASS and det["false_positives"] >= MIN_PER_CLASS
            det["needed_per_class"] = MIN_PER_CLASS
            short = [f"{MIN_PER_CLASS - det[k]} more {k.replace('_', ' ')}" for k in ("planet_candidates", "false_positives")
                     if det[k] < MIN_PER_CLASS]
            why = None if trainable else (f"training needs at least {MIN_PER_CLASS} planet candidates and {MIN_PER_CLASS} false positives "
                                          f"(have {det['planet_candidates']} / {det['false_positives']}): download {' and '.join(short)}")
        except Exception as e:  # noqa: BLE001
            why = str(e)[:200]
    n, b = _size("exoplanet")
    return {"id": "kepler", "module": "exoplanet", "title": "Kepler DR24 catalog + light curves",
            "dataset": {"ready": has_cat and bool(views), "trainable": trainable, "blocked_reason": why,
                        "files": n, "bytes": b, "details": det,
                        "summary": (f"{det['stars']} stars · {det['tces_with_views']} signals folded"
                                    + (f" · {det.get('planet_candidates', 0)} planet candidates / {det.get('false_positives', 0)} false positives"
                                       if "planet_candidates" in det else ""))
                        if views else ("catalog only" if has_cat else "not installed"),
                        "browse_only_label": "Browse only - more labeled signals needed"},
            "model": _model("exoplanet"),
            "train": [{"task": "exoplanet.train", "label": "Quick run", "params": {"mode": "quick", "max_stars": 80, "epochs": 40},
                       "note": "seeded subset of 80 stars - a pipeline check, not the final model"},
                      {"task": "exoplanet.train", "label": "Full training", "params": {"mode": "full", "epochs": 60},
                       "note": "every folded signal in the dataset; minutes on a CPU"}],
            "download": None if (has_cat and trainable) else {"task": "exoplanet.data",
                                                              "label": "Download light curves (needs internet)",
                                                              "params": {"n_stars": 300, "workers": 6}},
            **_jobs_for("exoplanet", ["exoplanet.train", "exoplanet.data", "exoplanet.lightcurves", "exoplanet.catalog",
                                      "exoplanet.setup"])}


def _entry_ai4mars() -> dict:
    from spaceml.mars import local
    from spaceml.mars import data as mdata
    v = local.validate()            # cached in the manifest; re-checks only changed files
    sp = v.get("splits", {})
    tr_c = sp.get("train", {})
    tr, tr_all = tr_c.get("usable", 0), tr_c.get("pairs", 0)
    te = sum(c.get("usable", 0) for k, c in sp.items() if k.startswith("test"))
    try:
        previews = len(mdata.list_samples()) if hasattr(mdata, "list_samples") else \
            len(list((MARS_DIR / "samples").glob("*/input.png")))
    except Exception:  # noqa: BLE001
        previews = 0
    ready = tr >= 16
    resumable = any((d / "last.pt").exists() for d in (MODELS_DIR / "mars").glob(".staging-*")) \
        if (MODELS_DIR / "mars").exists() else False
    n, b = _size("mars/ai4mars")
    idx = local.index() or {}
    probs = {k: c.get("problems") for k, c in sp.items() if c.get("problems")}
    why = None if ready else (v.get("error") or (f"only {tr} of {tr_all} train pairs usable: {probs.get('train')}"
                                                  if tr_all else "needs local train pairs"))
    return {"id": "ai4mars", "module": "mars", "title": "AI4Mars terrain labels (Curiosity NAVCAM)",
            "dataset": {"ready": ready, "trainable": ready, "blocked_reason": why, "files": n, "bytes": b,
                        "details": {"root": v.get("root"), "splits": sp, "source": idx.get("dataset"),
                                    "revision": idx.get("revision"), "image_size": idx.get("size"),
                                    "problem_examples": v.get("problem_examples")},
                        "summary": (f"training: {tr} usable train pairs"
                                    + (f" ({tr_all - tr} excluded)" if tr_all - tr else "")
                                    + f" · evaluation: {te} expert test pairs · preview: {previews} samples")
                                   if tr_all or te else "not installed",
                        "evaluation_ready": te > 0, "preview_samples": previews},
            "model": _model("mars"), "resumable": resumable,
            "train": [{"task": "mars.train", "label": "Quick run", "params": {"mode": "quick"},
                       "note": "300 pairs, 3 epochs, 256 px - a pipeline check (~5-15 min on CPU)"},
                      {"task": "mars.train", "label": "Full training", "params": {"mode": "full"},
                       "note": "every local train pair, 12 epochs, 384 px - hours on a CPU; resumable; a GPU (Colab) is much faster"}],
            "download": ({"task": "mars.dataset", "label": "Download (needs internet)", "params": {"train": "3000"}}
                         if not ready else
                         {"task": "mars.dataset", "label": f"Download more pairs ({tr} → 3000, needs internet)",
                          "params": {"train": "3000"}, "optional": True} if tr < 3000 else None),
            **_jobs_for("mars", ["mars.train", "mars.train_quick", "mars.resume", "mars.evaluate", "mars.dataset"])}


def _entry_omni() -> dict:
    from spaceml.spaceweather import data
    ys = data.cached_years()
    need = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2024}
    ready = need <= set(ys)
    n, b = _size("spaceweather")
    return {"id": "omni2", "module": "spaceweather", "title": "OMNI2 hourly solar wind + Dst",
            "dataset": {"ready": ready, "trainable": need - {2024} <= set(ys), "files": n, "bytes": b,
                        "blocked_reason": None if need - {2024} <= set(ys) else f"missing training/evaluation years: {sorted(need - {2024} - set(ys))}",
                        "details": {"years": ys}, "summary": f"{len(ys)} years ({min(ys)}–{max(ys)})" if ys else "not installed"},
            "model": _model("spaceweather"),
            "train": [{"task": "spaceweather.train", "label": "Train forecaster", "params": {},
                       "note": "gradient boosting on 2010-2016, tested on 2017; May 2024 replay is available if that year is installed"}],
            "download": None if ready else {"task": "spaceweather.download", "label": "Download (needs internet)"},
            **_jobs_for("spaceweather", ["spaceweather.train", "spaceweather.download", "spaceweather.setup", "spaceweather.replay"])}


def _entry_traverse() -> dict:
    from spaceml.mars import traverse
    ready = traverse.WAYPOINTS_PATH.exists()
    n, b = _size("mars/traverse")
    return {"id": "traverse", "module": "mars", "title": "Curiosity traverse waypoints",
            "dataset": {"ready": ready, "files": n, "bytes": b, "details": {},
                        "summary": "waypoints + cached image lists" if ready else "not installed"},
            "model": None, "train": [],
            "download": None if ready else {"task": "mars.traverse", "label": "Download (needs internet)"},
            **_jobs_for("mars", ["mars.traverse"])}


def _entry_flood() -> dict:
    d = DATA_DIR / "workbench_samples"
    scenes = sorted(p.name for p in d.glob("*.zip")) if d.exists() else []
    n, b = _size("workbench_samples")
    return {"id": "flood", "module": "workbench", "title": "Before/after satellite scenes (change detection)",
            "dataset": {"ready": bool(scenes), "files": n, "bytes": b, "details": {"scenes": scenes},
                        "summary": ", ".join(s.removesuffix(".zip") for s in scenes) if scenes else "not installed"},
            "model": None, "train": [], "open_in_workbench": scenes,
            "download": None if scenes else {"task": "workbench.samples", "label": "Download scenes (needs internet)"},
            **_jobs_for("workbench", ["workbench.samples"])}


def _train_actions(e: dict) -> dict:
    """Training consumes local data. Downloads are independent actions."""
    ds = e.get("dataset") or {}
    trainable = ds.get("trainable", ds.get("ready", False))
    for action in e.get("train") or []:
        action.update(downloads_first=False,
                      blocked=None if trainable else ds.get("blocked_reason") or "Install the dataset first.")
    return e


def _entry_mission(mid: str) -> dict:
    from spaceml.missions import catalog as mcat
    from spaceml.missions import models as mmodels
    from spaceml.missions import store as mstore
    meta = mcat.get(mid)
    man = mstore.manifest(mid) or {}
    inst = mstore.installed(mid)
    act = mmodels.active(mid)
    c = man.get("counts") or {}
    dates = man.get("date_range") or []
    coverage = f" · {dates[0][:10]} → {dates[-1][:10]}" if len(dates) >= 2 and all(dates) else ""
    summ = f"{man.get('scope')}{coverage}" if inst else "importer available - no local data installed"
    model = None
    if meta.get("train"):
        model = {"ready": bool(act), "stale": bool(act and act.get("stale")),
                 "version": act and act["version"], "run_kind": act and act["run_kind"],
                 "name": act and act["manifest"].get("model_name"), "evaluated": bool(act and act.get("metrics")),
                 "headline": act and (act.get("metrics") or {}).get("headline")}
    train = []
    if meta.get("train"):
        train = [{"task": f"missions.{mid}.train", "label": "Quick train", "params": {"mode": "quick"},
                  "note": meta["train"]["quick_note"]},
                 {"task": f"missions.{mid}.train", "label": "Full training", "params": {"mode": "full"},
                  "note": "more trees / windows on the same installed data"}]
    dl = {"task": "missions.opssat.download", "label": "Download OPSSAT-AD v2 (needs internet)"} \
        if mid == "opssat" and (not inst or (man.get("source") or {}).get("version") != "v2") else None
    return {"id": f"mission-{mid}", "module": "missions", "mission": mid, "title": meta["name"],
            "dataset": {"ready": inst, "trainable": inst and (man.get("training_readiness") or {}).get("trainable", True),
                        "files": man.get("prepared_files", 0),
                        "bytes": man.get("prepared_bytes", 0), "summary": summ,
                        "browse_only_label": "Installed - training disabled",
                        "blocked_reason": ("no local data yet - press Install data" if not inst else
                                           (man.get("training_readiness") or {}).get("blocked_reason")),
                        "details": {"counts": c, "source": man.get("source"), "time_basis": man.get("time_basis")}},
            "model": model, "train": train, "download": dl,
            "analysis_only": not meta.get("train"),
            "notes": None if meta.get("train") else f"No labels to train on: {meta.get('detect')} flags intervals for review.",
            "setup_help": f"Command line (inside the app container): {meta['prepare']}",
            **_jobs_for("missions", [f"missions.{mid}.train", f"missions.{mid}.install"]
                        + (["missions.opssat.download"] if mid == "opssat" else []))}


PAGES = {"telemetry": ("/telemetry", "Open anomaly detection"), "kepler": ("/exoplanet", "Open Exoplanet Finder"),
         "ai4mars": ("/mars", "Open a Mars sample"), "omni2": ("/replay", "Open the May 2024 replay"),
         "traverse": ("/mars/traverse", "Open the traverse map"), "flood": ("/workbench", "Open Data Workbench")}


def _latest_run(module: str, mission: str | None = None) -> str | None:
    from ..db import tx
    with tx() as c:
        rows = c.execute("SELECT id, input_ref FROM runs WHERE module=? AND status='succeeded' ORDER BY created_at DESC LIMIT 50",
                         (module,)).fetchall()
    for r in rows:
        if mission is None or f'"mission": "{mission}"' in r["input_ref"] or f'"mission":"{mission}"' in r["input_ref"]:
            return r["id"]
    return None


def _status(e: dict) -> dict:
    """ONE definition of dataset / model state and the next step, used by every view (Overview, Data &
    Training, module pages, mission pages)."""
    ds, m = e.get("dataset") or {}, e.get("model")
    mission = e.get("mission")
    page, open_label = (f"/telemetry?mission={mission}", "Open mission") if mission else PAGES.get(e["id"], ("/", "Open"))
    installed = bool(ds.get("ready"))
    trainable = installed and ds.get("trainable", True) is not False
    if not installed:
        data, label = "missing", "Not installed"
    elif m is not None and not trainable:
        data, label = "browse_only", ds.get("browse_only_label") or "Browse only - not enough data to train"
    else:
        data, label = "installed", "Installed"
    model = "not_applicable" if m is None else "available" if m.get("ready") else "not_trained"
    job = e.get("active_job")
    repair = None
    if mission and installed:
        from .missions import repair_needed
        repair = repair_needed(mission)
        if repair:
            data, label = "needs_repair", "Installed - needs re-import"
    if job:
        nxt = {"kind": "wait", "label": f"{job['title']} - {job['status']}"}
    elif repair:
        nxt = {"kind": "task", "label": repair["label"], "task": repair["task"], "params": repair["params"]}
    elif data == "missing":
        if mission:
            nxt = {"kind": "link", "label": "Install data", "to": f"{page}#install"}
        elif e.get("download"):
            nxt = {"kind": "task", "label": e["download"]["label"], **{k: e["download"][k] for k in ("task", "params") if k in e["download"]}}
        else:
            nxt = {"kind": "none", "label": ds.get("blocked_reason") or "not available in this installation"}
    elif e.get("resumable"):
        nxt = {"kind": "task", "label": "Resume training", "task": "mars.resume"}
    elif model == "not_trained" and trainable and e.get("train"):
        t = e["train"][0]
        nxt = {"kind": "task", "label": f"{t['label']} (next step)", "task": t["task"], "params": t.get("params", {})}
    elif data == "browse_only" and e.get("download"):
        nxt = {"kind": "task", "label": e["download"]["label"], **{k: e["download"][k] for k in ("task", "params") if k in e["download"]}}
    else:
        rid = _latest_run("missions" if mission else e.get("module", ""), mission) if e.get("module") else None
        nxt = {"kind": "link", "label": "View latest result", "to": f"{page.split('?')[0]}?{'mission=' + mission + '&' if mission else ''}run={rid}"} \
            if rid else {"kind": "link", "label": open_label, "to": page}
    if data == "missing":
        rd = "missing"
    elif model == "not_applicable":
        rd = "analysis_only"
    elif model == "available":
        rd = "model_available"
    elif data == "installed":
        rd = "ready_to_train"
    else:
        rd = "installed"                    # browse only / needs re-import: data is here, training is not possible yet
    return {"data": data, "data_label": label, "model": model, "next": nxt, "page": page,
            "readiness": rd, "readiness_label": READINESS[rd], "repair": repair and repair["reason"]}


READINESS = {"missing": "Missing", "installed": "Installed", "ready_to_train": "Ready to train",
             "model_available": "Model available", "analysis_only": "Analysis only"}


def _summary(entries: list[dict]) -> dict:
    st = [e["status"] for e in entries if "status" in e]
    return {"datasets": len(st), "installed": sum(s["data"] != "missing" for s in st),
            "trainable": sum(s["data"] == "installed" and s["model"] != "not_applicable" for s in st),
            "with_models": sum(s["model"] != "not_applicable" for s in st),
            "models_available": sum(s["model"] == "available" for s in st),
            "browse_only": sum(s["data"] == "browse_only" for s in st),
            "needs_repair": sum(s["data"] == "needs_repair" for s in st),
            "jobs_active": sum(1 for e in entries if e.get("active_job")),
            "readiness": {k: sum(s.get("readiness") == k for s in st) for k in READINESS},
            "readiness_labels": READINESS,
            "definitions": {"missing": "no data on this machine yet", "analysis_only": "installed; analysed with rolling "
                            "detection, no model needed", "ready_to_train": "installed and meets the trainer's requirements, "
                            "no model yet", "model_available": "a trained model for this data version can produce results",
                            "installed_only": "installed and explorable, but training is not possible (reason on the card)","installed": "data is on this machine and can be explored",
                            "trainable": "installed data that meets the trainer's requirements",
                            "models_available": "a trained model exists and can produce results"}}


@router.get("/catalog")
def catalog():
    from spaceml import release
    entries = []
    from spaceml.missions import catalog as mcat
    mission_entries = [(lambda m=m: _entry_mission(m)) for m in mcat.ORDER]
    for f in (_entry_telemetry, _entry_kepler, _entry_ai4mars, _entry_omni, _entry_traverse, _entry_flood, *mission_entries):
        try:
            e = _train_actions(f())
            e["status"] = _status(e)
            entries.append(e)
        except Exception as e:  # noqa: BLE001 - one broken dataset must not hide the others
            entries.append({"id": f.__name__.removeprefix("_entry_"), "title": f.__name__, "error": str(e)[:300],
                            "dataset": {"ready": False, "summary": "could not be read"}, "model": None, "train": []})
    rel = release.status()
    rel_job = _jobs_for("release", ["release.install", "release.build"])
    heavy = [j for j in jobs.list_jobs(40) if j["task"] in jobs.HEAVY_TASKS and j["status"] in ("queued", "running")]
    return {"datasets": entries, "summary": _summary(entries), "release": {**rel, **rel_job},
            "heavy_jobs": [{"id": j["id"], "title": j["title"], "status": j["status"]} for j in heavy],
            "free_bytes": shutil.disk_usage(DATA_DIR).free}


@router.post("/workbench/samples/{name}")
async def open_sample(name: str):
    """Create a Workbench dataset from a bundled scene archive (no upload needed)."""
    from spaceml.workbench import store
    src = DATA_DIR / "workbench_samples" / Path(name).name
    if src.suffix != ".zip" or not src.exists():
        raise HTTPException(404, "Unknown sample")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS_DIR / f"wb_{uuid.uuid4().hex}_{src.name}"
    shutil.copyfile(src, tmp)
    try:
        return await run_in_threadpool(store.create, tmp, src.name, "images")
    except store.WorkbenchError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)
