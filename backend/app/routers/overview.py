"""Dashboard overview, dataset catalog, model registry and evaluation results."""
from __future__ import annotations

from fastapi import APIRouter

from spaceml.config import DATA_DIR, MODELS_DIR

from .. import jobs, registry, worker
from ..db import tx

router = APIRouter(prefix="/api", tags=["overview"])

OUTPUT_LABEL = {"telemetry": "Flag for review", "exoplanet": "Candidate score", "mars": "Terrain classification",
                "spaceweather": "Dst forecast"}


def _telemetry_dataset() -> dict:
    from spaceml.telemetry import data
    info = {"module": "telemetry", "name": "NASA SMAP & MSL telemetry",
            "description": "Real telemetry from the SMAP satellite and the Curiosity rover with 105 "
                           "expert-labeled anomalies (Hundman et al., KDD 2018).",
            "source_url": "https://github.com/khundman/telemanom",
            "setup": ["python -m spaceml.telemetry.data download"], "ready": data.is_usable(), "stats": {}}
    if data.LABELS_PATH.exists():
        chans = data.load_channels()
        info["stats"] = {
            "Channels": len(chans),
            "SMAP channels": sum(c.spacecraft == "SMAP" for c in chans.values()),
            "MSL channels": sum(c.spacecraft == "MSL" for c in chans.values()),
            "Labeled anomalies": sum(len(c.sequences) for c in chans.values()),
            "Channels on disk": sum(data.has_channel(c) for c in chans),
        }
    return info


def _exoplanet_dataset() -> dict:
    from spaceml.exoplanet import data
    info = {"module": "exoplanet", "name": "Kepler Q1-Q17 DR24 TCEs + MAST light curves",
            "description": "Threshold-crossing events flagged by the Kepler pipeline, labeled PC / AFP / NTP, "
                           "with each star's long-cadence light curve from MAST.",
            "source_url": "https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html",
            "setup": ["python -m spaceml.exoplanet.data tce",
                      "python -m spaceml.exoplanet.data lightcurves --n-stars 300"],
            "ready": data.TCE_PATH.exists(), "stats": {}}
    from spaceml.exoplanet import uploads
    info["uploaded_datasets"] = len(uploads.list_all())
    if data.TCE_PATH.exists():
        info["catalog"] = data.catalog_info()
        df = data.tce_table()
        vc = df.av_training_set.value_counts()
        cached = data.cached_view_ids()
        info["stats"] = {"TCEs": len(df), "Planet candidates (PC)": int(vc.get("PC", 0)),
                         "False positives (AFP)": int(vc.get("AFP", 0)),
                         "Non-transiting (NTP)": int(vc.get("NTP", 0)),
                         "Cached folded views": len(cached),
                         "Cached stars": len({k for k, _ in cached})}
        info["ready"] = len(cached) > 0
    return info


def _mars_dataset() -> dict:
    from spaceml.mars import data
    samples = data.list_samples()
    return {"module": "mars", "name": "NASA JPL AI4Mars (Curiosity NAVCAM)",
            "description": "Rover images with crowdsourced terrain labels and expert-labeled test sets "
                           "(soil, bedrock, sand, big rock).",
            "source_url": "https://huggingface.co/datasets/hassanjbara/AI4MARS",
            "setup": ["python -m spaceml.mars.data samples --n 36"],
            "ready": len(samples) > 0,
            "stats": {"Curated expert-labeled samples": len(samples),
                      "Samples containing sand": sum(s["label_shares"]["sand"] > 0.02 for s in samples)}}


def _spaceweather_dataset() -> dict:
    from spaceml.spaceweather import data
    yrs = data.cached_years()
    return {"module": "spaceweather", "name": "NASA OMNIWeb OMNI2 hourly solar wind + indices",
            "description": "Near-Earth solar-wind plasma and IMF measured at L1 (ACE, Wind, DSCOVR) with the Dst, Kp, "
                           "AE and F10.7 indices, one record per hour since 1963.",
            "source_url": "https://omniweb.gsfc.nasa.gov/ow.html",
            "setup": ["python -m spaceml.spaceweather.data download"], "ready": bool(yrs),
            "stats": {"Years cached": len(yrs), "Hourly records": len(yrs) * 8766,
                      "Storms in catalog": len(data.EVENTS)} if yrs else {}}


def datasets() -> list[dict]:
    return [_telemetry_dataset(), _exoplanet_dataset(), _mars_dataset(), _spaceweather_dataset()]


@router.get("/health")
def health():
    from ..manifest import APP_VERSION
    return {"status": "ok", "mode": "local", "version": APP_VERSION,
            "read_only": [], "capabilities": {"write": True, "train": True, "live": True},
            "data_dir": str(DATA_DIR), "models_dir": str(MODELS_DIR)}


@router.get("/datasets")
def get_datasets():
    return datasets()


@router.get("/models")
def get_models():
    return {m: {"active": (registry.active(m) or {}).get("version"), "versions": registry.versions(m)}
            for m in registry.MODULES}


@router.get("/evaluation")
def get_evaluation():
    out = []
    for m in registry.MODULES:
        act = registry.active(m)
        evals = registry.evaluations(m)
        active_eval = next((e for e in evals if act and e["version"] == act["version"]), None)
        out.append({"module": m, "output_label": OUTPUT_LABEL[m],
                    "active_version": act["version"] if act else None,
                    "active": active_eval["metrics"] if active_eval else None,
                    "history": [{"version": e["version"], "created_at": e["created_at"],
                                 "headline": e["metrics"].get("headline")} for e in evals]})
    return out


@router.post("/admin/rescan")
def rescan():
    registry.scan()
    return get_models()


def _activity(days: int = 84) -> list[dict]:
    with tx() as c:
        rows = c.execute("SELECT substr(created_at,1,10) AS d, module, COUNT(*) AS n FROM runs "
                         "WHERE created_at >= date('now', ?) GROUP BY d, module", (f"-{days} days",)).fetchall()
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["d"], {"date": r["d"], "total": 0})
        by[r["d"]][r["module"]] = r["n"]
        by[r["d"]]["total"] += r["n"]
    return sorted(by.values(), key=lambda x: x["date"])


def _kepler_snippet() -> dict | None:
    """A real folded transit from the local cache (prefers a planet candidate)."""
    from spaceml.common.utils import clean_floats
    from spaceml.exoplanet import data
    try:
        df = data.tce_table(labeled_only=True)
    except FileNotFoundError:
        return None
    cached = data.cached_view_ids()
    if not cached:
        return None
    pcs = df[(df.planet == 1)]
    pcs = pcs[[(int(k), int(p)) in cached for k, p in zip(pcs.kepid, pcs.tce_plnt_num)]]
    pick = pcs.sort_values("tce_model_snr", ascending=False).head(1)
    k, pn = (int(pick.kepid.iloc[0]), int(pick.tce_plnt_num.iloc[0])) if len(pick) else sorted(cached)[0]
    try:
        v = data.get_views(k, pn, allow_download=False)
    except Exception:  # noqa: BLE001
        return None
    row = data.tce_row(k, pn)
    return {"kepid": k, "tce_plnt_num": pn, "label": row.av_training_set, "period_days": float(row.tce_period),
            "depth_ppm": float(row.tce_depth), "local": clean_floats(v["local"], 3)}


def _mars_terrain() -> dict | None:
    from spaceml.mars import data
    samples = data.list_samples()
    if not samples:
        return None
    tot = {c: 0.0 for c in data.CLASSES}
    for s_ in samples:
        for c in data.CLASSES:
            tot[c] += s_["label_shares"].get(c, 0) * s_["labeled_fraction"]
    z = sum(tot.values()) or 1
    return {"images": len(samples), "shares": {c: v / z for c, v in tot.items()}}


def _space_weather() -> dict | None:
    """Dst during the most recent cached catalogued storm (for the overview sparkline)."""
    try:
        from spaceml.spaceweather import data
        evs = [e for e in data.EVENTS if int(e["start"][:4]) in set(data.cached_years())]
        if not evs:
            return None
        e = evs[-1]
        d = data.load(e["start"], e["end"])
        dst = d["Dst"]
        with tx() as c:
            n = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='sw_annotations'").fetchone()[0]
            notes = c.execute("SELECT COUNT(*) FROM sw_annotations").fetchone()[0] if n else 0
        return {"event": e["name"], "start": e["start"], "dst": [None if v != v else round(float(v), 1) for v in dst.to_numpy()],
                "min_dst": float(dst.min()) if dst.notna().any() else None, "max_kp": float(d["Kp"].max()) if d["Kp"].notna().any() else None,
                "years": len(data.cached_years()), "annotations": notes}
    except Exception:  # noqa: BLE001 - overview must never fail because of one tile
        return None


def _workbench() -> dict:
    from spaceml.workbench.store import list_all
    ds = list_all()
    kinds: dict[str, int] = {}
    for d in ds:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    with tx() as c:
        n = c.execute("SELECT COUNT(*) FROM runs WHERE module='workbench'").fetchone()[0]
        reviewed = c.execute("SELECT verdict, COUNT(*) n FROM reviews GROUP BY verdict").fetchall()
    return {"datasets": len(ds), "by_kind": kinds, "runs": n, "reviews": {r["verdict"]: r["n"] for r in reviewed},
            "recent": ds[:4]}


@router.get("/overview")
def overview():
    ds = {d["module"]: d for d in datasets()}
    with tx() as c:
        counts = {r["module"]: dict(r) for r in c.execute(
            "SELECT module, COUNT(*) AS total, SUM(status='succeeded') AS succeeded, "
            "SUM(status='failed') AS failed FROM runs GROUP BY module").fetchall()}
    modules = []
    for m in registry.MODULES:
        act = registry.active(m)
        ev = next((e for e in registry.evaluations(m) if act and e["version"] == act["version"]), None)
        recent = worker.list_runs(m, 1)
        modules.append({
            "module": m, "output_label": OUTPUT_LABEL[m],
            "dataset_ready": ds[m]["ready"], "dataset_stats": ds[m]["stats"], "setup": ds[m]["setup"],
            "model_version": act["version"] if act else None,
            "model_name": act["model_name"] if act else None,
            "headline": ev["metrics"].get("headline") if ev else None,
            "runs": counts.get(m, {"total": 0, "succeeded": 0, "failed": 0}),
            "last_run": recent[0] if recent else None,
        })
    with tx() as c:
        status = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM runs GROUP BY status")}
    return {"modules": modules, "recent_runs": worker.list_runs(None, 8), "activity": _activity(),
            "run_status": status, "kepler_snippet": _kepler_snippet(), "mars_terrain": _mars_terrain(),
            "jobs": jobs.list_jobs(6), "workbench": _workbench(), "space_weather": _space_weather()}
