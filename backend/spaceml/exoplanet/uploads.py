"""User-uploaded light-curve datasets: storage, parsing and analysis."""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ..common.utils import clean_floats, now_iso, read_json, write_json
from ..config import STORAGE_DIR
from . import data, ingest, search, system

DATASETS_DIR = STORAGE_DIR / "datasets"


def _dir(dataset_id: str) -> Path:
    if not dataset_id.isalnum():
        raise ValueError("Invalid dataset id")
    return DATASETS_DIR / dataset_id


def _preview(t: np.ndarray, y: np.ndarray, n: int = 1500) -> dict:
    if len(t) <= n:
        return {"x": clean_floats(t, 5), "y": clean_floats(y, 6)}
    edges = np.linspace(0, len(t), n + 1).astype(int)
    xs = [float(np.mean(t[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    ys = [float(np.median(y[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return {"x": clean_floats(xs, 5), "y": clean_floats(ys, 6)}


def create(raw_path: Path, filename: str, combine: bool = False) -> dict:
    """Parse an uploaded file (or archive) into targets. Raises ingest.IngestError for bad input."""
    t0 = time.time()
    did = uuid.uuid4().hex[:12]
    root = _dir(did)
    (root / "targets").mkdir(parents=True, exist_ok=True)
    raw_dest = root / f"raw_{Path(filename).name}"
    shutil.move(str(raw_path), raw_dest)
    work = root / "extracted"
    work.mkdir()
    try:
        files = ingest.expand(raw_dest, work)
        targets, warnings, skipped = ingest.parse_files(files, combine=combine)
        records = []
        for tg in targets[:500]:
            t, y, e = tg.arrays()
            arrays = {"time": t, "flux": y.astype(np.float32)}
            if e is not None:
                arrays["err"] = e.astype(np.float32)
            np.savez_compressed(root / "targets" / f"{tg.id}.npz", **arrays)
            meta: dict = {}
            for s in tg.segments:
                for k, v in s.meta.items():
                    meta.setdefault(k, v)
            span = float(t.max() - t.min())
            cad = float(np.median(np.diff(t)) * 1440) if len(t) > 1 else 0.0
            records.append({
                "id": tg.id, "name": tg.name, "n_points": int(len(t)), "span_days": span,
                "cadence_minutes": cad, "segments": len(tg.segments),
                "mission": next((s.mission for s in tg.segments if s.mission), None),
                "time_system": tg.segments[0].time_system, "columns": tg.segments[0].columns,
                "sources": [s.source for s in tg.segments][:50], "star": meta,
                "searchable": span >= 1.0 and len(t) >= 100,
                "note": None if span >= 1.0 else "Spans less than 1 day - too short to search for transits",
                "preview": _preview(t, y),
                "noise_ppm": float(np.std(y) * 1e6),
            })
        manifest = {"id": did, "filename": Path(filename).name, "created_at": now_iso(),
                    "size_bytes": raw_dest.stat().st_size, "n_files": len(files), "combine": combine,
                    "targets": records, "warnings": warnings[:50], "skipped": skipped[:50],
                    "parse_seconds": round(time.time() - t0, 2)}
        write_json(root / "manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def list_all() -> list[dict]:
    if not DATASETS_DIR.exists():
        return []
    out = []
    for d in DATASETS_DIR.iterdir():
        m = read_json(d / "manifest.json")
        if m:
            out.append({**m, "targets": [{k: v for k, v in t.items() if k != "preview"} for t in m["targets"]]})
    return sorted(out, key=lambda m: m["created_at"], reverse=True)


def get(dataset_id: str) -> dict:
    m = read_json(_dir(dataset_id) / "manifest.json")
    if not m:
        raise FileNotFoundError("Dataset not found")
    return m


def delete(dataset_id: str) -> None:
    shutil.rmtree(_dir(dataset_id), ignore_errors=True)


def analyze(dataset_id: str, target_id: str, version_dir: Path | None, period: float | None = None,
            t0: float | None = None, duration_hr: float | None = None) -> dict:
    m = get(dataset_id)
    rec = next((t for t in m["targets"] if t["id"] == target_id), None)
    if rec is None:
        raise ValueError(f"Target {target_id} not found in dataset")
    z = np.load(_dir(dataset_id) / "targets" / f"{target_id}.npz")
    t, y = z["time"].astype(np.float64), z["flux"].astype(np.float64)
    bls = None
    if period and t0 is not None and duration_hr:
        eph_src = "entered by you"
    else:
        if not rec.get("searchable", True):
            raise ValueError(rec.get("note") or "This light curve is too short to search for transits")
        bls = search.bls_search(t, y)
        b = bls["best"]
        period, t0, duration_hr = b["period_days"], b["t0"], b["duration_hours"]
        eph_src = "Box Least Squares search"
    views = data.make_views(t, y, float(period), float(t0), float(duration_hr))
    depth = float(views["measured_depth_ppm"])
    if bls and depth <= 0:
        depth = bls["best"]["depth_ppm"]
    from .infer import score, view_payload  # lazy: keeps torch out of listing calls
    cnn, base, _ = score(views, float(period), float(duration_hr), max(depth, 1.0), version_dir)
    star_meta = rec.get("star") or {}
    st = system.star(teff=star_meta.get("teff"), radius=star_meta.get("radius"), logg=star_meta.get("logg"))
    pl = system.planet(float(period), max(depth, 1.0), float(duration_hr), st)
    sky = {"ra": star_meta["ra"], "dec": star_meta["dec"]} if "ra" in star_meta and "dec" in star_meta else None
    return {
        "source": "upload", "dataset_id": dataset_id, "dataset_name": m["filename"], "target_id": target_id,
        "target_name": rec["name"], "name": None, "mission": rec.get("mission"),
        "candidate_score": cnn, "baseline_score": base, "model_available": cnn is not None,
        "catalog_label": None, "split": "unseen" if cnn is not None else "no model",
        "ephemeris_source": eph_src,
        "tce": {"period_days": float(period), "epoch": float(t0), "epoch_system": rec.get("time_system"),
                "duration_hours": float(duration_hr), "depth_ppm": depth,
                "model_snr": bls["best"]["snr"] if bls else None, "planet_radius_earth": pl["rp_earth"]},
        "bls": bls, **view_payload(views, float(period)),
        "raw_preview": rec["preview"], "data_summary": {k: rec.get(k) for k in
                                                        ("n_points", "span_days", "cadence_minutes", "segments", "time_system")},
        "system": {"star": st, "planets": [{"id": rec["name"], "label": None, "selected": True, **pl}]},
        "sky": sky,
    }
