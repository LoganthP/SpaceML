"""Run manifests: everything another person needs to verify or repeat a result.

    GET /api/runs/{id}/manifest       JSON manifest
    GET /api/runs/{id}/evidence.zip   manifest + Markdown report + full result + CSV tables + images
    POST /api/runs/{id}/repeat        same input, same model version
    GET /api/runs/{id}/reproducibility  compare a repeat with its original
"""
from __future__ import annotations

import csv
import io
import json
import platform
import zipfile
from pathlib import Path

from spaceml.common.fetch import provenance, sha256_file
from spaceml.common.utils import now_iso, read_json, sanitize, write_json
from spaceml.config import RUNS_DIR, TELEMETRY_DIR, UPLOADS_DIR

from . import registry
from .db import tx

APP_VERSION = "3.1.10"
SCHEMA = "smc.run-manifest/1"


def _environment() -> dict:
    env = {"app": APP_VERSION, "python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return env


def _file(path: Path, source: str | None = None) -> dict:
    rec = {"name": path.name, "exists": path.exists()}
    if path.exists():
        rec.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        prov = provenance(path)
        if prov:
            rec["retrieved"] = {k: prov.get(k) for k in ("url", "retrieved_at", "sha256")}
    if source:
        rec["source"] = source
    return rec


def _model_hashes(vdir: Path) -> list[dict]:
    """sha256 of every model file, cached next to the model (keyed by size + mtime)."""
    cache_p = vdir / ".hashes.json"
    cache = read_json(cache_p, {}) or {}
    out, dirty = [], False
    for f in sorted(p for p in vdir.iterdir() if p.is_file() and not p.name.startswith(".")):
        key = f"{f.stat().st_size}:{int(f.stat().st_mtime)}"
        if cache.get(f.name, {}).get("key") != key:
            cache[f.name] = {"key": key, "sha256": sha256_file(f)}
            dirty = True
        out.append({"name": f.name, "bytes": f.stat().st_size, "sha256": cache[f.name]["sha256"]})
    if dirty:
        try:
            write_json(cache_p, cache)
        except OSError:
            pass
    return out


def _data_section(run: dict, result: dict | None) -> dict:
    inp, m = run.get("input_ref") or {}, run["module"]
    if m == "telemetry":
        ch = inp.get("chan_id")
        return {"source": "NASA SMAP & MSL (Curiosity) telemetry with expert anomaly labels "
                          "(Hundman et al., KDD 2018; telemanom release)",
                "files": [_file(TELEMETRY_DIR / "test" / f"{ch}.npy"), _file(TELEMETRY_DIR / "train" / f"{ch}.npy"),
                          _file(TELEMETRY_DIR / "labeled_anomalies.csv")],
                "window": {"split": "test", "steps": (result or {}).get("n_steps")}}
    if m == "spaceweather":
        r = result or {}
        return {"source": "NASA/GSFC SPDF OMNIWeb OMNI2 hourly (King & Papitashvili 2005)",
                "window": r.get("window"), "files": r.get("data_provenance", []),
                "quality": (r.get("quality") or {}).get("status")}
    if m == "mars":
        r = result or {}
        if inp.get("raw_image_id"):
            from spaceml.mars import traverse
            img = next(iter((traverse.RAW_DIR / "img").glob(f"{inp['raw_image_id']}.*")), None)
            return {"source": "NASA/JPL Curiosity raw Navcam image", "image": r.get("source_image"),
                    "files": [_file(img)] if img else [], "location": r.get("location")}
        if inp.get("sample_id"):
            from spaceml.mars import data as md
            p = md.sample_paths(inp["sample_id"])
            return {"source": "NASA/JPL AI4Mars expert-labeled test image", "sample": inp["sample_id"],
                    "files": [_file(v) for v in p.values()], "location": r.get("location")}
        return {"source": "uploaded image", "files": [_file(UPLOADS_DIR / f"{inp.get('upload_id')}.png")],
                "location": r.get("location")}
    if m == "exoplanet":
        if inp.get("dataset_id"):
            return {"source": "uploaded light curve", "dataset_id": inp["dataset_id"], "target_id": inp.get("target_id")}
        return {"source": "NASA Kepler DR24 TCE catalog + MAST light curves", "kepid": inp.get("kepid"),
                "tce_plnt_num": inp.get("tce_plnt_num")}
    if m == "workbench":
        try:
            from spaceml.workbench.store import get
            d = get(inp["dataset_id"])
            return {"source": "uploaded dataset", "dataset": {k: d.get(k) for k in ("id", "name", "kind", "sha256",
                                                                                 "created_at", "summary")}}
        except Exception:  # noqa: BLE001
            return {"source": "uploaded dataset", "dataset": {"id": inp.get("dataset_id"), "note": "deleted"}}
    return {}


def _metrics_section(module: str, r: dict | None) -> tuple[dict | None, dict | None]:
    """(model metrics, baseline description+metrics) for the manifest."""
    if not r:
        return None, None
    if module == "telemetry":
        return r.get("score"), {"name": "z-score threshold", "metrics": r.get("baseline_score")}
    if module == "spaceweather":
        ph = r.get("per_horizon", {})
        model = {h: {k: v.get(k) for k in ("rmse", "mae", "skill", "storm_rmse", "better_hours", "worse_hours",
                                           "first_warning", "forecast_min", "observed_min")} for h, v in ph.items()}
        base = {h: {k: v.get(k) for k in ("rmse_persistence", "mae_persistence", "storm_rmse_persistence",
                                          "persistence_first_warning")} for h, v in ph.items()}
        return model, {"name": r.get("baseline"), "metrics": base}
    if module == "mars":
        ev = r.get("evaluation")
        return ({"class_shares": r.get("class_shares"), **({"miou": ev["model"]["miou"]} if ev else {})},
                {"name": "all-soil", "metrics": ev["baseline"] if ev else None})
    if module == "exoplanet":
        return {"candidate_score": r.get("candidate_score")}, {"name": "gradient boosting on scalar features",
                                                               "metrics": {"score": r.get("baseline_score")}}
    if module == "workbench":
        return r.get("metrics"), None
    return None, None


def build_manifest(run: dict) -> dict:
    result = read_json(RUNS_DIR / run["id"] / "result.json") if run.get("result_path") else None
    with tx() as c:
        reviews = [dict(x) for x in c.execute("SELECT item, verdict, note, updated_at FROM reviews WHERE run_id=? "
                                              "ORDER BY item", (run["id"],))]
    model = None
    if run.get("model_version") and run["module"] in registry.MODULES:
        try:
            vdir = registry.version_dir(run["module"], run["model_version"])
            man = read_json(vdir / "manifest.json", {}) or {}
            ev = next((e for e in registry.evaluations(run["module"]) if e["version"] == run["model_version"]), None)
            model = {"version": run["model_version"], "name": man.get("model_name"), "created_at": man.get("created_at"),
                     "training": {k: man.get(k) for k in ("train_years", "test_years", "horizons", "hparams", "seed",
                                                          "channels", "split") if man.get(k) is not None},
                     "heldout": (ev or {}).get("metrics", {}).get("headline"),
                     "eval_split": (ev or {}).get("metrics", {}).get("eval_split"),
                     "files": _model_hashes(vdir)}
        except FileNotFoundError:
            model = {"version": run["model_version"], "note": "model version no longer installed"}
    metrics, baseline = _metrics_section(run["module"], result)
    inp = run.get("input_ref") or {}
    can_repeat, why = True, None
    if model and model.get("note"):
        can_repeat, why = False, "the model version is no longer installed"
    return sanitize({
        "schema": SCHEMA, "generated_at": now_iso(),
        "run": {k: run.get(k) for k in ("id", "module", "status", "error", "input_label", "created_at", "started_at",
                                        "finished_at", "duration_ms", "origin", "repeat_of")},
        "origin": run.get("origin") or "computed",
        "analysis": inp.get("analysis"),
        "input": inp, "settings": inp.get("params", {k: v for k, v in inp.items() if k not in ("dataset_sha256",)}),
        "data": _data_section(run, result),
        "model": model, "metrics": metrics, "baseline": baseline,
        "checks": {k: result.get(k) for k in ("causality_check", "quality") if result and result.get(k)},
        "reviews": reviews,
        "environment": _environment(),
        "repeat": {"available": can_repeat, "reason": why, "how": f"POST /api/runs/{run['id']}/repeat"},
    })


# ------------------------------------------------------------------ reproducibility
def _key_numbers(module: str, r: dict | None) -> dict:
    if not r:
        return {}
    out: dict = {}
    if module == "telemetry":
        out = {f"score.{k}": v for k, v in (r.get("score") or {}).items()}
        out["flags"] = [(f["start"], f["end"]) for f in r.get("flagged", [])]
    elif module == "spaceweather":
        for h, v in (r.get("per_horizon") or {}).items():
            for k in ("rmse", "rmse_persistence", "better_hours", "worse_hours", "forecast_min"):
                out[f"+{h}h.{k}"] = v.get(k)
    elif module == "mars":
        out = {f"share.{k}": v for k, v in (r.get("class_shares") or {}).items()}
    elif module == "exoplanet":
        out = {"candidate_score": r.get("candidate_score"), "baseline_score": r.get("baseline_score")}
    elif module == "workbench":
        out = {"metrics": json.dumps(r.get("metrics"), sort_keys=True, default=str),
               "regions": len(r.get("regions", [])) if "regions" in r else None}
    return out


def _same(a, b, tol=1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            return a == b
    return a == b


def compare_runs(orig: dict, rep: dict) -> dict:
    ro = read_json(RUNS_DIR / orig["id"] / "result.json") if orig.get("result_path") else None
    rr = read_json(RUNS_DIR / rep["id"] / "result.json") if rep.get("result_path") else None
    if rep["status"] != "succeeded":
        return {"status": "pending" if rep["status"] in ("queued", "running") else rep["status"],
                "original": orig["id"], "repeat": rep["id"]}
    a, b = _key_numbers(orig["module"], ro), _key_numbers(rep["module"], rr)
    diffs = [{"key": k, "original": a.get(k), "repeat": b.get(k)} for k in sorted(set(a) | set(b))
             if not _same(a.get(k), b.get(k))]
    return sanitize({"status": "identical" if not diffs else "differs", "original": orig["id"], "repeat": rep["id"],
                     "same_model": orig.get("model_version") == rep.get("model_version"),
                     "compared": len(set(a) | set(b)), "differences": diffs[:50]})


# ------------------------------------------------------------------ evidence bundle
def _csv(rows: list[list], header: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def evidence_zip(run: dict) -> bytes:
    from .reports import build
    man = build_manifest(run)
    data, md = build(run)
    result = data.get("result")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(man, indent=2))
        z.writestr("report.md", md)
        if result:
            z.writestr("result.json", json.dumps(sanitize(result), indent=1))
        if run["module"] == "spaceweather" and result:
            rows = []
            for h in result["horizons"]:
                for i, t in enumerate(result["issued"]):
                    rows.append([t, h, result["forecast"][str(h)][i], result["persistence"][i],
                                 result["truth"][str(h)][i], result["verdict"][str(h)][i],
                                 result["inputs_complete"][i]])
            z.writestr("forecasts.csv", _csv(rows, ["issued_utc", "horizon_h", "forecast_dst_nT", "persistence_dst_nT",
                                                    "observed_dst_nT_at_valid_time", "model_vs_persistence(+1 better,-1 worse)",
                                                    "inputs_complete"]))
            z.writestr("timeline.csv", _csv([[e["time"], e["kind"], e["label"], e["detail"]] for e in result["timeline"]],
                                            ["time_utc", "kind", "label", "detail"]))
        if run["module"] == "telemetry" and result:
            z.writestr("flagged.csv", _csv([[f["start"], f["end"], f.get("peak_error")] for f in result["flagged"]],
                                           ["start_step", "end_step", "peak_error"]))
        if man["reviews"]:
            z.writestr("reviews.csv", _csv([[r["item"], r["verdict"], r["note"], r["updated_at"]] for r in man["reviews"]],
                                           ["item", "verdict", "note", "updated_at"]))
        rd = RUNS_DIR / run["id"]
        for f in sorted(rd.glob("*.png")) if rd.exists() else []:
            z.write(f, f"images/{f.name}")
        z.writestr("README.txt", (
            f"Evidence bundle for run {run['id']} ({run['module']}) - origin: {man['origin']}.\n"
            "manifest.json: data source + hashes, data window, model version + file hashes, settings, baseline,\n"
            "metrics, reviewer notes and software versions. report.md: human-readable summary.\n"
            "To repeat: open the run in Space ML Mission Control and press 'Repeat run', or\n"
            f"POST /api/runs/{run['id']}/repeat ; then GET /api/runs/<new id>/reproducibility.\n"))
    return buf.getvalue()
