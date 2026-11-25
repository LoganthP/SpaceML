"""Regenerate mission snapshots from original MAST products, never altering starter archives.

Run with optional training dependencies installed. Network access is limited to
mast.stsci.edu; Vercel builds continue using the offline static-data compiler.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "backend/static_data"


def main():
    index = json.loads((CAPSULE / "index.json").read_text(encoding="utf-8"))
    def read(route):
        return json.loads((CAPSULE / "responses" / index["responses"][route]["file"]).read_text(encoding="utf-8"))

    original = read("/api/missions/webb/items?limit=5000")["items"]
    replaced = {r["id"] for r in read("/api/runs?limit=500") if r.get("module") == "missions" and
                r.get("input_ref", {}).get("mission") in ("opssat", "webb")}
    for collection in ("responses", "files"):
        for route in list(index[collection]):
            if any(route.startswith(f"/api/runs/{rid}") or f"run_id={rid}" in route for rid in replaced):
                del index[collection][route]
    names = sorted({name for item in original for name in item["files"]})
    raw = ROOT / "data/verified-mast-webb"
    raw.mkdir(parents=True, exist_ok=True)
    def download(name):
        if Path(name).name != name or not name.endswith(".fits"):
            raise ValueError("Unexpected MAST product name")
        path = raw / name
        url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/" + name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read(30_000_001)
            if not payload.startswith(b"SIMPLE  =") or len(payload) > 30_000_000:
                raise ValueError(f"Invalid FITS response: {name}")
            path.write_bytes(payload)
        return {"file": name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        provenance = list(pool.map(download, names))

    with tempfile.TemporaryDirectory(prefix="smc-mission-publish-") as temporary:
        base = Path(temporary)
        os.environ.update(SMC_MODE="local", SMC_AUTO_INSTALL="off", SPACEML_AUTOSETUP="off",
                          SPACEML_BOOTSTRAP="0", SPACEML_DATA_DIR=str(base / "data"),
                          SPACEML_MODELS_DIR=str(base / "models"), SPACEML_STORAGE_DIR=str(base / "storage"))
        sys.path.insert(0, str(ROOT / "backend"))
        import numpy as np
        from fastapi.testclient import TestClient
        import logging
        logging.getLogger("httpx2").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        from app.local_main import app
        from app import worker
        from app.routers import catalog as api_catalog
        from spaceml import release
        from spaceml.missions import models, pointing, opssat, store
        from spaceml.common.utils import sanitize

        def write(route, body):
            data = json.dumps(sanitize(body), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            name = hashlib.sha256(data).hexdigest() + ".json"
            (CAPSULE / "responses" / name).write_bytes(data)
            index["responses"][route] = {"file": name, "type": "application/json"}

        def capture(client, route, params=None):
            from urllib.parse import urlencode
            response = client.get(route, params=params or {})
            response.raise_for_status()
            body = response.json()
            # Remove temporary training paths from published JSON.
            body = json.loads(json.dumps(body).replace(str(base).replace("\\", "\\\\"), "<local>"))
            write(route + ("?" + urlencode(sorted(params.items())) if params else ""), body)
            return body

        with TestClient(app) as client:
            release.install(ROOT / "release/bundle")
            pointing.prepare_webb(raw)
            timing = []
            for item in store.items("webb"):
                times = store.load("webb", item["id"])["t"]
                diffs = np.diff(times)
                cadence = float(np.median(diffs))
                duration = float(times[-1] - times[0])
                if not np.all(np.isfinite(times)) or not np.all(diffs > 0) or not 0.03 < cadence < 0.1 or duration < 1:
                    raise ValueError(f"Invalid Webb timing: {item['id']}, {cadence=}, {duration=}")
                timing.append({"visit": item["id"], "samples": len(times), "duration_s": duration,
                               "median_spacing_s": cadence, "min_spacing_s": float(diffs.min()),
                               "max_spacing_s": float(diffs.max()), "time_unit": item["time_unit"]})
            if pointing.webb_time_problem(store.items("webb")):
                raise ValueError("Webb timing validation still requires repair")
            print(json.dumps(timing, indent=2), flush=True)
            new_runs = []
            for mission in ("opssat", "webb"):
                stage = models.Staging(mission, True)
                with stage as output:
                    metrics = (opssat.train(output, quick=True, seed=2137, n_jobs=2) if mission == "opssat"
                               else pointing.train(mission, output, quick=True, seed=2137, n_jobs=2))
                print(f"Trained {mission}: {metrics.get('headline')}", flush=True)
                run = worker.submit("missions", {"mission": mission, "action": "evaluate"}, f"{mission} validated evaluation")
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    run = worker.get(run["id"])
                    if run["status"] not in ("queued", "running"):
                        break
                    time.sleep(0.1)
                if run["status"] != "succeeded":
                    raise RuntimeError(run)
                new_runs.append(run)
                for suffix in ("", "/manifest", "/reviews"):
                    capture(client, f"/api/runs/{run['id']}" + suffix)
                for fmt in ("json", "md"):
                    route = f"/api/runs/{run['id']}/report"
                    response = client.get(route, params={"format": fmt})
                    response.raise_for_status()
                    if fmt == "json":
                        write(route + "?format=json", response.json())
                    else:
                        name = hashlib.sha256(response.content).hexdigest() + ".md"
                        (ROOT / "frontend/public/data/assets" / name).write_bytes(response.content)
                        index["files"][route + "?format=md"] = {"url": "/data/assets/" + name, "type": "text/markdown"}
            catalog = read("/api/catalog")
            for i, entry in enumerate(catalog["datasets"]):
                if entry.get("mission") in ("opssat", "webb", "hubble"):
                    fresh = api_catalog._train_actions(api_catalog._entry_mission(entry["mission"]))
                    fresh["status"] = api_catalog._status(fresh)
                    catalog["datasets"][i] = fresh
            catalog["summary"] = api_catalog._summary(catalog["datasets"])
            from spaceml.missions.availability import catalog_entries, SOURCES
            catalog["datasets"].extend(e for e in catalog_entries() if e["id"] not in {x["id"] for x in catalog["datasets"]})
            catalog["summary"] = api_catalog._summary(catalog["datasets"])
            write("/api/catalog", catalog)
            capture(client, "/api/missions")
            for mission in SOURCES:
                capture(client, f"/api/missions/{mission}")
            for mission in ("opssat", "hubble", "webb"):
                capture(client, f"/api/missions/{mission}")
                capture(client, f"/api/missions/{mission}/items", {"limit": 5000})
                if models.active(mission):
                    capture(client, f"/api/missions/{mission}/predictions")
                for item in store.items(mission):
                    if mission == "opssat" and f"/api/missions/opssat/series?item={item['id']}" in index["responses"]:
                        continue
                    capture(client, f"/api/missions/{mission}/series", {"item": item["id"]})
            runs = new_runs + [r for r in read("/api/runs?limit=500") if not
                               (r.get("module") == "missions" and r.get("input_ref", {}).get("mission") in ("opssat", "webb"))]
            write("/api/runs?limit=500", runs)
            write("/api/runs", runs)
            queue = capture(client, "/api/reviews/queue", {"limit": 1000})
            for item in queue["items"]:
                capture(client, "/api/reviews/evidence", {"run_id": item["run_id"], "item": item["item"]})
            for mission in ("opssat", "webb"):
                for route, record in list(index["files"].items()):
                    if route.startswith(f"/api/missions/{mission}"):
                        del index["files"][route]
            report = {"source_files": provenance, "visits": timing,
                      "models": {m: {k: v for k, v in models.active(m).items() if k in ("version", "metrics", "stale")} for m in ("opssat", "webb")},
                      "evaluation_runs": [r["id"] for r in new_runs]}
            (ROOT / "docs/mission-validation.json").write_text(json.dumps(sanitize(report), indent=2), encoding="utf-8")
            # Publish real model weights and the corrected full-resolution store
            # separately from immutable original release/starter archives.
            import shutil
            publication = ROOT / "release/mission-results"
            publication.mkdir(parents=True, exist_ok=True)
            for mission in ("opssat", "webb"):
                active = models.active(mission)
                destination = publication / "models" / mission / active["version"]
                shutil.copytree(active["path"], destination, dirs_exist_ok=True)
            corrected = publication / "data/webb"
            shutil.copytree(store.mdir("webb"), corrected, dirs_exist_ok=True)
            records = []
            for file in sorted(publication.rglob("*")):
                if file.is_file() and file != publication / "manifest.json":
                    records.append({"path": file.relative_to(publication).as_posix(),
                                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size})
            manifest = {"schema": 1, "models": {m: models.active(m)["version"] for m in ("opssat", "webb")},
                        "files": records, "source_files": provenance, "visits": timing}
            (publication / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
            index["mission_publication_sha256"] = hashlib.sha256((publication / "manifest.json").read_bytes()).hexdigest()
        (CAPSULE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
