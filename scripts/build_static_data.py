"""Offline static-data compiler.

--refresh uses the optional local training environment to run the unchanged API
against verified starter/release assets in a temporary directory. Normal Vercel
builds use only the standard library to verify and export that frozen capsule.
No network is allowed during either build path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "backend" / "static_data"
PUBLIC = ROOT / "frontend" / "public" / "data"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    manifest = json.loads((ROOT / "DATASETS-MANIFEST.json").read_text(encoding="utf-8"))
    records = list(manifest["starter_pack"]["files"])
    for dataset in manifest["release_bundle"]["datasets"]:
        records.extend({**f, "path": f"release/bundle/datasets/{dataset['id']}/{f['path']}"} for f in dataset["files"])
    records += [{"path": name, "sha256": digest(ROOT / name)} for name in
                ("starter/pack.json", "release/bundle/manifest.json")]
    result = {}
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise RuntimeError(f"Dataset checksum failed: {record['path']}")
        result[record["path"]] = record["sha256"]
    return result


def refresh() -> None:
    import socket
    import urllib.error
    import urllib.request

    # TestClient uses an in-process ASGI transport. All outbound connections are
    # forbidden, including downloads hidden inside an inference helper.
    def blocked(*args, **kwargs):
        raise urllib.error.URLError("Network is disabled by the static-data compiler")
    urllib.request.urlopen = blocked
    socket.create_connection = blocked
    connect = socket.socket.connect
    def no_public_connections(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise OSError("Network is disabled by the static-data compiler")
        return connect(sock, address)
    socket.socket.connect = no_public_connections
    scratch = tempfile.TemporaryDirectory(prefix="smc-static-")
    with scratch as tmp:
        base = Path(tmp)
        os.environ.update(SMC_MODE="local", SPACEML_DATA_DIR=str(base / "data"),
                          SPACEML_MODELS_DIR=str(base / "models"), SPACEML_STORAGE_DIR=str(base / "storage"),
                          SPACEML_STARTER_DIR=str(ROOT / "starter"), SPACEML_BOOTSTRAP="0",
                          SPACEML_AUTOSETUP="off", SMC_AUTO_INSTALL="off")
        sys.path.insert(0, str(ROOT / "backend"))
        import torch
        torch.set_num_threads(2)
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpx2").setLevel(logging.WARNING)
        from fastapi.testclient import TestClient
        from app.local_main import app
        from app import registry, starter, worker
        from spaceml import release
        from spaceml.telemetry import data as telemetry
        from spaceml.missions import catalog, store
        from spaceml.common.utils import sanitize
        responses, files, unavailable = {}, {}, []
        CAPSULE.mkdir(parents=True, exist_ok=True)
        (CAPSULE / "responses").mkdir(exist_ok=True)
        (PUBLIC / "assets").mkdir(parents=True, exist_ok=True)

        def capture(client, path, params=None, required=False):
            response = client.get(path, params=params or {})
            if response.status_code >= 400:
                unavailable.append({"path": path, "params": params or {}, "status": response.status_code,
                                    "detail": response.text[:400]})
                if required:
                    raise RuntimeError(f"Cannot freeze {path}: {response.text}")
                return None
            key = path + ("?" + urlencode(sorted(params.items())) if params else "")
            content_type = response.headers.get("content-type", "application/octet-stream")
            suffix = ".json" if "json" in content_type else ".bin"
            payload = response.content
            if suffix == ".json":
                body = response.json()
                # Temporary build paths must never leak into the published API.
                payload = json.dumps(sanitize(body), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                payload = payload.replace(str(base).encode(), b"[offline-build]")
            name = hashlib.sha256(payload).hexdigest() + suffix
            if suffix == ".json":
                (CAPSULE / "responses" / name).write_bytes(payload)
                responses[key] = {"file": name, "type": content_type}
                return response.json()
            (PUBLIC / "assets" / name).write_bytes(payload)
            files[key] = {"url": f"/data/assets/{name}", "type": content_type}
            return None

        def compute(module, inp, label):
            row = worker.submit(module, inp, label)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                row = worker.get(row["id"])
                if row["status"] not in ("queued", "running"):
                    if row["status"] != "succeeded":
                        raise RuntimeError(f"Static {module} inference failed: {row.get('error')}")
                    return row
                time.sleep(0.05)
            raise RuntimeError(f"Static inference timed out: {label}")

        with TestClient(app) as client:
            installed = starter.install()
            if installed.get("errors"):
                raise RuntimeError(installed)
            verified = release.verify(ROOT / "release" / "bundle")
            if verified["problems"]:
                raise RuntimeError(verified)
            release.install(ROOT / "release" / "bundle")
            publication = ROOT / "release/mission-results"
            if (publication / "manifest.json").is_file():
                from spaceml.missions.published import install as install_publication
                install_publication(publication)
            registry.scan()
            from spaceml.missions import models as mission_models
            for mid in catalog.ORDER:
                if store.installed(mid) and mission_models.active(mid):
                    compute("missions", {"mission": mid, "action": "evaluate"}, f"{mid} validated evaluation")
            for chan in sorted(telemetry.load_channels()):
                if telemetry.has_channel(chan) and registry.telemetry_dir_for(chan):
                    compute("telemetry", {"chan_id": chan}, f"{chan} · frozen NASA telemetry")
                    capture(client, "/api/telemetry/preview", {"chan_id": chan})
            print("Computed telemetry results from the bundled trained models", flush=True)

            # Capture every parameter-free GET using its original response shape.
            for path, methods in app.openapi()["paths"].items():
                get = methods.get("get")
                if get and "{" not in path and not path.startswith("/api/live/"):
                    if any(p.get("required") for p in get.get("parameters", [])):
                        continue
                    capture(client, path)
            capture(client, "/api/runs", {"limit": 500}, required=True)
            from spaceml.missions.availability import SOURCES as unavailable_missions
            for mid in [*catalog.ORDER, *unavailable_missions]:
                capture(client, f"/api/missions/{mid}")
                capture(client, f"/api/missions/{mid}/items", {"limit": 5000})
                capture(client, f"/api/missions/{mid}/predictions")
                if store.installed(mid):
                    for item in store.items(mid):
                        capture(client, f"/api/missions/{mid}/series", {"item": item["id"]})
            for run in worker.list_runs(None, 500):
                rid = run["id"]
                for tail in ("", "/manifest", "/reviews", "/reproducibility", "/evidence.zip"):
                    capture(client, f"/api/runs/{rid}{tail}")
                for fmt in ("md", "json"):
                    capture(client, f"/api/runs/{rid}/report", {"format": fmt})
            from spaceml.mars import data as mars
            for sample in mars.list_samples():
                for kind in ("image", "label", "thumb", "ignore"):
                    if sample.get(kind):
                        capture(client, f"/api/mars/samples/{sample['id']}/{kind}")
            from spaceml.exoplanet import data as exo
            for kepid, plnt in sorted(exo.cached_view_ids()):
                capture(client, f"/api/exoplanet/views/{kepid}/{plnt}")
                compute("exoplanet", {"kepid": kepid, "tce_plnt_num": plnt}, f"KIC {kepid} · TCE {plnt}")
            # Re-capture summaries after computing any optional cached samples.
            for path in ("/api/runs", "/api/overview", "/api/reviews/queue"):
                capture(client, path, {"limit": 500} if path == "/api/runs" else None)
            queue = capture(client, "/api/reviews/queue", {"limit": 1000}) or {"items": []}
            for item in queue["items"]:
                capture(client, "/api/reviews/evidence", {"run_id": item["run_id"], "item": item["item"]})
        index = {"schema": 1, "version": "3.1.10", "sources": sources(),
                 "responses": responses, "files": files, "unavailable": unavailable}
        if (ROOT / "release/mission-results/manifest.json").is_file():
            index["mission_publication_sha256"] = digest(ROOT / "release/mission-results/manifest.json")
        (CAPSULE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"Frozen {len(responses)} JSON responses and {len(files)} static downloads", flush=True)


def export() -> None:
    current_sources = sources()
    index = json.loads((CAPSULE / "index.json").read_text(encoding="utf-8"))
    if current_sources != index["sources"]:
        raise RuntimeError("Frozen data is stale. Run build_static_data.py --refresh in the optional training environment")
    if index.get("mission_publication_sha256"):
        publication = ROOT / "release/mission-results"
        manifest = publication / "manifest.json"
        if digest(manifest) != index["mission_publication_sha256"]:
            raise RuntimeError("Published mission manifest changed; regenerate the frozen capsule")
        for record in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            path = (publication / record["path"]).resolve()
            if not path.is_relative_to(publication.resolve()) or not path.is_file() or digest(path) != record["sha256"]:
                raise RuntimeError(f"Published mission checksum failed: {record['path']}")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for record in index["responses"].values():
        source = CAPSULE / "responses" / record["file"]
        if digest(source) != Path(record["file"]).stem:
            raise RuntimeError(f"Frozen response checksum failed: {source.name}")
        destination = PUBLIC / source.name
        if not destination.is_file() or digest(destination) != Path(record["file"]).stem:
            # The two independent service builds can export concurrently.
            # Publish complete files atomically, never truncate a shared asset.
            with tempfile.NamedTemporaryFile(dir=PUBLIC, prefix=".export-", delete=False) as temp:
                temp.write(source.read_bytes())
                temporary = Path(temp.name)
            temporary.replace(destination)
    for record in index["files"].values():
        asset = ROOT / "frontend" / "public" / record["url"].lstrip("/")
        if not asset.is_file() or digest(asset) != asset.stem:
            raise RuntimeError(f"Static download checksum failed: {asset}")
    (PUBLIC / "build.json").write_text(json.dumps({"version": index["version"], "responses": len(index["responses"]),
                                                   "source_files_verified": len(current_sources)}), encoding="utf-8")
    print(f"Verified and exported {len(index['responses'])} frozen responses; no network or ML imports")


def augment() -> None:
    """Compile review evidence from frozen outputs without rerunning inference."""
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT / "backend"))
    from app.routers import reviews
    from spaceml.common.utils import sanitize
    index = json.loads((CAPSULE / "index.json").read_text(encoding="utf-8"))
    extensions = {"text/markdown": ".md", "text/csv": ".csv", "application/zip": ".zip",
                  "image/png": ".png", "image/jpeg": ".jpg"}
    for record in index["files"].values():
        asset = ROOT / "frontend/public" / record["url"].lstrip("/")
        extension = extensions.get(record["type"].split(";")[0])
        if extension and asset.suffix != extension:
            output = asset.with_suffix(extension)
            shutil.copyfile(asset, output)
            record["url"] = "/data/assets/" + output.name

    def read(path, params=None):
        k = path + ("?" + urlencode(sorted(params.items())) if params else "")
        record = index["responses"].get(k)
        return json.loads((CAPSULE / "responses" / record["file"]).read_text(encoding="utf-8")) if record else None

    def put(path, body, params=None):
        payload = json.dumps(sanitize(body), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        name = hashlib.sha256(payload).hexdigest() + ".json"
        (CAPSULE / "responses" / name).write_bytes(payload)
        k = path + ("?" + urlencode(sorted(params.items())) if params else "")
        index["responses"][k] = {"file": name, "type": "application/json"}

    runs = read("/api/runs", {"limit": 500}) or []
    detailed = {r["id"]: read(f"/api/runs/{r['id']}") for r in runs}
    queue, by_module, seen = [], {}, set()
    for run in runs:
        result = (detailed.get(run["id"]) or {}).get("result") or {}
        for item in reviews._queue_items(run, result):
            identity = (run["module"], item.get("kind"), item.get("title"))
            if identity in seen:
                continue
            seen.add(identity)
            row = {**item, "run_id": run["id"], "module": run["module"], "input_label": run["input_label"],
                   "origin": run.get("origin"), "created_at": run["created_at"]}
            queue.append(row)
            by_module[run["module"]] = by_module.get(run["module"], 0) + 1
            with patch.object(reviews.worker, "get", lambda rid: detailed[rid]), patch.object(
                reviews, "read_json", lambda path, *args: (detailed.get(Path(path).parent.name) or {}).get("result", {})
            ):
                evidence = reviews.evidence(run_id=run["id"], item=item["item"])
            put("/api/reviews/evidence", evidence, {"run_id": run["id"], "item": item["item"]})
    body = {"items": queue[:1000], "pending": len(queue), "runs_scanned": len(runs), "by_module": by_module}
    put("/api/reviews/queue", body, {"limit": 1000})
    put("/api/reviews/queue", {**body, "items": queue[:60]})
    put("/api/reviews/export", {"reviews": [], "read_only": True}, {"format": "json"})
    csv_record = index["files"].get("/api/reviews/export")
    if csv_record:
        import io
        import zipfile
        csv_bytes = (ROOT / "frontend/public" / csv_record["url"].lstrip("/")).read_bytes()
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("reviewed_labels.csv", csv_bytes)
        payload = output.getvalue()
        name = hashlib.sha256(payload).hexdigest() + ".zip"
        (PUBLIC / "assets" / name).write_bytes(payload)
        index["files"]["/api/reviews/export?format=zip"] = {"url": "/data/assets/" + name, "type": "application/zip"}
    (CAPSULE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"Compiled evidence for {len(queue)} review items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        sources()
        refresh()
    if args.augment or args.refresh:
        augment()
    export()
