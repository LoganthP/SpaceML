"""Package the repository without pruning nested starter/data or starter/models.

The final archive is verified against DATASETS-MANIFEST.json from inside the ZIP.
No source dataset or manifest is changed by this command.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "space-ml-mission-control-3.1.10-vercel.zip"


def excluded(relative: Path) -> bool:
    parts = relative.parts
    if parts[0] in {".git", ".vercel", ".venv", ".venv-tools", "build", "data", "storage", "deployment", "audit-results-sandbox", "audit-results"}:
        return True
    if any(p in {"node_modules", "__pycache__", ".pytest_cache", ".vercel", ".venv", "secrets"} or p.endswith(".egg-info") for p in parts):
        return True
    if len(parts) > 1 and parts[:2] == ("frontend", "dist"):
        return True
    # These JSON copies are regenerated at build time from backend/static_data.
    # Store the frozen results once, keeping the upload and ZIP small.
    if parts[:3] == ("frontend", "public", "data") and len(parts) == 4 and relative.suffix == ".json":
        return True
    if relative.name.startswith(".env") and relative.name != ".env.example":
        return True
    if relative.suffix in {".pyc", ".pyo", ".part", ".log", ".tsbuildinfo", ".whl"}:
        return True
    if len(parts) == 1 and (relative.suffix == ".zip" or relative.name == "vercel-preview.png"
                            or relative.name.endswith(".verification.json")):
        return True
    return False


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP CRC check failed")
        manifest = json.loads(archive.read("DATASETS-MANIFEST.json"))
        records = list(manifest["starter_pack"]["files"])
        for dataset in manifest["release_bundle"]["datasets"]:
            records.extend({**f, "path": f"release/bundle/datasets/{dataset['id']}/{f['path']}"} for f in dataset["files"])
        for record in records:
            content = archive.read(record["path"])
            if hashlib.sha256(content).hexdigest() != record["sha256"] or len(content) != record["bytes"]:
                raise RuntimeError(f"Final ZIP dataset mismatch: {record['path']}")
        pack_bytes = archive.read("starter/pack.json")
        if pack_bytes != (ROOT / "starter/pack.json").read_bytes():
            raise RuntimeError("starter/pack.json was modified while packaging")
        pack = json.loads(pack_bytes)
        for item in pack["contents"]:
            for record in item["files"]:
                content = archive.read("starter/" + record["path"])
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise RuntimeError(f"Final ZIP pack mismatch: {record['path']}")
        frozen = json.loads(archive.read("backend/static_data/index.json"))
        for record in frozen["responses"].values():
            content = archive.read("backend/static_data/responses/" + record["file"])
            if hashlib.sha256(content).hexdigest() != Path(record["file"]).stem:
                raise RuntimeError(f"Final ZIP frozen response mismatch: {record['file']}")
        for record in frozen["files"].values():
            content = archive.read("frontend/public/" + record["url"].lstrip("/"))
            if hashlib.sha256(content).hexdigest() != Path(record["url"]).stem:
                raise RuntimeError(f"Final ZIP static asset mismatch: {record['url']}")
        return {"files": len(archive.infolist()), "dataset_files_verified": len(records),
                "frozen_responses_verified": len(frozen["responses"]), "static_downloads_verified": len(frozen["files"]),
                "starter_pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
                "zip_bytes": path.stat().st_size, "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(ROOT.rglob("*")):
            relative = file.relative_to(ROOT)
            if file.is_file() and not excluded(relative):
                archive.write(file, relative.as_posix())
    result = verify_archive(OUTPUT)
    OUTPUT.with_suffix(".verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
