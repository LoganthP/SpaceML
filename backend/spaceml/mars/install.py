"""Install a trained Mars terrain model (e.g. the mars_model.zip produced by the Colab notebook).

    python -m spaceml.mars.install --zip mars_model.zip
    python -m spaceml.mars.install --url https://github.com/<you>/<repo>/releases/download/v1/mars_model.zip

The zip may have any layout; the folder that holds model.pt is installed as
models/mars/<version>/ (version = its manifest's version, else the folder name). The checkpoint
is loaded once before it is published, so a broken or incompatible file never becomes active.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from ..common.progress import fail, step
from ..common.utils import new_version, now_iso, read_json, write_json
from ..config import MODELS_DIR

MAX_ZIP = 2_000_000_000


class InstallError(ValueError):
    pass


def install_zip(zpath: Path, source: str = "upload") -> dict:
    if not zipfile.is_zipfile(zpath):
        raise InstallError("Not a .zip file")
    work = Path(tempfile.mkdtemp(prefix="marsinst_", dir=MODELS_DIR))
    try:
        with zipfile.ZipFile(zpath) as z:
            total = 0
            for m in z.infolist():
                total += m.file_size
                name = Path(m.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise InstallError(f"Unsafe path in zip: {m.filename}")
            if total > MAX_ZIP:
                raise InstallError("Zip is larger than 2 GB once extracted")
            z.extractall(work)
        ckpts = sorted(work.rglob("model.pt"))
        if not ckpts:
            raise InstallError("The zip has no model.pt (expected models/mars/<version>/model.pt)")
        return install_dir(ckpts[0].parent, source)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install_dir(src: Path, source: str) -> dict:
    import torch

    from .model import load
    step("Checking the checkpoint")
    try:
        model, size = load(src)
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        if out.shape[1] != 4:
            raise InstallError(f"Model predicts {out.shape[1]} classes, expected 4 (soil, bedrock, sand, big rock)")
    except InstallError:
        raise
    except Exception as e:  # noqa: BLE001
        raise InstallError(f"model.pt could not be loaded as a DeepLabV3+ terrain model: {e}") from e
    manifest = read_json(src / "manifest.json", {}) or {}
    version = str(manifest.get("version") or src.name or new_version())
    version = "".join(ch for ch in version if ch.isalnum() or ch in "-_.")[:40] or new_version()
    dest = MODELS_DIR / "mars" / version
    if dest.exists():
        version = f"{version}-{new_version()[1:]}"
        dest = MODELS_DIR / "mars" / version
    stage = MODELS_DIR / "mars" / f".staging-{version}"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(src, stage)
    manifest = {"module": "mars", "model_name": "DeepLabV3+ terrain model", "created_at": now_iso(),
                "img_size": size, **manifest, "version": version, "installed_from": source,
                "installed_at": now_iso()}
    write_json(stage / "manifest.json", manifest)
    stage.replace(dest)
    print(f"Installed Mars model {version} into {dest}")
    return {"version": version, "path": str(dest), "has_metrics": (dest / "metrics.json").exists()}


def main() -> None:
    p = argparse.ArgumentParser(description="Install a trained Mars terrain model")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--url")
    a = p.parse_args()
    (MODELS_DIR / "mars").mkdir(parents=True, exist_ok=True)
    try:
        if a.zip:
            r = install_zip(Path(a.zip), source=a.zip)
        else:
            from ..common.fetch import SourceUnavailable, fetch
            tmp = MODELS_DIR / "mars" / ".download.zip"
            step("Downloading the trained Mars model")
            try:
                fetch(a.url, tmp, source="Mars model URL", timeout=120, retries=3, max_bytes=MAX_ZIP)
            except SourceUnavailable as e:
                fail(f"source_unavailable: {e}")
            try:
                r = install_zip(tmp, source=a.url)
            finally:
                tmp.unlink(missing_ok=True)
                tmp.with_name(tmp.name + ".source.json").unlink(missing_ok=True)
    except InstallError as e:
        fail(str(e))
    if not r["has_metrics"]:
        print("No metrics.json in the zip - evaluate it from Data & Training (Mars → Re-evaluate).")


if __name__ == "__main__":
    main()
