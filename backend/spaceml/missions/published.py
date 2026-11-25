"""Explicitly install verified published mission artifacts into persistent Docker volumes.

Existing data is moved to a retained version. Existing models are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid


def install(source: Path):
    from . import store, models
    from ..config import IS_VERCEL
    if IS_VERCEL:
        raise ValueError("Artifact installation is available in Docker/local mode")
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = (source / record["path"]).resolve()
        if not path.is_relative_to(source) or not path.is_file() or path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Published artifact failed verification: {record['path']}")
    new = json.loads((source / "data/webb/manifest.json").read_text(encoding="utf-8"))
    current = store.manifest("webb") or {}
    if new.get("fingerprint") != current.get("fingerprint"):
        staged = store.ROOT / (".published-webb-" + uuid.uuid4().hex)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "data/webb", staged)
        target = store.mdir("webb")
        if target.exists():
            backup = store.VERS / "webb" / ("before-published-" + uuid.uuid4().hex)
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.rename(backup)
        staged.rename(target)
    for mission, version in manifest["models"].items():
        if mission not in ("opssat", "webb") or Path(version).name != version:
            raise ValueError("Invalid published model identifier")
        target = models.ROOT / mission / version
        if target.exists():
            # An existing version must be identical; do not replace user models.
            for record in manifest["files"]:
                prefix = f"models/{mission}/{version}/"
                if record["path"].startswith(prefix):
                    path = target / record["path"][len(prefix):]
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                        raise ValueError("An existing model version differs; preserving it")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(".published-" + uuid.uuid4().hex)
        shutil.copytree(source / "models" / mission / version, staging)
        staging.rename(target)
    print("Verified mission artifacts installed; existing data versions and models retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    install(parser.parse_args().source)
