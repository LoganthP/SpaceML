"""Verify the exact bytes of all shipped deployment datasets before provisioning.

Run from any directory: python scripts/verify_deployment_data.py
This uses only the standard library and never downloads or changes data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path) -> list[str]:
    problems = []

    def check(base: Path, record: dict) -> None:
        path = (base / record["path"]).resolve()
        if base.resolve() not in path.parents:
            problems.append(f"unsafe manifest path: {record['path']}")
            return
        if not path.is_file():
            problems.append(f"missing: {path.relative_to(root)}")
            return
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"] or path.stat().st_size != record["bytes"]:
            problems.append(f"checksum/size mismatch: {path.relative_to(root)}")

    starter = root / "starter"
    pack = json.loads((starter / "pack.json").read_text(encoding="utf-8"))
    for item in pack["contents"]:
        for record in item["files"]:
            check(starter, record)
    for run_id in pack.get("reference_runs", []):
        for name in ("run.json", "result.json"):
            if not (starter / "runs" / run_id / name).is_file():
                problems.append(f"missing reference run: {run_id}/{name}")
    bundle = root / "release" / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for dataset in manifest["datasets"]:
        for record in dataset["files"]:
            check(bundle / "datasets" / dataset["id"], record)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    problems = verify(args.root.resolve())
    if problems:
        print("Deployment data verification failed:\n" + "\n".join(problems))
        raise SystemExit(1)
    print("All starter-pack and release-bundle files match their manifests.")


if __name__ == "__main__":
    main()
