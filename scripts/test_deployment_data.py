from __future__ import annotations

import hashlib
import json

from verify_deployment_data import verify


def make_bundle(root):
    starter = root / "starter"
    starter.mkdir()
    payload = b"real artifact bytes\n"
    (starter / "artifact.bin").write_bytes(payload)
    record = {"path": "artifact.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (starter / "pack.json").write_text(json.dumps({"contents": [{"files": [record]}]}))
    bundle = root / "release" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"datasets":[]}')
    return starter


def test_valid_bundle(tmp_path):
    make_bundle(tmp_path)
    assert verify(tmp_path) == []


def test_corrupted_or_missing_files_are_rejected(tmp_path):
    starter = make_bundle(tmp_path)
    (starter / "artifact.bin").write_bytes(b"changed artifact")
    assert "checksum/size mismatch" in verify(tmp_path)[0]
    (starter / "artifact.bin").unlink()
    assert "missing:" in verify(tmp_path)[0]


def test_manifest_cannot_escape_root(tmp_path):
    starter = make_bundle(tmp_path)
    pack = {"contents": [{"files": [{"path": "../outside.bin"}]}]}
    (starter / "pack.json").write_text(json.dumps(pack))
    assert "unsafe manifest path" in verify(tmp_path)[0]
