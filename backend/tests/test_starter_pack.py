"""The shipped starter pack: integrity and a real analysis on its bundled NASA data (offline)."""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

from spaceml.common.fetch import sha256_file

PACK = Path(__file__).resolve().parents[2] / "starter"


def test_shipped_pack_files_match_their_hashes():
    if not (PACK / "pack.json").exists():
        pytest.skip("no starter pack in this checkout")
    p = json.loads((PACK / "pack.json").read_text())
    files = [f for it in p["contents"] for f in it["files"]]
    assert files and all(sha256_file(PACK / f["path"]) == f["sha256"] for f in files)
    for rid in p["reference_runs"]:
        assert (PACK / "runs" / rid / "result.json").exists() and (PACK / "runs" / rid / "run.json").exists()


def test_bundled_telemetry_run_reproduces(tmp_path, monkeypatch):
    """Fresh install from the pack, then repeat a bundled reference run: identical result."""
    if not (PACK / "pack.json").exists():
        pytest.skip("no starter pack in this checkout")
    pytest.importorskip("torch")
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_STARTER_DIR", str(PACK))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        runs = c.get("/api/runs", params={"module": "telemetry"}).json()
        ref = next(r for r in runs if r["origin"] == "bundled")
        assert c.get(f"/api/runs/{ref['id']}").json()["result"]["chan_id"]
        rep = c.post(f"/api/runs/{ref['id']}/repeat").json()
        t0 = time.time()
        while time.time() - t0 < 180:
            r = c.get(f"/api/runs/{rep['id']}", params={"include_result": False}).json()
            if r["status"] not in ("queued", "running"):
                break
            time.sleep(0.3)
        assert r["status"] == "succeeded", r.get("error")
        chk = c.get(f"/api/runs/{rep['id']}/reproducibility").json()
        assert chk["status"] == "identical", chk
