"""API contract tests against an isolated, empty storage directory."""
from __future__ import annotations

import importlib
import os
import sys

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEML_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SPACEML_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPACEML_STORAGE_DIR", str(tmp_path / "storage"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient

    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def test_health_and_catalog(client):
    assert client.get("/api/health").json()["status"] == "ok"
    ds = client.get("/api/datasets").json()
    assert {d["module"] for d in ds} == {"telemetry", "exoplanet", "mars", "spaceweather"}
    assert all(not d["ready"] for d in ds)  # empty data dir
    ov = client.get("/api/overview").json()
    assert len(ov["modules"]) == 4 and ov["recent_runs"] == []
    assert [e["active"] for e in client.get("/api/evaluation").json()] == [None, None, None, None]


def test_run_validation_and_missing_setup(client):
    r = client.post("/api/runs", json={"module": "telemetry", "input": {"chan_id": "../../etc"}})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"module": "telemetry", "input": {"chan_id": "P-1"}})
    assert r.status_code == 409 and "download" in r.json()["detail"]
    r = client.post("/api/runs", json={"module": "mars", "input": {}})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"module": "venus", "input": {}})
    assert r.status_code == 422
    assert client.get("/api/runs/doesnotexist").status_code == 404


def test_upload_rejects_non_images(client):
    r = client.post("/api/mars/uploads", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415


def test_upload_accepts_png(client):
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.random.rand(128, 160) * 255).astype("uint8")).save(buf, format="PNG")
    r = client.post("/api/mars/uploads", files={"file": ("rover.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["width"] == 160
    # no model installed -> clear 409 rather than a queued failure
    r2 = client.post("/api/runs", json={"module": "mars", "input": {"upload_id": r.json()["upload_id"]}})
    assert r2.status_code == 409 and "setup:mars" in r2.json()["detail"]




def test_job_fails_with_clear_reason(client):
    import time

    # training without data now downloads first; evaluating without the dataset fails fast with a clear reason
    r = client.post("/api/jobs", json={"task": "mars.evaluate", "params": {}})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    dup = client.post("/api/jobs", json={"task": "mars.samples", "params": {}})
    assert dup.status_code in (200, 409)
    for _ in range(120):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    assert j["status"] == "failed"
    assert "not in this installation" in j["error"].lower()
    assert any("python -m spaceml.mars.evaluate" in l for l in j["log"])
    assert client.post("/api/jobs", json={"task": "rm -rf", "params": {}}).status_code == 422
    assert client.post("/api/jobs", json={"task": "exoplanet.lightcurves", "params": {"n_stars": 10 ** 9}}).status_code in (409, 422)
