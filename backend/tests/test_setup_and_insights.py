"""Automatic setup, dataset classification, image insights, review queue and Mars model install.

Images here are generated only to exercise code paths; no network is used.
"""
from __future__ import annotations

import importlib
import io
import sys
import time
import zipfile

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _wait(client, run_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] not in ("queued", "running"):
            return r
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def _rover_zip() -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, (cam, sclk) in enumerate([("NLF", "0817769320"), ("NRF", "0817769321"), ("FLF", "0817769600")]):
            a = ndimage.gaussian_filter(rng.random((240, 320)), 1.5) * 240 - 60 + 60
            if k < 2:
                a[:70] = 180
            out = io.BytesIO()
            Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out, "PNG")
            z.writestr(f"{cam}_1699_{sclk}_891ECM_N0790102NCAM00709_01_195J.png", out.getvalue())
    return buf.getvalue()


def test_setup_status_lists_every_step(client):
    st = client.get("/api/setup").json()
    ids = [s["id"] for s in st["steps"]]
    assert {"spaceweather", "telemetry_data", "telemetry_models", "exoplanet", "mars_model", "traverse"} <= set(ids)
    assert st["profile"] == "off"
    assert client.post("/api/setup/run", json={"steps": ["nope"]}).status_code == 422


def test_rover_images_are_classified_and_summarised(client):
    up = client.post("/api/workbench/datasets", files={"file": ("rover.zip", _rover_zip(), "application/zip")}).json()
    ds = client.get(f"/api/workbench/datasets/{up['id']}").json()
    assert ds["profile"]["category"] == "mars_rover_images"
    assert ds["profile"]["facts"]["stereo_pairs"] == 1
    assert ds["compatible"][0] == "image.insights"      # recommended analysis first
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": up["id"], "analysis": "image.insights", "params": {}}}).json()
    r = _wait(client, run["id"])
    assert r["status"] == "succeeded", r.get("error")
    res = r["result"]
    assert res["n_analysed"] == 3 and len(res["insights"]) >= 3
    assert all(im["scene"] in res["scene_labels"] for im in res["images"])
    assert all(abs(sum(im["regions"].values()) - 1) < 1e-6 for im in res["images"])
    assert "regions" in res["summary"] and res["images"][0]["region_map"].endswith("_regions.png")
    for suffix in ("_regions.png", "_texture.png"):
        assert client.get(f"/api/runs/{run['id']}/files/img/{res['images'][0]['id']}{suffix}").status_code == 200
    q = client.get("/api/reviews/queue").json()
    assert any(i["item"].startswith("image:") for i in q["items"])
    assert client.get(f"/api/runs/{run['id']}/files/img/{res['images'][0]['id']}.jpg").status_code == 200


def test_timeseries_results_carry_findings_and_feed_the_review_queue(client):
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 3000)
    v[2000:2040] += 12
    lab = np.zeros(3000, int)
    lab[1995:2045] = 1
    csv = "step,value,anomaly\n" + "\n".join(f"{i},{x:.4f},{l}" for i, (x, l) in enumerate(zip(v, lab)))
    up = client.post("/api/workbench/datasets", files={"file": ("t.csv", csv.encode(), "text/csv")}).json()
    ds = client.get(f"/api/workbench/datasets/{up['id']}").json()
    assert ds["profile"]["recommended"] == "timeseries.anomaly"
    params = ds["tables"][0]["suggested"]
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": up["id"], "analysis": "timeseries.anomaly", "params": params}}).json()
    r = _wait(client, run["id"])
    assert r["status"] == "succeeded", r.get("error")
    assert any(f["title"] == "Against your labels" for f in r["result"]["insights"])
    q = client.get("/api/reviews/queue").json()
    assert q["pending"] >= 1
    it = q["items"][0]
    client.put(f"/api/runs/{it['run_id']}/reviews/{it['item']}", json={"verdict": "confirmed", "note": ""})
    assert client.get("/api/reviews/queue").json()["pending"] == q["pending"] - 1


def test_mars_model_zip_install_and_rejects(client, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from spaceml.mars.model import build_model
    m = build_model(None, "resnet18")
    d = tmp_path / "colab_out" / "v-colab"
    d.mkdir(parents=True)
    torch.save({"state_dict": m.state_dict(), "img_size": 64, "encoder": "resnet18"}, d / "model.pt")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(d / "model.pt", "models/mars/v-colab/model.pt")
    r = client.post("/api/models/mars/upload", files={"file": ("mars_model.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200, r.text
    assert client.get("/api/models").json()["mars"]["active"] == "v-colab"
    bad = client.post("/api/models/mars/upload", files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert bad.status_code == 422


def test_partial_telemetry_retrain_does_not_replace_full_version(client, tmp_path):
    import json
    root = tmp_path / "models" / "telemetry"
    for ver, chans, t in (("v-full", ["A-1", "A-2", "A-3"], "2026-01-01"), ("v-part", ["A-1"], "2026-06-01")):
        (root / ver).mkdir(parents=True)
        for c in chans:
            (root / ver / f"{c}.pt").write_bytes(b"x")
        (root / ver / "manifest.json").write_text(json.dumps({"version": ver, "created_at": t}))
    client.post("/api/admin/rescan")
    assert client.get("/api/models").json()["telemetry"]["active"] == "v-full"


def test_starter_archive_extracts_only_safe_missing_files(tmp_path):
    import tarfile

    from app import starter
    src = tmp_path / "src" / "telemetry"
    src.mkdir(parents=True)
    (src / "a.npy").write_bytes(b"x")
    arc = tmp_path / "t.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(src, arcname="telemetry")
    root = tmp_path / "data"
    assert starter._extract(arc, root) == 1
    assert (root / "telemetry" / "a.npy").read_bytes() == b"x"
    assert starter._extract(arc, root) == 0          # idempotent
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tf:
        tf.add(src / "a.npy", arcname="../evil.npy")
    with pytest.raises(ValueError):
        starter._extract(bad, root)
