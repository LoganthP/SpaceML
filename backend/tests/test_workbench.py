"""Workbench: upload -> preview -> analysis -> report/review, through the real API.

Data here is generated only to exercise code paths in a temporary storage folder.
"""
from __future__ import annotations

import importlib
import io
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _wait(client, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("succeeded", "failed"):
            return r
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def _telemetry_csv() -> bytes:
    rng = np.random.default_rng(0)
    n = 3000
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    a = np.sin(np.arange(n) / 50) + rng.normal(0, 0.05, n)
    b = 20 + np.cumsum(rng.normal(0, 0.01, n))
    label = np.zeros(n, int)
    a[2000:2030] += 3
    label[2000:2030] = 1
    df = pd.DataFrame({"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "bus_voltage": a, "temp": b,
                       "anomaly": label, "mode": ["nominal"] * n})
    return df.to_csv(index=False).encode()


def test_timeseries_upload_preview_analysis_report_review(client):
    r = client.post("/api/workbench/datasets", files={"file": ("telemetry.csv", _telemetry_csv(), "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["kind"] == "timeseries"
    tab = ds["tables"][0]
    types = {c["name"]: c["type"] for c in tab["columns"]}
    assert types == {"timestamp": "datetime", "bus_voltage": "numeric", "temp": "numeric",
                     "anomaly": "binary", "mode": "text"}
    sug = tab["suggested"]
    assert sug["time_col"] == "timestamp" and sug["label_col"] == "anomaly" and "bus_voltage" in sug["sensor_cols"]
    det = client.get(f"/api/workbench/datasets/{ds['id']}").json()
    assert det["compatible"] == ["timeseries.anomaly"]
    prev = client.get(f"/api/workbench/datasets/{ds['id']}/series", params={"table": tab["id"]}).json()
    assert "bus_voltage" in prev["series"]

    bad = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "timeseries.anomaly", "params": {"table": tab["id"], "sensor_cols": ["mode"]}}})
    assert bad.status_code == 422 and "text" in bad.json()["detail"]

    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "timeseries.anomaly",
        "params": {"table": tab["id"], "time_col": "timestamp", "sensor_cols": ["bus_voltage"],
                   "label_col": "anomaly", "train_fraction": 0.3}}}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "succeeded", done["error"]
    res = done["result"]
    assert res["viewer"] == "timeline" and res["x_kind"] == "datetime"
    assert res["metrics"]["robust"]["recall"] == 1.0 and res["metrics"]["zscore"]["recall"] == 1.0
    assert res["robust"]["intervals"][0]["x0"].startswith("2024-01-02")

    rv = client.put(f"/api/runs/{run['id']}/reviews/flag:0", json={"verdict": "confirmed", "note": "real spike"})
    assert rv.status_code == 200
    md = client.get(f"/api/runs/{run['id']}/report").text
    assert ds["sha256"] in md and "confirmed" in md and "precision" in md
    js = client.get(f"/api/runs/{run['id']}/report", params={"format": "json"}).json()
    assert js["dataset"]["sha256"] == ds["sha256"] and js["reviews"][0]["verdict"] == "confirmed"


def test_image_pair_change_detection_from_zip(client):
    rng = np.random.default_rng(1)
    before = (rng.random((200, 240)) * 40 + 100).astype(np.uint8)
    after = (before.astype(float) * 1.15 + 5).clip(0, 255).astype(np.uint8)  # global brightness change only
    after[50:110, 60:140] = 250  # real change: 60x80 px
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, arr in (("site_2021.png", before), ("site_2023.png", after)):
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format="PNG")
            z.writestr(f"pair/{name}", b.getvalue())
    ds = client.post("/api/workbench/datasets", files={"file": ("pair.zip", buf.getvalue(), "application/zip")}).json()
    assert ds["kind"] == "images" and len(ds["images"]) == 2
    det = client.get(f"/api/workbench/datasets/{ds['id']}").json()
    assert "image.change" in det["compatible"] and "image.terrain" in det["compatible"]
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "image.change",
        "params": {"before": "i0", "after": "i1", "pixel_size_m": 10}}}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "succeeded", done["error"]
    res = done["result"]
    expected = 60 * 80
    assert abs(res["counts"]["changed"] - expected) / expected < 0.25
    assert abs(res["changed_area_km2"] - expected * 100 / 1e6) < 0.15
    for f in ("before.png", "after.png", "change.png"):
        assert client.get(f"/api/runs/{run['id']}/files/{f}").status_code == 200
    # terrain needs a model -> clear 409 before queueing
    r = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "image.terrain", "params": {"image": "i0"}}})
    assert r.status_code == 409


def test_water_mode_needs_bands_and_lightcurve_routing(client):
    b = io.BytesIO()
    Image.fromarray(np.zeros((80, 80), np.uint8)).save(b, format="PNG")
    ds = client.post("/api/workbench/datasets", files={"file": ("a.zip", _zip({"a.png": b.getvalue(), "b.png": b.getvalue()}),
                                                                  "application/zip")}).json()
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "image.change", "params": {"before": "i0", "after": "i1", "method": "water"}}}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "failed" and "band" in done["error"]

    t = np.linspace(0, 30, 3000)
    f = 1 + np.random.default_rng(2).normal(0, 3e-4, t.size)
    f[((t % 3.2) < 0.1)] -= 0.004
    csv = pd.DataFrame({"time": t, "flux": f}).to_csv(index=False).encode()
    r = client.post("/api/workbench/datasets", files={"file": ("star.csv", csv, "text/csv")}).json()
    assert r["kind"] == "lightcurve" and r["exoplanet_dataset"]["targets"]
    kinds = {d["kind"] for d in client.get("/api/workbench/datasets").json()}
    assert kinds == {"images", "lightcurve"}


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_geotiff_water_change_area_from_crs(client, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    def scene(water_cols):
        a = np.full((4, 100, 100), 0.2, np.float32)  # bands: blue, green, red, nir
        a[3] = 0.4                                  # land: NIR > green -> NDWI < 0
        a[1][:, :water_cols] = 0.3                  # water: green high, NIR low
        a[3][:, :water_cols] = 0.05
        return a

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, cols in (("river_before.tif", 20), ("river_after.tif", 50)):
            p = tmp_path / name
            with rasterio.open(p, "w", driver="GTiff", width=100, height=100, count=4, dtype="float32",
                               crs="EPSG:32633", transform=from_origin(500000, 4600000, 30, 30)) as dst:
                dst.write(scene(cols))
            z.write(p, name)
    ds = client.post("/api/workbench/datasets", files={"file": ("river.zip", buf.getvalue(), "application/zip")}).json()
    assert ds["images"][0]["geo"]["georeferenced"]
    ids = {im["file"]: im["id"] for im in ds["images"]}
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": ds["id"], "analysis": "image.change",
        "params": {"before": ids["river_before.tif"], "after": ids["river_after.tif"], "method": "water",
                   "green_band": 2, "nir_band": 4}}}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "succeeded", done["error"]
    r = done["result"]
    assert r["counts"]["gained"] == 30 * 100 and r["counts"]["lost"] == 0
    assert abs(r["changed_area_km2"] - 3000 * 900 / 1e6) < 1e-6  # 30 m pixels


def test_lightcurve_in_julian_date_runs(client):
    """Regression: BJD/JD time stamps (~2.46e6 days) used to overflow the detrend step."""
    t = np.arange(0, 27, 2 / 1440) + 2459000.0
    f = 1 + np.random.default_rng(3).normal(0, 4e-4, t.size)
    ph = (t - 2459001.3) % 3.853
    f[(ph < 0.06) | (ph > 3.853 - 0.06)] -= 0.006
    csv = pd.DataFrame({"time": t, "flux": f}).to_csv(index=False).encode()
    ds = client.post("/api/workbench/datasets", files={"file": ("bjd.csv", csv, "text/csv")}).json()
    tid = ds["exoplanet_dataset"]["targets"][0]["id"]
    run = client.post("/api/runs", json={"module": "exoplanet", "input": {"dataset_id": ds["id"], "target_id": tid}}).json()
    done = _wait(client, run["id"], timeout=180)
    assert done["status"] == "succeeded", done["error"]
    assert abs(done["result"]["tce"]["period_days"] - 3.853) < 0.01
    assert done["result"]["system"]["planets"]


def test_review_studio_export_and_mask_correction(client, tmp_path):
    """Reviews from any run show up in the studio and export as CSV / ZIP with corrected masks."""
    r = client.post("/api/workbench/datasets", files={"file": ("telemetry.csv", _telemetry_csv(), "text/csv")}).json()
    run = client.post("/api/runs", json={"module": "workbench", "input": {
        "dataset_id": r["id"], "analysis": "timeseries.anomaly",
        "params": {"table": "t0", "time_col": "timestamp", "sensor_cols": ["bus_voltage"], "label_col": "anomaly"}}}).json()
    _wait(client, run["id"])
    client.put(f"/api/runs/{run['id']}/reviews/robust:0", json={"verdict": "confirmed"})
    studio = client.get("/api/reviews").json()
    assert studio["counts"] == {"confirmed": 1} and studio["reviews"][0]["kind"] == "series_interval"
    csv_text = client.get("/api/reviews/export", params={"format": "csv"}).text
    assert "series_interval" in csv_text and "confirmed" in csv_text
    # corrected mask: needs a run with mask.png -> create one by hand in the run folder
    from spaceml.config import RUNS_DIR
    (RUNS_DIR / run["id"] / "mask.png").parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((20, 30), np.uint8)).save(RUNS_DIR / run["id"] / "mask.png")
    Image.fromarray(np.zeros((20, 30), np.uint8)).save(RUNS_DIR / run["id"] / "input.png")
    bad = io.BytesIO(); Image.fromarray(np.full((20, 30), 7, np.uint8)).save(bad, format="PNG")
    assert client.put(f"/api/runs/{run['id']}/mask", files={"file": ("m.png", bad.getvalue(), "image/png")}).status_code == 422
    fix = np.zeros((20, 30), np.uint8); fix[:, :15] = 2
    good = io.BytesIO(); Image.fromarray(fix).save(good, format="PNG")
    res = client.put(f"/api/runs/{run['id']}/mask", files={"file": ("m.png", good.getvalue(), "image/png")}).json()
    assert abs(res["changed_fraction"] - 0.5) < 1e-9
    z = zipfile.ZipFile(io.BytesIO(client.get("/api/reviews/export", params={"format": "zip"}).content))
    assert {"reviewed_labels.csv", f"terrain/labels/{run['id']}.png", f"terrain/images/{run['id']}.png"} <= set(z.namelist())
