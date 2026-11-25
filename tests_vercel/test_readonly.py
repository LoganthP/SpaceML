from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("SMC_MODE", "vercel")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "cold-cache"))
    sys.path.insert(0, str(ROOT / "backend"))
    for name in list(sys.modules):
        if name.startswith(("app", "spaceml")):
            del sys.modules[name]
    main = importlib.import_module("app.main")
    with TestClient(main.app) as client:
        yield client


def test_empty_tmp_cold_start_has_no_database_or_writes(api):
    from spaceml.config import STORAGE_DIR
    health = api.get("/api/health").json()
    assert health["mode"] == "vercel" and health["version"] == "3.1.10"
    assert health["capabilities"]["write"] is False
    assert not STORAGE_DIR.exists()
    assert "app.db" not in sys.modules and "app.jobs" not in sys.modules


def test_demo_catalog_links_only_published_results_and_marks_missing_missions(api):
    catalog = api.get("/api/catalog").json()
    entries = {e["id"]: e for e in catalog["datasets"]}
    for mid in ("hubble", "esa-m1", "esa-m2", "esa-m3", "funcube", "satnogs"):
        entry = entries["mission-" + mid]
        assert entry["status"]["data"] == "missing"
        assert entry["status"]["next"] == {"kind": "none", "label": "Available in Docker"}
    for name in ("kepler", "ai4mars", "omni2"):
        assert entries[name]["status"]["next"]["kind"] == "none"
    for mid in ("opssat", "webb"):
        entry = entries["mission-" + mid]
        assert entry["model"]["ready"] and not entry["status"].get("repair")
        next_step = entry["status"]["next"]
        assert next_step["label"] == "View sample result"
        from urllib.parse import parse_qs
        run_id = parse_qs(urlsplit(next_step["to"]).query)["run"][0]
        run = api.get("/api/runs/" + run_id).json()
        assert run["status"] == "succeeded"
        assert run["input_ref"]["mission"] == mid


def test_all_published_webb_visits_have_real_duration_and_spacing(api):
    body = api.get("/api/missions/webb/items?limit=5000").json()
    assert body["total"] == 10
    for item in body["items"]:
        assert item["t1"] - item["t0"] > 1
        assert 0.03 < item["cadence_s"] < 0.1
        assert item["time_unit"] == "MJD days"
    mission = api.get("/api/missions/webb").json()
    assert mission["repair"] is None and mission["model"]["ready"]


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/jobs"), ("POST", "/api/setup/run"), ("POST", "/api/mars/uploads"),
    ("POST", "/api/mars/traverse/download"), ("PUT", "/api/runs/example/reviews/candidate"),
    ("DELETE", "/api/runs/example"), ("POST", "/api/models/mars/upload"),
    ("POST", "/api/spaceweather/annotations"), ("POST", "/api/admin/rescan"),
])
def test_writes_are_rejected_before_touching_disk(api, method, path):
    response = api.request(method, path, json={})
    assert response.status_code == 409
    assert response.json()["detail"] == "Available in local/Docker mode"


def test_stored_scoring_and_repeating_never_queue_jobs(api):
    result = api.post("/api/runs", json={"module": "telemetry", "input": {"chan_id": "P-1"}})
    assert result.status_code == 200
    row = result.json()
    assert row["execution"] == "precomputed" and row["status"] == "succeeded"
    detailed = api.get(f"/api/runs/{row['id']}")
    assert detailed.status_code == 200 and detailed.json()["result"]["chan_id"] == "P-1"
    assert api.post(f"/api/runs/{row['id']}/repeat").json()["id"] == row["id"]
    assert api.get("/api/jobs").json() == []


def test_every_frozen_route_has_a_bounded_success_response(api):
    index = json.loads((ROOT / "backend/static_data/index.json").read_text(encoding="utf-8"))
    for path in index["responses"]:
        response = api.get(path, follow_redirects=False)
        assert response.status_code in (200, 307), (path, response.text[:200])
        assert len(response.content) < 4_500_000, path
    for path in index["files"]:
        response = api.get(path, follow_redirects=False)
        assert response.status_code == 307, path
        destination = ROOT / "frontend/public" / response.headers["location"].lstrip("/")
        assert destination.is_file(), path


def test_filters_and_pagination_are_real(api):
    channels = api.get("/api/telemetry/channels?mission=MSL").json()
    assert channels and all(c["spacecraft"] == "MSL" for c in channels)
    first = api.get("/api/missions/opssat/items?limit=2&offset=0").json()
    second = api.get("/api/missions/opssat/items?limit=2&offset=2").json()
    assert first["total"] == second["total"] > 4
    assert len(first["items"]) == 2 and first["items"][0]["id"] != second["items"][0]["id"]
    series = api.get("/api/missions/opssat/series?item=1").json()
    start, end = series["t"][20], series["t"][40]
    window = api.get(f"/api/missions/opssat/series?item=1&start={start}&end={end}&max_points=50").json()
    assert window["t"] and all(start <= t <= end for t in window["t"])
    assert all(len(values) == len(window["t"]) for values in window["cols"].values())
    if isinstance(window.get("label"), list):
        assert len(window["label"]) == len(window["t"])


def test_offline_live_feed_is_calm_and_cdn_cached(api, monkeypatch):
    from spaceml.common import stlive
    from spaceml.common.fetch import SourceUnavailable
    def offline(*args, **kwargs):
        raise SourceUnavailable("Space Telescope Live", stlive.SITE, "offline")
    monkeypatch.setattr(stlive, "_get", offline)
    response = api.get("/api/live/webb")
    assert response.status_code == 200 and response.json()["source"] == "cached"
    assert "offline" in response.json()["notice"].lower()
    assert "s-maxage=" in response.headers["cache-control"]
    sky = api.get("/api/live/webb/sky/unknown")
    assert sky.status_code == 200 and sky.headers["content-type"].startswith("image/")


def test_runtime_imports_without_scientific_dependencies():
    program = '''
import importlib.abc,sys,os
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy','pandas','torch','torchvision','sklearn','PIL','scipy','joblib'}:
            raise AssertionError('Unexpected runtime import: '+fullname)
sys.meta_path.insert(0,BlockML())
from api.index import app
from fastapi.testclient import TestClient
with TestClient(app) as c:
    assert c.get('/api/health').status_code==200
    assert c.get('/api/runs').status_code==200
    import spaceml.common.stlive
print('small runtime works')
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                            env={**os.environ, "SMC_MODE": "vercel"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_export_formats_keep_their_content_types(api):
    response = api.get("/api/reviews/export?format=json", follow_redirects=False)
    assert response.status_code == 200 and response.json()["reviews"] == []
    response = api.get("/api/reviews/export?format=zip", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"].endswith(".zip")


def test_report_default_and_json_format(api):
    row = api.get("/api/runs?limit=1").json()[0]
    response = api.get(f"/api/runs/{row['id']}/report", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"].endswith(".md")
    response = api.get(f"/api/runs/{row['id']}/report?format=json")
    assert response.status_code == 200 and isinstance(response.json(), dict)
