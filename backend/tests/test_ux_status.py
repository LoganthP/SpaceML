"""3.1.4: one status definition everywhere, in-app Install data (upload path), token secrecy,
Webb guider clock units, and keeping locally re-imported mission data over the bundle."""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests import missions_fixtures as fx


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    monkeypatch.setenv("SPACEML_BUNDLE_DIR", str(tmp_path / "no-bundle"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    return tmp_path


# ------------------------------------------------------------------ shared status definition
def _entry(**kw):
    base = {"id": "kepler", "module": "exoplanet", "train": [{"task": "exoplanet.train", "label": "Quick run", "params": {}}],
            "download": {"task": "exoplanet.download", "label": "Download light curves (needs internet)"}}
    base.update(kw)
    return base


def test_kepler_browse_only_is_not_called_ready(env):
    cat = importlib.import_module("app.routers.catalog")
    e = _entry(dataset={"ready": True, "trainable": False, "browse_only_label": "Browse only - more labeled signals needed",
                        "blocked_reason": "training needs at least 30 planet candidates and 30 false positives (have 12 / 46): download 18 more planet candidates"},
               model={"ready": False})
    st = cat._status(e)
    assert st["data"] == "browse_only" and st["data_label"] == "Browse only - more labeled signals needed"
    assert st["next"]["kind"] == "task" and st["next"]["task"] == "exoplanet.download"
    assert "needs needs" not in e["dataset"]["blocked_reason"]
    e["status"] = st
    s = cat._summary([e])
    assert s["installed"] == 1 and s["trainable"] == 0 and s["browse_only"] == 1 and s["models_available"] == 0


def test_status_next_steps(env):
    cat = importlib.import_module("app.routers.catalog")
    missing_mission = {"id": "mission-hubble", "mission": "hubble", "module": "missions", "dataset": {"ready": False}, "model": {"ready": False}}
    assert cat._status(missing_mission)["next"] == {"kind": "link", "label": "Install data", "to": "/telemetry?mission=hubble#install"}
    untrained = {"id": "mission-opssat", "mission": "opssat", "module": "missions", "dataset": {"ready": True}, "model": {"ready": False},
                 "train": [{"task": "missions.opssat.train", "label": "Quick train", "params": {"mode": "quick"}}]}
    n = cat._status(untrained)["next"]
    assert n["kind"] == "task" and n["task"] == "missions.opssat.train" and "next step" in n["label"]
    busy = dict(untrained, active_job={"title": "Train OPS-SAT", "status": "running"})
    assert cat._status(busy)["next"]["kind"] == "wait"
    analysis_only = {"id": "traverse", "dataset": {"ready": True}, "model": None}
    st = cat._status(analysis_only)
    assert st["model"] == "not_applicable" and st["next"]["kind"] == "link"


# ------------------------------------------------------------------ install command validation
def test_install_cmd_validation(env):
    jobs = importlib.import_module("app.jobs")
    with pytest.raises(ValueError):
        jobs._install_cmd("hubble", {"source": "../../etc"})
    with pytest.raises(ValueError):
        jobs._install_cmd("hubble", {"files": ["a b;rm -rf /"]})
    cmd = jobs._install_cmd("webb", {"files": "jw02732001001_gs-fg_2022162160927_cal.fits"})
    assert cmd[:3] == ["spaceml.missions.install", "webb", "--download"] and cmd[-1].endswith("_cal.fits")


# ------------------------------------------------------------------ Webb guider clock (MJD days, not ms)
def test_jw_times_units():
    pointing = importlib.import_module("spaceml.missions.pointing")
    end = 1654963767.0
    mjd = (end - np.arange(4184)[::-1] * 0.064) / 86400 + 40587.0
    t, basis = pointing.jw_times(mjd, None, end)
    assert basis == "MJD days" and abs(t[-1] - end) < 1e-3 and 0.06 < np.median(np.diff(t)) < 0.07
    t, _ = pointing.jw_times(np.arange(100) * 64.0, "ms", end)
    assert abs(np.median(np.diff(t)) - 0.064) < 1e-6
    t, _ = pointing.jw_times(np.arange(100) * 0.064, "s", end)
    assert abs(np.median(np.diff(t)) - 0.064) < 1e-6


def test_webb_upload_install_mjd_clock(env):
    """Full in-app path: upload real-shaped guider files -> Install job -> installed with a sane clock."""
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    src = fx.webb(env / "src", n_visits=3, time_mode="mjd")
    with TestClient(main.app) as c:
        files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in sorted(src.glob("*.fits"))]
        up = c.post("/api/missions/webb/upload", files=files).json()
        assert len(up["files"]) == 6
        j = c.post("/api/jobs", json={"task": "missions.webb.install", "params": {"source": up["folder"]}}).json()
        t0 = time.time()
        while time.time() - t0 < 120 and (s := c.get(f"/api/jobs/{j['id']}").json())["status"] in ("queued", "running"):
            time.sleep(0.5)
        assert s["status"] == "succeeded", s.get("error")
        st = c.get("/api/missions/webb").json()
        assert st["dataset"]["installed"] and st["repair"] is None
        it = c.get("/api/missions/webb/items?limit=1").json()["items"][0]
        assert 0.06 < it["cadence_s"] < 0.07 and it["t1"] - it["t0"] > 100
        cat = c.get("/api/catalog").json()
        w = next(e for e in cat["datasets"] if e["id"] == "mission-webb")
        assert w["status"]["data"] == "installed" and w["status"]["next"]["kind"] == "task"   # Quick train is next
        assert not list((Path(env) / "data" / "missions-source" / "webb").glob("upload-*"))       # upload cleaned up


def test_collapsed_webb_store_offers_repair(env):
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    pointing = importlib.import_module("spaceml.missions.pointing")
    src = fx.webb(env / "src", n_visits=2, time_mode="mjd")
    # reproduce the pre-3.1.4 behaviour: treat the MJD column as milliseconds
    real = pointing.jw_times
    pointing.jw_times = lambda tms, unit, end: (end - (tms[-1] - tms) / 1000.0, "milliseconds (anchored to file stamp)")
    with pytest.raises(ValueError, match="collapse"):
        pointing.prepare("webb", src)          # the new guard refuses to install it
    pointing.jw_times = real
    items = [{"n": 4000, "t0": 1.0, "t1": 1.000003, "files": ["a_cal.fits"]}]
    assert pointing.webb_time_problem(items) == ["a_cal.fits"]
    assert pointing.webb_time_problem([{"n": 4000, "t0": 1.0, "t1": 300.0, "cadence_s": 0.064, "files": ["a"]}]) is None
    assert pointing.webb_time_problem([{"n": 4000, "t0": 1.0, "t1": 3600.0, "cadence_s": 7e-7, "files": ["b"]}]) == ["b"]
    with TestClient(main.app) as c:
        assert c.get("/api/missions/webb").json()["repair"] is None   # nothing installed - nothing to repair


# ------------------------------------------------------------------ uploads
def test_upload_rejects_wrong_files_and_removed_missions(env):
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        bad = c.post("/api/missions/webb/upload", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
        assert bad.status_code in (400, 415, 422)
        for gone in ("funcube", "satnogs", "esa-m1"):
            assert c.post(f"/api/missions/{gone}/upload", files=[("files", ("a.csv", b"t,v\n", "text/csv"))]).status_code == 404
            assert c.get(f"/api/missions/{gone}").status_code == 404


def test_release_keeps_locally_reimported_mission(env, tmp_path):
    rel = importlib.import_module("spaceml.release")
    b = tmp_path / "bundle"
    man_rel = "missions/webb/manifest.json"
    (b / "datasets" / "mission-webb" / "missions" / "webb").mkdir(parents=True)
    (b / "datasets" / "mission-webb" / man_rel).write_text(json.dumps({"fingerprint": "old"}))
    d = {"id": "mission-webb", "files": [{"path": man_rel}]}
    root = rel._root(d)
    (root / "missions" / "webb").mkdir(parents=True, exist_ok=True)
    (root / man_rel).write_text(json.dumps({"fingerprint": "new-local-import"}))
    assert rel._kept_local(b, d) is True
    (root / man_rel).write_text(json.dumps({"fingerprint": "old"}))
    assert rel._kept_local(b, d) is False
