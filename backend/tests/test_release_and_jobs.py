"""Release bundle (build → verify → install offline, tamper detection) and job-runner rules."""
from __future__ import annotations

import importlib
import json
import sys
import time

import pytest


def _fresh(monkeypatch, tmp_path, name):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / name / k.lower()))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]


def test_bundle_build_install_and_tamper(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "src")
    from spaceml.config import DATA_DIR
    (DATA_DIR / "spaceweather").mkdir(parents=True)
    (DATA_DIR / "spaceweather" / "omni2_2024.npz").write_bytes(b"omni")
    (DATA_DIR / "mars" / "traverse").mkdir(parents=True)
    (DATA_DIR / "mars" / "traverse" / "MSL_waypoints.json").write_text(json.dumps({"features": []}))
    release = importlib.import_module("spaceml.release")
    out = tmp_path / "bundle"
    m = release.build(out, from_data=True)
    ids = {d["id"] for d in m["datasets"]}
    assert {"omni2", "traverse"} <= ids and "kepler" not in ids
    assert all(f["sha256"] and f["bytes"] for d in m["datasets"] for f in d["files"])
    assert release.verify(out)["problems"] == []

    _fresh(monkeypatch, tmp_path, "dst")            # a clean machine
    monkeypatch.setenv("SPACEML_BUNDLE_DIR", str(out))
    release = importlib.import_module("spaceml.release")
    st = release.install()
    from spaceml.config import DATA_DIR as D2
    assert st["complete"] and st["copied"] == 2 and (D2 / "spaceweather" / "omni2_2024.npz").read_bytes() == b"omni"
    assert release.install()["copied"] == 0         # idempotent

    (out / "datasets" / "omni2" / "spaceweather" / "omni2_2024.npz").write_bytes(b"evil")
    assert release.verify(out)["problems"]
    (D2 / "spaceweather" / "omni2_2024.npz").unlink()
    st = release.install()
    assert not st["complete"] and not (D2 / "spaceweather" / "omni2_2024.npz").exists()


def test_final_exception_line_is_reported_not_the_raise_line(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "j")
    jobs = importlib.import_module("app.jobs")
    tb = ["Traceback (most recent call last):", '  File "x.py", line 3, in <module>',
          '    raise KeyError(f"AI4Mars row has no label column (columns: {sorted(row)})")',
          "KeyError: \"AI4Mars row has no label column (columns: ['image'])\""]
    assert jobs._final_error(tb).startswith("KeyError: \"AI4Mars row has no label column (columns: ['image'])")


def test_double_start_reuses_and_heavy_jobs_queue(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "q")
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    jobs = importlib.import_module("app.jobs")
    slow = lambda p: [["spaceml.common.sleep_for_tests", "4"]]   # noqa: E731
    monkeypatch.setitem(jobs.TASKS, "test.heavy_a", jobs.Task("telemetry", "Heavy A", slow, {}, heavy=True))
    monkeypatch.setitem(jobs.TASKS, "test.heavy_b", jobs.Task("mars", "Heavy B", slow, {}, heavy=True))
    monkeypatch.setattr(jobs, "HEAVY_TASKS", ["test.heavy_a", "test.heavy_b"])
    with TestClient(main.app) as c:
        a = c.post("/api/jobs", json={"task": "test.heavy_a", "params": {}}).json()
        again = c.post("/api/jobs", json={"task": "test.heavy_a", "params": {}}).json()
        assert again["id"] == a["id"] and again.get("reused")
        time.sleep(1.0)
        b = c.post("/api/jobs", json={"task": "test.heavy_b", "params": {}}).json()
        time.sleep(1.5)
        jb = c.get(f"/api/jobs/{b['id']}").json()
        assert jb["status"] == "queued" and jb["waiting_for"] == "Heavy A"
        t0 = time.time()
        while time.time() - t0 < 30 and c.get(f"/api/jobs/{b['id']}").json()["status"] in ("queued", "running"):
            time.sleep(0.3)
        assert c.get(f"/api/jobs/{a['id']}").json()["status"] == "succeeded"
        assert c.get(f"/api/jobs/{b['id']}").json()["status"] == "succeeded"
