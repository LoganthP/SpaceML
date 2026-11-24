"""Data & Training status and offline training prerequisites."""
from __future__ import annotations

import json
import sys
import uuid

import pytest

from tests.test_release_and_jobs import _fresh


def _insert(jobs, task, status, created, error="boom", version=None):
    from app.db import tx
    jid = uuid.uuid4().hex[:12]
    with tx() as c:
        c.execute("INSERT INTO jobs (id, task, module, title, params, status, progress, created_at, error, app_version) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (jid, task, jobs.TASKS[task].module, jobs.TASKS[task].title, "{}", status, 0.0, created, error, version))
    return jid


def test_stale_superseded_and_dismissed_failures_are_hidden(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "cat")
    from app import db, jobs
    from app.manifest import APP_VERSION
    from app.routers import catalog
    db.init_db()
    jobs.init()
    # the user's screenshot: failures left in the volume by the previous app version
    _insert(jobs, "mars.train", "failed", "2026-11-20T10:00:00", 'raise KeyError(f"AI4Mars row has no label column")', None)
    _insert(jobs, "telemetry.retrain", "interrupted", "2026-11-20T10:00:00", "Interrupted", "3.0.0")
    assert catalog._jobs_for("mars", ["mars.train"])["last_failure"] is None
    assert catalog._jobs_for("telemetry", ["telemetry.retrain"])["last_failure"] is None

    # a failure from this version is shown ...
    jid = _insert(jobs, "exoplanet.train", "failed", "2026-11-21T10:00:00", "Not enough", APP_VERSION)
    assert catalog._jobs_for("exoplanet", ["exoplanet.train"])["last_failure"]["id"] == jid
    # ... until dismissed
    jobs.dismiss(jid)
    assert catalog._jobs_for("exoplanet", ["exoplanet.train"])["last_failure"] is None
    # an older failure followed by a success is hidden
    _insert(jobs, "spaceweather.train", "failed", "2026-11-21T10:00:00", "x", APP_VERSION)
    _insert(jobs, "spaceweather.train", "succeeded", "2026-11-21T11:00:00", None, APP_VERSION)
    assert catalog._jobs_for("spaceweather", ["spaceweather.train"])["last_failure"] is None


def test_missing_data_uses_explicit_download_actions(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "dl")
    from app import db, jobs
    from app.routers import catalog

    db.init_db()
    jobs.init()
    cases = [
        ("mars.train", catalog._entry_ai4mars, "Mars needs"),
        ("exoplanet.train", catalog._entry_kepler, "Install the Kepler"),
        ("spaceweather.train", catalog._entry_omni, "OMNI2 years"),
    ]
    for task, entry, message in cases:
        with pytest.raises(ValueError, match=message):
            jobs.TASKS[task].build({})
        card = catalog._train_actions(entry())
        assert card["download"]
        assert all(a["blocked"] and not a["downloads_first"] for a in card["train"])

    assert catalog._entry_flood()["download"]["task"] == "workbench.samples"


def test_omni_training_does_not_require_replay_year(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "omni")
    from app import jobs
    from spaceml.spaceweather import data

    monkeypatch.setattr(data, "cached_years", lambda: [2016, 2017])
    commands = jobs.TASKS["spaceweather.train"].build({"train_years": "2016", "test_years": "2017"})
    assert commands == [["spaceml.spaceweather.forecast", "--train-years", "2016", "--test-years", "2017"]]


def test_full_training_uses_local_pairs_offline(tmp_path, monkeypatch):
    """Regression (audit): with a local AI4Mars subset, Full Training must train on it - not try to download."""
    _fresh(monkeypatch, tmp_path, "full")
    from app import db, jobs
    db.init_db()
    jobs.init()
    from spaceml.mars import local
    monkeypatch.setattr(local, "validate", lambda: {"splits": {"train": {"usable": 600}, "test_min3": {"usable": 273}}})
    for mode in ("quick", "full"):
        cmds = jobs.TASKS["mars.train"].build({"mode": mode})
        assert not any(c[0] == "spaceml.mars.local" for c in cmds), cmds
        assert cmds[0][0] == "spaceml.mars.train" and ("--quick" in cmds[0]) == (mode == "quick")
