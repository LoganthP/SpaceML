"""Multi-mission telemetry: OPS-SAT on the real shipped release; other sources on documented-schema fixtures."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests import missions_fixtures as fx

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    return tmp_path


def _install_opssat(tmp: Path):
    import tarfile
    with tarfile.open(REPO / "starter" / "data" / "missions_opssat.tar.gz") as tf:
        tf.extractall(tmp / "data", filter="data")


def test_opssat_real_release_quick_train_and_review(env):
    _install_opssat(env)
    store = importlib.import_module("spaceml.missions.store")
    opssat = importlib.import_module("spaceml.missions.opssat")
    models = importlib.import_module("spaceml.missions.models")
    man = store.manifest("opssat")
    assert man["source"]["version"] == "v1" and man["counts"]["segments"] == 2123
    # the shipped copies are the published files
    assert opssat._md5(store.mdir("opssat") / "source" / "dataset.csv") == "c12fb83863afa4463dec07303a20bb14"
    assert opssat._md5(store.mdir("opssat") / "source" / "segments.csv") == "1f4b6ffd0359efac96a6698ff557e0f9"
    q = store.query("opssat", "13", max_points=50)
    assert q["n_total"] == 230 and q["n_returned"] <= 50 and q["time_is_epoch"]
    stg = models.Staging("opssat", True)
    with stg as d:
        m = opssat.train(d, quick=True)
    assert m["split_overlap"] == 0 and m["test_segments"] == 529 and m["train_segments"] == 1594
    assert set(m["excluded_from_inputs"]) == {"segment", "anomaly", "train", "channel"}
    assert not set(m["inputs"]) & set(m["excluded_from_inputs"])
    cm = m["model"]["confusion_matrix"]["matrix"]
    assert sum(map(sum, cm)) == 529 and m["model"]["f1_anomaly"] > m["baseline"]["f1_anomaly"]
    runs = importlib.import_module("spaceml.missions.runs")
    res, summ = runs.handle({"mission": "opssat", "action": "evaluate"}, stg.final, env)
    assert len(res["review_items"]) == 529 and res["review_items"][0]["outcome"] in ("FN", "FP")
    it = res["review_items"][0]
    assert {"label", "pred", "prob", "target"} <= set(it)


def test_opssat_refuses_unknown_data(env, tmp_path):
    opssat = importlib.import_module("spaceml.missions.opssat")
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "dataset.csv").write_text("segment,anomaly\n1,0\n")
    (tmp_path / "x" / "segments.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="match no published"):
        opssat.validate_sources(tmp_path / "x")


def test_failed_training_keeps_previous_model(env):
    models = importlib.import_module("spaceml.missions.models")
    stg = models.Staging("opssat", True)
    with stg as d:
        (d / "manifest.json").write_text(json.dumps({"run_kind": "quick"}))
    with pytest.raises(RuntimeError):
        with models.Staging("opssat", False) as d:
            (d / "manifest.json").write_text("{}")
            raise RuntimeError("cancelled")
    assert models.active("opssat")["version"] == stg.version
    assert not list((models.ROOT / "opssat").glob(".staging-*"))


@pytest.mark.parametrize("mission", ["hubble", "webb"])
def test_pointing_import_train_score(env, tmp_path, mission):
    src = fx.hubble(tmp_path / "src") if mission == "hubble" else fx.webb(tmp_path / "src")
    pointing = importlib.import_module("spaceml.missions.pointing")
    store = importlib.import_module("spaceml.missions.store")
    models = importlib.import_module("spaceml.missions.models")
    man = pointing.prepare(mission, src)
    its = store.items(mission)
    if mission == "hubble":
        assert len(its) == 6 and all(i["jif_matched"] for i in its)
        bad = store.item("hubble", "IEPP0104Q")
        assert bad["guiding"]["GSFAIL"] == "NOLOCK" and any("missing" in w for w in bad["warnings"])
        assert store.item("hubble", "IEPP0100Q")["missing"]["SI_V2_AVG"] == 1     # -32767 -> missing
        assert man["time_is_epoch"] is False
    else:
        assert len(its) == 5 and its[0]["segments"] == 2 and its[0]["guiding_modes"] == ["fg"]
        assert abs(its[0]["cadence_s"] - 0.064) < 1e-6 and man["time_is_epoch"]
        assert store.manifest("webb")["channels"][0]["unit"] in ("mas", "0=moving 1=finished 2=offline")
    tr, te = pointing.split_items(mission)
    assert not set(tr) & set(te) and tr and te
    stg = models.Staging(mission, True)
    with stg as d:
        m = pointing.train(mission, d, quick=True)
    assert m["test_items"] == te and "no fault-classification" in m["labels"]
    runs = importlib.import_module("spaceml.missions.runs")
    target = "IEPP0101Q" if mission == "hubble" else "jw01234003001"
    res, _ = runs.handle({"mission": mission, "action": "score", "item": target}, stg.final, env)
    assert res["model_intervals"] or res["baseline_intervals"]
    assert res["review_items"] and res["review_items"][0]["kind"] == "mission_interval"


def test_webb_underscore_column_names(env, tmp_path):
    pointing = importlib.import_module("spaceml.missions.pointing")
    pointing.prepare("webb", fx.webb(tmp_path / "w2", spaced_names=False))
    store = importlib.import_module("spaceml.missions.store")
    assert "delta_ddc_ra" in store.load("webb", "jw01234001001")["cols"]


def test_api_catalog_and_untrainable_missions(env):
    _install_opssat(env)
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        ms = {m["id"]: m for m in c.get("/api/missions").json()["missions"]}
        assert ms["opssat"]["dataset"]["installed"] and not ms["hubble"]["dataset"]["installed"]
        assert ms["hubble"]["dataset"]["state"] == "importer available - no local data"
        assert ms["opssat"]["trainable"] and set(ms) == {"opssat", "hubble", "webb"}
        cat = {e["id"]: e for e in c.get("/api/catalog").json()["datasets"]}
        assert cat["mission-opssat"]["dataset"]["ready"] and cat["mission-opssat"]["train"]
        assert not any(k.startswith(("mission-esa", "mission-funcube", "mission-satnogs")) for k in cat)
        r = c.post("/api/runs", json={"module": "missions", "input": {"mission": "webb", "action": "score", "item": "x"}})
        assert r.status_code == 409 and "prepare" in r.json()["detail"]
        assert c.post("/api/jobs", json={"task": "missions.satnogs.install"}).status_code in (404, 422)
