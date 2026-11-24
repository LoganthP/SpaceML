"""Readiness, ZIP method checks and model availability across dataset versions (3.1.6-3.1.8)."""
from __future__ import annotations

import importlib
import json
import sys
import zipfile

import numpy as np
import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    monkeypatch.setenv("SMC_AUTO_INSTALL", "off")
    monkeypatch.setenv("SPACEML_BUNDLE_DIR", str(tmp_path / "no-bundle"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    return tmp_path


def test_one_readiness_definition(env):
    cat = importlib.import_module("app.routers.catalog")
    st = lambda **kw: cat._status({"id": "x", "dataset": kw.get("ds", {"ready": True}), "model": kw.get("m", {"ready": False}),
                                   "train": [{"task": "t", "label": "Quick train"}]})["readiness"]
    assert st(ds={"ready": False}) == "missing"
    assert st(m=None) == "analysis_only"
    assert st(m={"ready": True}) == "model_available"
    assert st() == "ready_to_train"
    assert st(ds={"ready": True, "trainable": False}) == "installed"
    s = cat._summary([{"status": cat._status({"id": "a", "dataset": {"ready": False}, "model": None})}])
    assert s["readiness"]["missing"] == 1 and set(s["readiness_labels"].values()) == {
        "Missing", "Installed", "Ready to train", "Model available", "Analysis only"}


def test_unsupported_method_is_named(env):
    import io
    zs = importlib.import_module("spaceml.common.zipsupport")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    zf.getinfo("a.txt").compress_type = 93                       # zstandard: not readable here
    with pytest.raises(ValueError, match="zstandard"):
        zs.check_members(zf, ["a.txt"])


def test_model_trained_on_other_version_stays_available(env):
    from fastapi.testclient import TestClient
    store = importlib.import_module("spaceml.missions.store")
    models = importlib.import_module("spaceml.missions.models")
    w = store.Writer("opssat")
    w.add({"id": "s1", "label": "s1"}, np.arange(30.0), {"v": np.ones(30)})
    w.publish({"scope": "x"})
    for name, fp in (("v1-quick", "other"), ):
        d = models.ROOT / "opssat" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"dataset_fingerprint": fp, "run_kind": "quick"}))
    legacy = models.ROOT / "webb" / "v0-quick"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(json.dumps({"run_kind": "quick"}))          # no fingerprint recorded
    w2 = store.Writer("webb"); w2.add({"id": "v", "label": "v"}, np.arange(30.0), {"v": np.ones(30)}); w2.publish({"scope": "y"})
    assert models.active("webb")["stale"] is False
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        st = c.get("/api/missions/opssat").json()["model"]
        assert st["ready"] is True and st["stale"] is True
