"""3.1.5: labeled-window datasets (SWAN-SF flare JSON Lines), review evidence, data versions,
automatic mission sources (MAST, mocked) and startup auto-install planning."""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

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


PARAMS = ["TOTUSJH", "R_VALUE", "USFLUX", "MEANSHR"]


def flare_zip(path: Path, n: int = 120, seed: int = 0, name: str = "train_partition1_data.json") -> Path:
    """SWAN-SF layout: JSON Lines, {"<id>": {"label": .., "values": {param: {"0": v, ...}}}}; R_VALUE separates flares."""
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n):
        lab = "M" if i % 10 == 0 else "C" if i % 10 == 1 else "B" if i % 10 == 2 else "Q"
        flare = lab in ("M", "C")
        vals = {}
        for p in PARAMS:
            base = (4.0 if flare else 0.5) if p == "R_VALUE" else rng.normal(10, 2)
            vals[p] = {str(t): float(base + rng.normal(0, 0.3)) for t in range(60)}
        vals["MEANSHR"]["7"] = None                     # a missing reading
        lines.append(json.dumps({str(i + 1): {"label": lab, "values": vals}}))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, "\n".join(lines) + "\n")
    path.write_bytes(buf.getvalue())
    return path


# ------------------------------------------------------------------ workbench
def test_json_lines_table_no_longer_fails(env, tmp_path):
    ingest = importlib.import_module("spaceml.exoplanet.ingest")
    p = tmp_path / "rows.json"
    p.write_text('{"t": 1, "v": 2.5}\n{"t": 2, "v": 2.7}\n')
    df = ingest._read_table(p)
    assert list(df.columns) == ["t", "v"] and len(df) == 2


def test_flare_upload_classify_and_review(env, tmp_path):
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    z = flare_zip(tmp_path / "flare.zip")
    with TestClient(main.app) as c:
        up = c.post("/api/workbench/datasets", files={"file": ("bigdata2020-flare-prediction.zip", z.read_bytes(), "application/zip")},
                    data={"kind": "auto"})
        assert up.status_code == 200, up.text
        ds = up.json()
        assert ds["kind"] == "samples" and ds["samples"]["n"] == 120 and ds["samples"]["flare"]
        assert ds["samples"]["class_counts"] == {"Q": 84, "B": 12, "C": 12, "M": 12}
        assert ds["samples"]["separation"][0]["param"] == "R_VALUE"
        full = c.get(f"/api/workbench/datasets/{ds['id']}").json()
        assert full["compatible"][0] == "samples.classify" and full["profile"]["recommended"] == "samples.classify"
        prof = c.get(f"/api/workbench/datasets/{ds['id']}/class-profile", params={"param": "R_VALUE"}).json()
        assert {x["class"] for x in prof["classes"]} == {"Q", "B", "C", "M"} and len(prof["classes"][0]["median"]) == 60
        r = c.post("/api/runs", json={"module": "workbench", "input": {"dataset_id": ds["id"], "analysis": "samples.classify",
                                                                         "params": {"target": "ge_M", "trees": 80}}}).json()
        t0 = time.time()
        while time.time() - t0 < 120 and (run := c.get(f"/api/runs/{r['id']}").json())["status"] in ("queued", "running"):
            time.sleep(0.3)
        assert run["status"] == "succeeded", run.get("error")
        res = run["result"]
        assert res["viewer"] == "classification" and res["n_positive"] == 12
        assert res["model"]["tss"] > 0.8 and res["baseline"]["tss"] > 0.8          # R_VALUE separates perfectly
        assert "TSS" in run["summary"]["headline"]
        q = c.get("/api/reviews/queue", params={"module": "workbench"}).json()
        samples = [i for i in q["items"] if i["kind"] == "labeled_sample"]
        assert samples, "hits / misses / false alarms go to Review Studio"
        ev = c.get("/api/reviews/evidence", params={"run_id": r["id"], "item": samples[0]["item"]}).json()
        assert ev["type"] == "series" and ev["panels"] and ev["facts"]["label"] in ("M", "C", "B", "Q")
        put = c.put(f"/api/runs/{r['id']}/reviews/{samples[0]['item']}", json={"verdict": "confirmed", "note": "flare region"})
        assert put.status_code == 200
        rows = c.get("/api/reviews", params={"module": "workbench"}).json()["reviews"]
        assert rows[0]["kind"] == "labeled_sample" and rows[0]["sample_id"]


def test_too_few_positives_is_a_clear_error(env, tmp_path):
    store = importlib.import_module("spaceml.workbench.store")
    mvts = importlib.import_module("spaceml.workbench.mvts")
    meta = store.create(flare_zip(tmp_path / "f.zip", n=20), "f.zip")
    with pytest.raises(store.WorkbenchError, match="need at least 3"):
        mvts.run(meta["id"], {"target": "ge_M"})


def test_two_partitions_are_tested_file_by_file(env, tmp_path):
    store = importlib.import_module("spaceml.workbench.store")
    mvts = importlib.import_module("spaceml.workbench.mvts")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zo:
        for k in (1, 2):
            with zipfile.ZipFile(flare_zip(tmp_path / f"p{k}.zip", n=60, seed=k)) as zi:
                zo.writestr(f"train_partition{k}_data.json", zi.read("train_partition1_data.json"))
    (tmp_path / "two.zip").write_bytes(buf.getvalue())
    meta = store.create(tmp_path / "two.zip", "two.zip")
    res = mvts.run(meta["id"], {"target": "ge_C", "trees": 60})
    assert res["split"].startswith("leave-one-file-out") and {s["file"] for s in res["samples"]} == {
        "train_partition1_data.json", "train_partition2_data.json"}


# ------------------------------------------------------------------ review queue de-duplication
def test_queue_shows_repeated_flags_once(env, monkeypatch):
    rv = importlib.import_module("app.routers.reviews")
    runs = [{"id": f"r{i}", "module": "missions", "status": "succeeded", "input_label": "Webb", "created_at": f"2026-11-2{i}", "origin": None}
            for i in range(3)]
    monkeypatch.setattr(rv.worker, "list_runs", lambda m, n: runs)
    monkeypatch.setattr(rv, "read_json", lambda p: {"review_items": [{"item": "iv:0", "kind": "mission_interval", "title": "jw1 · unusual interval",
                                                                      "detail": "11 samples"}]} if str(p).endswith("result.json") else {})
    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): return []
    monkeypatch.setattr(rv, "tx", lambda: _C())
    q = rv.review_queue(None, 60)
    assert q["pending"] == 1 and q["items"][0]["repeats"] == 2 and q["by_module"] == {"missions": 1}


# ------------------------------------------------------------------ data versions
def _publish(store, mission: str, fp_seed: int, start: str):
    w = store.Writer(mission)
    t = np.arange(50, dtype=float) + fp_seed * 1000
    w.add({"id": f"item{fp_seed}", "label": "x"}, t, {"v": np.ones(50) * fp_seed})
    return w.publish({"scope": f"{fp_seed} items", "date_range": [start, start], "source": {"version": f"v{fp_seed}"}})


def test_new_data_keeps_the_old_version_and_models_follow(env):
    store = importlib.import_module("spaceml.missions.store")
    models = importlib.import_module("spaceml.missions.models")
    m1 = _publish(store, "webb", 1, "2022-06-11")
    vdir = models.ROOT / "webb" / "v1-quick"
    vdir.mkdir(parents=True)
    (vdir / "manifest.json").write_text(json.dumps({"dataset_fingerprint": m1["fingerprint"], "run_kind": "quick"}))
    assert models.active("webb")["stale"] is False
    m2 = _publish(store, "webb", 2, "2025-10-01")
    vs = store.versions("webb")
    assert [v["active"] for v in vs] == [True, False] and "2025-10-01" in vs[0]["label"] and "2022-06-11" in vs[1]["label"]
    assert models.active("webb")["stale"] is True                  # trained on the older data
    store.activate("webb", vs[1]["id"])
    assert store.manifest("webb")["fingerprint"] == m1["fingerprint"]
    assert models.active("webb")["stale"] is False
    assert {v["fingerprint"] for v in store.versions("webb")} == {m1["fingerprint"], m2["fingerprint"]}


def test_version_api_refuses_while_a_job_runs(env, monkeypatch):
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    store = importlib.import_module("spaceml.missions.store")
    jobs = importlib.import_module("app.jobs")
    _publish(store, "opssat", 1, "2022-01-04")
    _publish(store, "opssat", 2, "2022-01-05")
    old = store.versions("opssat")[1]["id"]
    with TestClient(main.app) as c:
        monkeypatch.setattr(jobs, "list_jobs", lambda n=30: [{"status": "running", "task": "missions.opssat.train"}])
        assert c.post(f"/api/missions/opssat/versions/{old}/activate").status_code == 409
        monkeypatch.setattr(jobs, "list_jobs", lambda n=30: [])
        r = c.post(f"/api/missions/opssat/versions/{old}/activate")
        assert r.status_code == 200 and r.json()["active"]["id"] == old
        st = c.get("/api/missions/opssat").json()
        assert len(st["versions"]) == 2 and st["auto_install"]["automatic"] is True


# ------------------------------------------------------------------ automatic sources (network mocked)
def test_webb_latest_picks_newest_public_visit_per_visit(env, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    now = inst._mjd_now()
    rows = [{"filename": "jw01111001001_gs-fg_2025300101010_cal.fits", "expstart": now - 5},
            {"filename": "jw01111001001_gs-fg_2025300111010_cal.fits", "expstart": now - 5.1},     # same visit
            {"filename": "jw02222001001_gs-fg_2025200101010_cal.fits", "expstart": now - 60},
            {"filename": "jw03333001001_gs-fg_2025100101010_cal.fits", "expstart": now - 90},
            {"filename": "jw03333001001_gs-acq1_2025100101010_cal.fits", "expstart": now - 90}]   # not fine guide
    monkeypatch.setattr(inst, "_mast", lambda req: rows)
    got = []
    def fake_fetch(url, out, **k):
        if "jw01111" in url:
            raise ValueError("x: HTTP 401 (access denied)")                                       # exclusive access
        got.append(out.name)
        out.write_bytes(b"x")
        return out
    monkeypatch.setattr(inst, "_fetch", fake_fetch)
    inst._webb_latest(tmp_path, 2)
    assert got == ["jw02222001001_gs-fg_2025200101010_cal.fits", "jw03333001001_gs-fg_2025100101010_cal.fits"]


def test_hubble_latest_downloads_jit_jif_pairs(env, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    def mast(req):
        if req["service"] == "Mast.Caom.Filtered":
            return [{"obsid": "1", "t_min": 60900}, {"obsid": "2", "t_min": 60800}]
        return [{"productFilename": "ab1_jit.fits", "productSubGroupDescription": "JIT", "dataRights": "PUBLIC", "dataURI": "mast:HST/product/ab1_jit.fits"},
                {"productFilename": "ab1_jif.fits", "productSubGroupDescription": "JIF", "dataRights": "PUBLIC", "dataURI": "mast:HST/product/ab1_jif.fits"},
                {"productFilename": "ab2_jit.fits", "productSubGroupDescription": "JIT", "dataRights": "PUBLIC", "dataURI": "mast:HST/product/ab2_jit.fits"},
                {"productFilename": "ab2_flt.fits", "productSubGroupDescription": "FLT", "dataRights": "PUBLIC", "dataURI": "mast:HST/product/ab2_flt.fits"}]
    monkeypatch.setattr(inst, "_mast", mast)
    got = []
    monkeypatch.setattr(inst, "_fetch", lambda url, out, **k: (got.append(out.name), out.write_bytes(b"x"))[1] or out)
    inst._hubble_latest(tmp_path, 1)
    assert sorted(got) == ["ab1_jif.fits", "ab1_jit.fits"]                 # ab2 has no jif -> skipped


def test_install_without_files_uses_the_automatic_source(env, monkeypatch, tmp_path):
    inst = importlib.import_module("spaceml.missions.install")
    called = {}
    monkeypatch.setattr(inst, "_webb_latest", lambda d, n: called.setdefault("webb", n))
    monkeypatch.setattr(inst, "_hubble_latest", lambda d, n: called.setdefault("hubble", n))
    inst._download("webb", tmp_path, [])
    inst._download("hubble", tmp_path, [])
    assert called == {"webb": 10, "hubble": 12}


def test_autoinstall_plan_skips_installed_bundled_and_large(env, monkeypatch):
    ai = importlib.import_module("app.autoinstall")
    rel = importlib.import_module("spaceml.release")
    monkeypatch.setattr(rel, "load_manifest", lambda *a, **k: {"datasets": [{"id": "mission-webb"}]})
    plan = ai.plan()
    assert "webb" not in plan and "hubble" in plan and "opssat" in plan
    assert set(plan) <= {"opssat", "hubble"}
    monkeypatch.setenv("SMC_AUTO_INSTALL", "off")
    assert ai.run() == []
