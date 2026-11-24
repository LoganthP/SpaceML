"""Release checks that run without network access.

* the OMNI2 parser against real NASA rows (column mapping proven by physics identities)
* Mission Replay: causal issuance, persistence baseline, timeline, run manifest, evidence zip,
  Repeat run -> identical result
* data-quality statuses, job retry / interrupted state, starter pack build + install (bundled runs)
* Traverse Explorer: product-id parsing, waypoint matching, sections, API with the NASA
  responses replaced by small fixtures (structure copied from the real NASA endpoints)
"""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_spaceweather import omni_year

HERE = Path(__file__).parent


def _fresh(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    monkeypatch.setenv("SPACEML_STARTER_DIR", str(tmp_path / "starter"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]


def _wait(c, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{run_id}").json()
        if r["status"] not in ("queued", "running"):
            return r
        time.sleep(0.2)
    raise AssertionError("run did not finish")


# ---------------------------------------------------------------- real OMNI2 format
def test_parser_on_real_nasa_rows():
    from spaceml.spaceweather.data import parse_omni2
    df = parse_omni2((HERE / "data" / "omni2_2000_excerpt.dat").read_text())
    assert len(df) == 25 and str(df.index[0]) == "2000-01-01 00:00:00+00:00"
    first = df.iloc[0]
    assert first["V"] == 675 and first["Bz"] == 1.6 and first["Dst"] == -45 and first["Kp"] == pytest.approx(5.3)
    assert df.iloc[-1].isna().all() or df.iloc[-1]["V"] != df.iloc[-1]["V"]    # last record carries fill values
    d = df.dropna()
    assert len(d) == 24
    assert np.allclose(d["E"], -d["V"] * d["Bz"] / 1000, atol=0.01)          # E = -V Bz
    assert np.allclose(d["Pdyn"], 2e-6 * d["N"] * d["V"] ** 2, rtol=0.01)    # Pdyn = 2e-6 N V^2


def test_parser_rejects_shifted_columns():
    from spaceml.spaceweather.data import FormatError, parse_omni2
    lines = (HERE / "data" / "omni2_2000_excerpt.dat").read_text().splitlines()
    shifted = "\n".join(" ".join(ln.split()[:24] + ln.split()[25:] + ["0"]) for ln in lines)  # drop V column
    with pytest.raises(FormatError):
        parse_omni2(shifted)


def test_quality_statuses():
    from spaceml.common.quality import series_quality
    idx = pd.date_range("2024-05-10", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"V": np.r_[np.full(10, 500.0), np.linspace(400, 800, 38)], "Dst": np.linspace(0, -300, 48)}, index=idx)
    q = series_quality(df, ["V", "Dst"], continuous={"V"})
    assert q["status"] == "degraded" and q["columns"][0]["stale_samples"] == 10       # frozen speed reading
    df.loc[idx[5:40], "Dst"] = np.nan
    q = series_quality(df, ["V", "Dst"], required=["Dst"])
    assert q["status"] == "insufficient_data" and "not enough" in q["findings"][0]


# ---------------------------------------------------------------- Mission Replay + evidence
@pytest.fixture()
def sw_client(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    from spaceml.spaceweather import data, forecast
    for i, y in enumerate((2014, 2015, 2016)):
        data.save_year(y, data.parse_omni2(omni_year(y, 10 + i)))
    forecast.train([2014, 2015], [2016], [1, 6], version="vtest")
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def test_replay_evidence_and_repeat(sw_client):
    c = sw_client
    df = importlib.import_module("spaceml.spaceweather.data").load("2016-01-01", "2016-12-31")
    tmin = df["Dst"].idxmin()
    start, end = (tmin - pd.Timedelta(hours=36)).isoformat(), (tmin + pd.Timedelta(hours=36)).isoformat()
    q = c.get("/api/spaceweather/quality", params={"start": start, "end": end}).json()
    assert q["status"] in ("ready", "degraded") and q["provenance"][0]["year"] == 2016
    run = c.post("/api/runs", json={"module": "spaceweather", "input": {"start": start, "end": end}}).json()
    run = _wait(c, run["id"])
    assert run["status"] == "succeeded", run.get("error")
    r = run["result"]
    assert r["causality_check"]["passed"]
    assert len(r["issued"]) == 73 and set(r["forecast"]) == {"1", "6"}
    h6 = r["per_horizon"]["6"]
    assert h6["better_hours"] + h6["worse_hours"] + h6["tie_hours"] == h6["n"]
    # persistence really is Dst at issuance; truth is Dst at issuance + h
    i = 10
    t = pd.Timestamp(r["issued"][i])
    assert r["persistence"][i] == pytest.approx(df["Dst"][t])
    assert r["truth"]["6"][i] == pytest.approx(df["Dst"][t + pd.Timedelta(hours=6)])
    kinds = {e["kind"] for e in r["timeline"]}
    assert {"min", "dst50"} <= kinds
    assert r["in_sample_years"] == [] and r["model"]["train_years"] == [2014, 2015]

    man = c.get(f"/api/runs/{run['id']}/manifest").json()
    assert man["schema"] == "smc.run-manifest/1" and man["origin"] == "computed"
    assert man["model"]["version"] == "vtest" and all(len(f["sha256"]) == 64 for f in man["model"]["files"])
    assert man["data"]["files"][0]["year"] == 2016 and man["baseline"]["name"].startswith("Persistence")
    assert man["checks"]["causality_check"]["passed"] and man["repeat"]["available"]

    c.put(f"/api/runs/{run['id']}/reviews/replay", json={"verdict": "confirmed", "note": "peak timing checked"})
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/runs/{run['id']}/evidence.zip").content))
    assert {"manifest.json", "report.md", "result.json", "forecasts.csv", "timeline.csv", "reviews.csv"} <= set(z.namelist())
    assert "peak timing checked" in z.read("reviews.csv").decode()
    assert len(z.read("forecasts.csv").decode().splitlines()) == 1 + 73 * 2

    rep = c.post(f"/api/runs/{run['id']}/repeat").json()
    rep = _wait(c, rep["id"])
    assert rep["repeat_of"] == run["id"] and rep["model_version"] == "vtest"
    chk = c.get(f"/api/runs/{rep['id']}/reproducibility").json()
    assert chk["status"] == "identical" and chk["same_model"] and chk["compared"] >= 8


def test_replay_missing_data_and_unknown_event(sw_client):
    c = sw_client
    r = c.post("/api/runs", json={"module": "spaceweather", "input": {"event_id": "2024-05"}})
    assert r.status_code == 409 and "2024" in r.json()["detail"]
    assert c.post("/api/runs", json={"module": "spaceweather", "input": {"event_id": "nope"}}).status_code == 404
    q = c.get("/api/spaceweather/quality", params={"start": "2024-05-10", "end": "2024-05-12"}).json()
    assert q["status"] == "not_downloaded" and q["missing_years"] == [2024]


# ---------------------------------------------------------------- jobs + starter pack
def test_job_retry_and_interrupted(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    from app import db, jobs
    db.init_db()
    jobs.init()
    with db.tx() as c:
        c.execute("INSERT INTO jobs (id, task, module, title, params, status, created_at) VALUES "
                  "('j1','mars.traverse','mars','x','{}','running','2026-01-01')")
    jobs.init()  # simulated restart
    j = jobs.get("j1")
    assert j["status"] == "interrupted" and "Retry" in j["error"]
    monkeypatch.setattr(jobs, "_run", lambda *a, **k: None)  # do not start a process
    again = jobs.retry("j1")
    assert again["task"] == "mars.traverse" and again["id"] != "j1"


def test_starter_build_and_install(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    from spaceml.spaceweather import data, forecast
    for i, y in enumerate((2014, 2015, 2016)):
        data.save_year(y, data.parse_omni2(omni_year(y, 10 + i)))
    forecast.train([2014, 2015], [2016], [1, 6], version="vtest")
    # the pack builder bundles 2024 + replays the May 2024 event: use 2016 data relabelled as 2024 here
    df = data.parse_omni2(omni_year(2024, 99))
    data.save_year(2024, df)
    from app import starter
    p = starter.build()
    assert any(it["id"] == "spaceweather" for it in p["contents"]) and p["reference_runs"]
    rid = p["reference_runs"][-1]
    assert rid.startswith("ref-")
    # fresh machine: empty storage/models/data except the pack
    import shutil
    for d in ("data", "models", "storage"):
        shutil.rmtree(tmp_path / d)
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        st = c.get("/api/starter").json()
        assert st["present"] and st["installed"]["complete"]
        runs = c.get("/api/spaceweather/replays", params={"event_id": "2024-05"}).json()
        assert runs[0]["id"] == rid and runs[0]["origin"] == "bundled"
        assert c.get(f"/api/runs/{rid}/manifest").json()["origin"] == "bundled"
        assert c.get("/api/spaceweather/status").json()["cached_years"] == [2024]
    # tampered pack file is refused
    for d in ("data", "models", "storage"):
        shutil.rmtree(tmp_path / d, ignore_errors=True)
    f = next((tmp_path / "starter" / "data" / "spaceweather").glob("*.npz"))
    f.write_bytes(f.read_bytes()[:-10] + b"0123456789")
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from app import db, starter as st2
    db.init_db()
    res = st2.install()
    assert any("checksum mismatch" in e for e in res["errors"])


# ---------------------------------------------------------------- traverse explorer
WAYPOINTS = {"type": "FeatureCollection", "name": "SELECT", "features": [
    {"type": "Feature", "properties": {"RMC": f"{s}_{d}", "site": s, "drive": d, "sol": sol, "lon": lon, "lat": lat,
                                       "dist_total_m": dist, "drive_type": "drive"},
     "geometry": {"type": "Point", "coordinates": [lon, lat, -4500.0]}}
    for s, d, sol, lon, lat, dist in [(2, 0, 3, 137.441633, -4.589467, 0.0), (48, 0, 990, 137.3700, -4.7200, 10500.0),
                                      (48, 1570, 1000, 137.3720, -4.7230, 10620.0), (49, 0, 1003, 137.3740, -4.7260, 10700.0)]]}


def test_traverse_parsing_and_location(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    from spaceml.mars import traverse
    pid = traverse.parse_product_id("NLB_486272784EDR_F0481570NCAM00415M_.JPG")
    assert (pid["site"], pid["drive"], pid["sol_estimate"]) == (48, 1570, 1000)   # NASA API: sol 1000, site 48, drive 1570
    assert traverse.parse_product_id("photo_001.png") is None
    assert traverse.locate("NLB_486272784EDR_F0481570NCAM00415M_")["precision"] == "unlocated"
    traverse.WAYPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    traverse.WAYPOINTS_PATH.write_text(json.dumps(WAYPOINTS))
    loc = traverse.locate("NLB_486272784EDR_F0481570NCAM00415M_")
    assert loc["precision"] == "exact" and loc["dist_total_m"] == 10620.0
    assert traverse.locate("NLB_486272784EDR_F0481600NCAM00415M_")["precision"] == "site"
    assert traverse.locate("x", {"sol": 1002, "site": None, "drive": None})["precision"] == "sol"
    sec = traverse.sections([{"dist_total_m": 10620, "class_shares": {"soil": 0.6, "sand": 0.4}, "reviewed": True},
                             {"dist_total_m": 10700, "class_shares": {"soil": 0.2, "sand": 0.8}, "corrected": True}], 1000)
    assert len(sec) == 1 and sec[0]["shares"]["sand"] == pytest.approx(0.6) and sec[0]["corrected"] == 1
    with pytest.raises(traverse.TraverseError):
        traverse._parse_waypoints({"type": "Feature"})


def test_traverse_api_with_fixture_and_unavailable_source(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from spaceml.common.fetch import SourceUnavailable
    from spaceml.mars import traverse
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        assert c.get("/api/mars/traverse").json()["status"]["downloaded"] is False

        def offline(*a, **k):
            raise SourceUnavailable("NASA/JPL MMGIS", traverse.WAYPOINTS_URL, "timed out")
        monkeypatch.setattr(traverse, "fetch", offline)
        r = c.post("/api/mars/traverse/download")
        assert r.status_code == 503 and r.json()["detail"]["status"] == "source_unavailable"

        def fake_fetch(url, dest, **k):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(WAYPOINTS))
            return {"source": "test", "url": url, "bytes": dest.stat().st_size, "sha256": "0" * 64}
        monkeypatch.setattr(traverse, "fetch", fake_fetch)
        assert c.post("/api/mars/traverse/download").json()["waypoints"] == 4
        ov = c.get("/api/mars/traverse").json()
        assert ov["waypoints"]["sol"] == [3, 990, 1000, 1003] and ov["status"]["total_km"] == 10.7

        item = {"imageid": "NLB_486272784EDR_F0481570NCAM00415M_", "sol": 1000, "site": 48, "drive": 1570,
                "instrument": "NAV_LEFT_B", "url": "http://mars.jpl.nasa.gov/msl-raw-images/x.JPG",
                "date_taken": "2015-05-30T16:02:29.000Z", "extended": {"lmst": "Sol-01000M14:01:56.851", "sample_type": "full"}}

        def fake_json(url, cache, **k):
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {"items": [item] if "NAV_LEFT_B" in url else [], "total": 1}
            cache.write_text(json.dumps(payload))
            return payload, {"source": "test", "url": url}
        monkeypatch.setattr(traverse, "fetch_json", fake_json)
        w = c.get("/api/mars/traverse/waypoint", params={"sol": 1000, "site": 48, "drive": 1570}).json()
        assert w["raw_status"] == "ready" and w["raw"]["images"][0]["match"] == "exact"
        # a raw image from a non-NASA host is refused
        item["url"] = "http://evil.example.com/x.JPG"
        (traverse.RAW_DIR / "sol_01000_NAV_LEFT_B.json").write_text(json.dumps({"items": [item]}))
        assert c.get(f"/api/mars/traverse/raw/{item['imageid']}").status_code == 404
