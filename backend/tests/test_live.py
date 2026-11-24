"""In-app live observing view: parsing both telescopes, caching, offline fallback, sky image limits.

The responses below follow the field names Space Telescope Live's own frontend uses (Webb: targets[].standardName,
scheduledStartTime, boreSight*; Hubble: targetName, predictedStartTime, target*InDegrees). The live site is not
reachable from the build machine, so the network is mocked."""
from __future__ import annotations

import importlib
import json
import sys

import pytest

WEBB = {"data": {"id": "01JWEBB0001", "scheduledStartTime": "2026-11-24T10:00:00Z", "scheduledEndTime": "2099-01-01T00:00:00Z",
                 "boreSightRightAscensionInDegrees": "83.82", "boreSightDeclinationInDegrees": "-5.39",
                 "targets": [{"standardName": "Orion Bar", "name": "ORION-BAR"}],
                 "proposal": {"title": "PDRs4All", "proposalID": 1288, "scientificCategory": {"title": "Stellar Physics"},
                              "instruments": [{"code": "NIRCAM", "title": "Near Infrared Camera"}, {"code": "MIRI", "title": "Mid-Infrared Instrument"}],
                              "primaryInvestigator": {"formalName": "Dr. Example"}}},
        "_pagination": {"nextRecord": {"startTime": "2099-01-01T00:00:00Z"}}}
HUBBLE = {"data": {"id": "HST42", "predictedStartTime": "2020-01-01T00:00:00", "predictedEndTime": "2020-01-01T01:00:00",
                   "targetName": "ANY,SNR-0509?67.5", "targetRightAscensionInDegrees": 77.4, "targetDeclinationInDegrees": -67.5,
                   "proposal": {"title": "Light echoes", "proposalID": 17001, "instruments": [{"code": "WFC3", "title": "Wide Field Camera 3"}]}}}


@pytest.fixture()
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEML_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("SPACEML_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SPACEML_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPACEML_AUTOSETUP", "off")
    monkeypatch.setenv("SMC_AUTO_INSTALL", "off")
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    return importlib.import_module("spaceml.common.stlive")


def _serve(monkeypatch, st, answers: dict):
    calls = []

    def fake(url, headers=None, timeout=12, max_bytes=0):
        calls.append((url, headers))
        for key, val in answers.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise st.SourceUnavailable("x", url, "network error: offline")
    monkeypatch.setattr(st, "_get", fake)
    return calls


def test_webb_and_hubble_are_normalized(live, monkeypatch):
    calls = _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode(), "/api/get/hubble": json.dumps(HUBBLE).encode()})
    w = live.current("webb")
    o = w["observation"]
    assert w["source"] == "live" and calls[0][1] == {"endpoint": "current"}
    assert o["target"] == "Orion Bar" and o["instruments"] == ["NIRCam", "MIRI"] and o["program"] == "1288"
    assert o["ra"] == pytest.approx(83.82) and o["phase"] == "in_progress" and o["next_start"].startswith("2099")
    h = live.current("hubble")["observation"]
    assert h["target"] == "SNR 0509−67.5" and h["start"] == "2020-01-01T00:00:00Z" and h["phase"] == "completed"


def test_offline_shows_last_known_observation(live, monkeypatch):
    _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode()})
    live.current("webb")
    _serve(monkeypatch, live, {})                                   # network gone
    r = live.current("webb", refresh=True)
    assert r["source"] == "cached" and "not reachable (no internet connection" in r["notice"] and r["observation"]["target"] == "Orion Bar"


def test_cache_avoids_refetching_within_max_age(live, monkeypatch):
    calls = _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode()})
    live.current("webb")
    live.current("webb")
    assert len(calls) == 1


def test_unreadable_answer_is_a_calm_notice(live, monkeypatch):
    _serve(monkeypatch, live, {"/api/get/webb": b'{"data": null}'})
    with pytest.raises(live.SourceUnavailable):
        live.current("webb")


def test_history_keeps_observations_seen(live, monkeypatch):
    _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode()})
    live.current("webb")
    nxt = json.loads(json.dumps(WEBB))
    nxt["data"]["id"] = "01JWEBB0002"
    _serve(monkeypatch, live, {"/api/get/webb": json.dumps(nxt).encode()})
    r = live.current("webb", refresh=True)
    assert r["observation"]["id"] == "01JWEBB0002" and [o["id"] for o in r["history"]] == ["01JWEBB0001"]


def test_sky_image_only_for_known_observations_and_cached(live, monkeypatch):
    jpg = b"\xff\xd8\xff\xe0fakejpeg"
    calls = _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode(), "hips2fits": jpg})
    live.current("webb")
    with pytest.raises(KeyError):
        live.sky("webb", "someone-elses-id", 0.5)
    with pytest.raises(KeyError):
        live.sky("webb", "01JWEBB0001", 7.0)                        # fov not offered
    p = live.sky("webb", "01JWEBB0001", 0.5)
    assert p.read_bytes() == jpg and "ra=83.820000" in calls[-1][0]
    n = len(calls)
    live.sky("webb", "01JWEBB0001", 0.5)
    assert len(calls) == n                                          # served from disk next time (works offline)


def test_api_routes(live, monkeypatch):
    from fastapi.testclient import TestClient
    _serve(monkeypatch, live, {})
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        r = c.get("/api/live/webb")
        assert r.status_code == 503 and "not available right now" in r.json()["detail"]
        assert c.get("/api/live/kepler").status_code == 404
        _serve(monkeypatch, live, {"/api/get/webb": json.dumps(WEBB).encode()})
        assert c.get("/api/live/webb").json()["observation"]["target"] == "Orion Bar"
        assert c.get("/api/live/webb/sky/..%2Fsecret?fov=0.5").status_code == 404
