"""Space weather: OMNI2 parsing, chronological forecast training vs persistence, API.

The OMNI2 text written here is generated (Burton-like Dst response to solar-wind driving)
only to exercise the parser and pipeline in a temporary folder.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

FILL = {8: 999.9, 15: 999.9, 16: 999.9, 22: 9999999.0, 23: 999.9, 24: 9999.0, 28: 99.99, 35: 999.99,
        38: 99, 39: 999, 40: 99999, 41: 9999, 49: 999, 50: 999.9}


def omni_year(year: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    n = 8760
    v = 400 + 60 * np.sin(np.arange(n) / 200) + rng.normal(0, 20, n)
    bz = np.zeros(n)
    for i in range(1, n):
        bz[i] = 0.9 * bz[i - 1] + rng.normal(0, 1.2)
    for s in rng.choice(n - 60, 6, replace=False):  # storms: strong southward IMF + fast wind
        bz[s:s + 18] -= 15
        v[s:s + 30] += 300
    dst = np.zeros(n)
    for i in range(1, n):
        q = -4.4 * (v[i - 1] * max(-bz[i - 1], 0) / 1000 - 0.5) if v[i - 1] * max(-bz[i - 1], 0) / 1000 > 0.5 else 0
        dst[i] = dst[i - 1] + q - dst[i - 1] / 8
    lines = []
    for i in range(n):
        f = [0.0] * 55
        f[0], f[1], f[2] = year, i // 24 + 1, i % 24
        f[8], f[15], f[16] = abs(bz[i]) + 3, rng.normal(0, 2), bz[i]
        f[22], f[23], f[24], f[28] = 1e5, 5 + rng.random() * 3, v[i], 2.0
        f[35] = -v[i] * bz[i] / 1000
        f[38], f[39], f[40], f[41], f[49], f[50] = int(min(90, max(0, -dst[i] / 3))), 50, round(dst[i]), 100, 10, 120
        if i % 500 == 7:  # gaps use the official fill values
            f[24], f[40] = FILL[24], FILL[40]
        lines.append(" ".join(f"{x:g}" for x in f))
    return "\n".join(lines)


def test_parse_handles_fill_values():
    from spaceml.spaceweather.data import parse_omni2
    df = parse_omni2(omni_year(2015, 1))
    assert len(df) == 8760 and df.index[0].year == 2015 and df.index[1].hour == 1
    assert df["V"].isna().sum() == len(range(7, 8760, 500)) and df["Kp"].max() <= 9


def test_forecast_beats_persistence_and_api(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from spaceml.spaceweather import data, forecast
    for i, y in enumerate((2014, 2015, 2016)):
        data.save_year(y, data.parse_omni2(omni_year(y, 10 + i)))
    per_h = forecast.train([2014, 2015], [2016], [1, 6], version="vtest")
    assert per_h[6]["rmse"] < per_h[6]["rmse_persistence"]  # learnt the solar-wind driving
    with pytest.raises(SystemExit):
        forecast.train([2015], [2015], [1])  # overlapping split refused

    from fastapi.testclient import TestClient
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        st = c.get("/api/spaceweather/status").json()
        assert st["cached_years"] == [2014, 2015, 2016] and st["model"]["horizons"] == [1, 6]
        s = c.get("/api/spaceweather/series", params={"start": "2016-03-01", "end": "2016-03-10"}).json()
        assert len(s["time"]) == 9 * 24 + 1 and s["summary"]["min_dst"] is not None
        f = c.get("/api/spaceweather/forecast", params={"start": "2016-03-01", "end": "2016-03-03", "horizon": 6}).json()
        assert len(f["forecast"]) == 49 and f["valid"][0].startswith("2016-03-01T06")
        a = c.post("/api/spaceweather/annotations", json={"start": "2016-03-02", "end": "2016-03-03", "label": "CME arrival"}).json()
        assert c.get("/api/spaceweather/events").json()["annotations"][0]["id"] == a["id"]
        ev = [e for e in c.get("/api/evaluation").json() if e["module"] == "spaceweather"][0]
        assert ev["active"]["baseline_name"].startswith("Persistence")
