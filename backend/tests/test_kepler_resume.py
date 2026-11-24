"""Kepler ingestion is checkpointed per quarter: a crash mid-star resumes from the last completed quarter."""
import sys
import types

import numpy as np
import pytest


class _LC:
    def __init__(self, q):
        self.q = q
        self.meta = {"TEFF": 5700.0}


class _Product:
    def __init__(self, q, log, fail_at):
        self.q, self.log, self.fail_at = q, log, fail_at

    def download(self, download_dir, flux_column):
        import pathlib
        f = pathlib.Path(download_dir) / f"q{self.q}.fits"
        if f.exists():                       # astroquery reuses a complete file
            self.log.append(("reuse", self.q))
            return _LC(self.q)
        if self.q in self.fail_at:
            raise ConnectionError(f"simulated drop at quarter {self.q}")
        f.write_text("x")
        self.log.append(("download", self.q))
        return _LC(self.q)


class _Search(list):
    def __getitem__(self, i):
        r = list.__getitem__(self, i)
        return _Search(r) if isinstance(i, (list, slice)) else r


def _fake_lk(log, fail_at):
    m = types.ModuleType("lightkurve")
    m.search_lightcurve = lambda *a, **k: _Search(_Product(q, log, fail_at) for q in range(6))

    class Coll(list):
        def stitch(self):
            n = 100 * len(self)
            lc = types.SimpleNamespace(time=types.SimpleNamespace(value=np.arange(n, dtype=float)),
                                       flux=types.SimpleNamespace(value=np.ones(n)))
            lc.remove_nans = lambda: lc
            return lc
    m.LightCurveCollection = Coll
    return m


def test_restart_resumes_from_last_completed_quarter(tmp_path, monkeypatch):
    from spaceml.exoplanet import data
    monkeypatch.setattr(data, "LC_DIR", tmp_path)
    monkeypatch.setattr(data, "QUARTER_DIR", tmp_path / "quarters")
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    log = []
    # first run: the link drops after quarter 1 -> the star fails, quarters 0..1 stay on disk
    monkeypatch.setitem(sys.modules, "lightkurve", _fake_lk(log, fail_at={2, 3, 4, 5}))
    with pytest.raises(RuntimeError, match="kept"):
        data.fetch_lightcurve(42, retries=2)
    assert [q for kind, q in log if kind == "download"] == [0, 1]
    assert sorted(p.name for p in (tmp_path / "quarters" / "42").iterdir()) == ["q0.fits", "q1.fits"]

    # "restart": the network is back; completed quarters are reused, only 2..5 are fetched
    log.clear()
    monkeypatch.setitem(sys.modules, "lightkurve", _fake_lk(log, fail_at=set()))
    t, f, star = data.fetch_lightcurve(42, retries=2)
    assert [q for kind, q in log if kind == "reuse"] == [0, 1]
    assert [q for kind, q in log if kind == "download"] == [2, 3, 4, 5]
    assert len(t) == 600
    assert data.lc_path(42).exists() and not (tmp_path / "quarters" / "42").exists()


def test_one_unavailable_quarter_does_not_block_the_star(tmp_path, monkeypatch):
    from spaceml.exoplanet import data
    monkeypatch.setattr(data, "LC_DIR", tmp_path)
    monkeypatch.setattr(data, "QUARTER_DIR", tmp_path / "quarters")
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    log = []
    monkeypatch.setitem(sys.modules, "lightkurve", _fake_lk(log, fail_at={2}))
    t, f, _ = data.fetch_lightcurve(7, retries=3)
    assert len(t) == 500                        # 5 of 6 quarters
    assert int(np.load(data.lc_path(7))["quarters_missing"]) == 1
