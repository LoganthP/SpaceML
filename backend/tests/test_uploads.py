"""Upload parsing + transit search. Real FITS samples come from lightkurve's test data when
available (tests/data/lightkurve/*); generated files only exercise the code paths."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL = Path(__file__).parent / "data"


def _transit_flux(t, period=3.21, t0=1.3, dur_hr=2.6, depth=2500e-6, noise=4e-4, seed=0):
    rng = np.random.default_rng(seed)
    f = 1 + 0.003 * np.sin(t / 5.3) + rng.normal(0, noise, len(t))
    ph = (t - t0 + period / 2) % period - period / 2
    f[np.abs(ph) < dur_hr / 48] -= depth
    return f


def _fits_sector(path: Path, t: np.ndarray, sector: int, flux_scale=12000.0):
    from astropy.io import fits

    f = _transit_flux(t, seed=sector) * flux_scale
    q = np.zeros(len(t), np.int32)
    q[::97] = 1024  # some flagged cadences
    cols = [fits.Column("TIME", "D", array=t), fits.Column("PDCSAP_FLUX", "E", array=f),
            fits.Column("PDCSAP_FLUX_ERR", "E", array=np.full(len(t), 5.0)), fits.Column("QUALITY", "J", array=q)]
    h0 = fits.Header({"TELESCOP": "TESS", "OBJECT": "TIC 1234567", "TICID": 1234567, "RA_OBJ": 84.29,
                      "DEC_OBJ": -80.47, "TEFF": 5870.0, "RADIUS": 1.1, "LOGG": 4.4, "SECTOR": sector})
    h1 = fits.Header({"BJDREFI": 2457000, "BJDREFF": 0.0, "TIMESYS": "TDB"})
    fits.HDUList([fits.PrimaryHDU(header=h0), fits.BinTableHDU.from_columns(cols, header=h1)]).writeto(path)


def test_zip_of_two_tess_sectors_is_stitched_and_period_found(tmp_path):
    from spaceml.exoplanet import ingest, search

    for s, (a, b) in enumerate([(1500.0, 1527.0), (1530.0, 1556.0)], 1):
        _fits_sector(tmp_path / f"tess_s{s}.fits", np.arange(a, b, 2 / 1440), s)
    z = tmp_path / "upload.zip"
    with zipfile.ZipFile(z, "w") as zz:
        for s in (1, 2):
            zz.write(tmp_path / f"tess_s{s}.fits", f"folder/tess_s{s}.fits")
        zz.writestr("__MACOSX/._junk", b"xx")
        zz.writestr("readme.pdf", b"%PDF-1.4 not data")
    out = tmp_path / "x"
    out.mkdir()
    targets, warnings, skipped = ingest.parse_files(ingest.expand(z, out))
    assert len(targets) == 1 and len(targets[0].segments) == 2
    t, y, e = targets[0].arrays()
    assert abs(np.median(y) - 1) < 1e-3 and e is not None
    assert targets[0].segments[0].meta["teff"] == 5870.0
    res = search.bls_search(t, y)
    assert abs(res["best"]["period_days"] - 3.21) < 0.01
    assert 1500 < res["best"]["depth_ppm"] < 3500


def test_csv_magnitudes_headerless_and_multi_target_tables(tmp_path):
    from spaceml.exoplanet import ingest

    t = np.arange(0, 20, 0.02)
    mags = 12.0 - 2.5 * np.log10(_transit_flux(t))
    pd.DataFrame({"BJD_TDB": t + 2459000, "mag": mags, "mag_err": 0.001}).to_csv(tmp_path / "ground.csv", index=False)
    np.savetxt(tmp_path / "plain.txt", np.c_[t, _transit_flux(t), np.full(len(t), 4e-4)])
    df = pd.DataFrame({"star": ["A"] * len(t) + ["B"] * len(t), "time": np.r_[t, t],
                       "rel_flux": np.r_[_transit_flux(t), _transit_flux(t, seed=3)]})
    df.to_excel(tmp_path / "two_stars.xlsx", index=False)
    (tmp_path / "commented.dat").write_text("# time flux\n" + "\n".join(f"{a} {b}" for a, b in zip(t, _transit_flux(t))))
    targets, _, skipped = ingest.parse_files([tmp_path / n for n in
                                              ("ground.csv", "plain.txt", "two_stars.xlsx", "commented.dat")])
    assert not skipped
    names = sorted(tg.name for tg in targets)
    assert names == ["A", "B", "commented", "ground", "plain"]
    ground = next(tg for tg in targets if tg.name == "ground")
    assert ground.segments[0].columns["magnitudes"] is True
    _, y, _ = ground.arrays()
    assert y.min() < 0.9985  # magnitude dimming became a flux dip


@pytest.mark.skipif(not (REAL / "lightkurve").exists(), reason="real lightkurve sample files not present")
def test_real_tess_and_corrupted_files(tmp_path):
    from spaceml.exoplanet import ingest

    real = REAL / "lightkurve"
    targets, _, skipped = ingest.parse_files(sorted(real.glob("*.fits*")))
    names = " ".join(tg.name for tg in targets)
    assert "pi Men" in names or "261136679" in names or "TIC" in names
    assert any("corrupted" in s or "not a readable" in s or "image" in s or "no light-curve" in s for s in skipped)


def test_upload_api_end_to_end(tmp_path, monkeypatch):
    import importlib
    import sys
    import time

    monkeypatch.setenv("SPACEML_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SPACEML_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPACEML_STORAGE_DIR", str(tmp_path / "storage"))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    from fastapi.testclient import TestClient

    main = importlib.import_module("app.main")
    t = np.arange(0, 27, 0.01)
    buf = io.StringIO()
    pd.DataFrame({"time": t, "flux": _transit_flux(t)}).to_csv(buf, index=False)
    with TestClient(main.app) as c:
        r = c.post("/api/exoplanet/datasets", files={"file": ("my_star.csv", buf.getvalue().encode(), "text/csv")})
        assert r.status_code == 200, r.text
        ds = r.json()
        assert ds["targets"][0]["searchable"]
        bad = c.post("/api/exoplanet/datasets", files={"file": ("x.csv", b"hello", "text/csv")})
        assert bad.status_code == 422
        run = c.post("/api/runs", json={"module": "exoplanet",
                                        "input": {"dataset_id": ds["id"], "target_id": "t0"}}).json()
        for _ in range(120):
            got = c.get(f"/api/runs/{run['id']}").json()
            if got["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        assert got["status"] == "succeeded", got.get("error")
        res = got["result"]
        assert abs(res["tce"]["period_days"] - 3.21) < 0.02
        assert res["candidate_score"] is None and res["system"]["planets"][0]["a_rstar"] > 2
        assert res["diagnostics"]["odd_even_sigma"] is not None
