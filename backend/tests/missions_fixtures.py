"""SYNTHETIC fixtures that follow each source's documented file layout (never shipped, never shown as real data).

They exercise the importers, detectors, runs and UI where the real sources (MAST) were not reachable from the build environment. Real-data verification for those sources
is done by `python -m spaceml.missions.prepare ...` on a machine with access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def hubble(folder: Path, n_exp: int = 6, seed: int = 1) -> Path:
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    root = "iepp01010"
    jit_hdus, jif_hdus = [fits.PrimaryHDU()], [fits.PrimaryHDU()]
    jit_hdus[0].header.update(NEXTEND=n_exp, PROPOSID=16748, TARGNAME="NGC-1898")
    jif_hdus[0].header.update(NEXTEND=n_exp, PROPOSID=16748, TARGNAME="NGC-1898")
    for e in range(n_exp):
        n = 300
        sec = np.arange(n) * 3.0
        v2 = rng.normal(0, 0.002, n)
        v3 = rng.normal(0, 0.002, n)
        r2 = np.abs(rng.normal(0.004, 0.0005, n))
        r3 = np.abs(rng.normal(0.004, 0.0005, n))
        if e in (1, 4):                       # a jitter burst / drift in two exposures
            r2[120:150] += 0.05
            v3[200:260] += np.linspace(0, 0.08, 60)
        v2[10] = -32767.0                     # documented missing-value sentinel
        cols = [fits.Column(name="Seconds", format="E", array=sec, unit="s"),
                fits.Column(name="SI_V2_AVG", format="E", array=v2, unit="arcsec"),
                fits.Column(name="SI_V3_AVG", format="E", array=v3, unit="arcsec"),
                fits.Column(name="SI_V2_RMS", format="E", array=r2 * 1000, unit="mas"),
                fits.Column(name="SI_V3_RMS", format="E", array=r3 * 1000, unit="mas")]
        h = fits.BinTableHDU.from_columns(cols)
        h.header.update(EXPNAME=f"IEPP01{e:02d}Q", EXPSTART=59500.0 + e * 0.01)
        jit_hdus.append(h)
        jh = fits.ImageHDU(np.zeros((4, 4), np.int16))
        jh.header.update(EXPNAME=f"IEPP01{e:02d}Q", GUIDEACT="FINE LOCK", GSACQ="2021.300:00:00:00",
                         GSFAIL="NOLOCK" if e == 4 else "", NLOSSES=1 if e == 4 else 0,
                         V2_RMS=4.0 if e != 4 else -32767, V3_RMS=4.0 if e != 4 else -32767)
        jif_hdus.append(jh)
    fits.HDUList(jit_hdus).writeto(folder / f"{root}_jit.fits", overwrite=True)
    fits.HDUList(jif_hdus).writeto(folder / f"{root}_jif.fits", overwrite=True)
    return folder


def webb(folder: Path, n_visits: int = 5, seed: int = 2, spaced_names: bool = True, time_mode: str = "ms") -> Path:
    """time_mode "ms": relative clock in ms (unit ms); "mjd": MJD days with no TUNIT, as in real MAST
    gs-fg cal products (15.6 Hz cadence)."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    names = (["Time", "Jitter", "Delta DDC RA", "Delta DDC Dec", "Delta Aper PA", "HGA motion"] if spaced_names else
             ["time", "jitter", "delta_ddc_ra", "delta_ddc_dec", "delta_aper_pa", "hga_motion"])
    for v in range(n_visits):
        for seg in range(2):                  # segmented guider files, concatenated per visit
            n = 2000
            stamp = f"2023{150 + v:03d}{10 + seg:02d}0000"
            tms = np.arange(n) * 64.0         # 64 ms cadence
            tunit = "ms"
            if time_mode == "mjd":           # end of data = file stamp, expressed as MJD days
                from datetime import datetime, timezone
                end = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() + (149 + v) * 86400 + (10 + seg) * 3600
                tms = (end - (n - 1 - np.arange(n)) * 0.064) / 86400.0 + 40587.0
                tunit = None
            ra, dec = rng.normal(0, 1.0, n), rng.normal(0, 1.0, n)
            if v == 2 and seg == 1:
                ra[900:1000] += 25.0
            jit = np.sqrt(ra ** 2 + dec ** 2)
            hga = np.where((np.arange(n) > 1500) & (v == 3), 0, 1)
            cols = [fits.Column(name=names[0], format="D", array=tms, unit=tunit),
                    fits.Column(name=names[1], format="E", array=jit, unit="mas"),
                    fits.Column(name=names[2], format="E", array=ra, unit="mas"),
                    fits.Column(name=names[3], format="E", array=dec, unit="mas"),
                    fits.Column(name=names[4], format="E", array=rng.normal(0, 0.5, n), unit="mas"),
                    fits.Column(name=names[5], format="I", array=hga)]
            p = fits.BinTableHDU.from_columns(cols, name="POINTING")
            p.header.update(JITTRAVG=float(jit.mean()), JITTRMIN=float(jit.min()), JITTRMAX=float(jit.max()))
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(), fits.ImageHDU(), p]).writeto(
                folder / f"jw01234{v + 1:03d}001_gs-fg_{stamp}_cal.fits", overwrite=True)
    return folder
