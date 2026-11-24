"""NASA OMNIWeb OMNI2 hourly solar-wind + geomagnetic data.

    python -m spaceml.spaceweather.data download                  # default years (storms + training span)
    python -m spaceml.spaceweather.data download --years 2003 2015 2017 2024
    python -m spaceml.spaceweather.data status

Source: NASA/GSFC Space Physics Data Facility, low-resolution OMNI (OMNI2) hourly files
https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/  (format: omni2.text in that folder).
Each yearly ASCII file is parsed once and cached as data/spaceweather/omni2_<year>.npz.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..common.fetch import SourceUnavailable, fetch, sha256_file  # noqa: F401
from ..common.utils import now_iso, read_json, write_json
from ..config import DATA_DIR

SW_DIR = DATA_DIR / "spaceweather"
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
DEFAULT_YEARS = [2003, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2024]

# (name, 0-based column in omni2.text, fill value, unit, description)
COLUMNS = [
    ("B", 8, 999.9, "nT", "IMF magnitude |B|"),
    ("By", 15, 999.9, "nT", "IMF By (GSM)"),
    ("Bz", 16, 999.9, "nT", "IMF Bz (GSM)"),
    ("T", 22, 9999999.0, "K", "Proton temperature"),
    ("N", 23, 999.9, "cm⁻³", "Proton density"),
    ("V", 24, 9999.0, "km/s", "Solar-wind speed"),
    ("Pdyn", 28, 99.99, "nPa", "Flow pressure"),
    ("E", 35, 999.99, "mV/m", "Electric field (-V·Bz)"),
    ("Kp", 38, 99.0, "", "Kp index"),
    ("R", 39, 999.0, "", "Sunspot number"),
    ("Dst", 40, 99999.0, "nT", "Dst index"),
    ("AE", 41, 9999.0, "nT", "AE index"),
    ("ap", 49, 999.0, "nT", "ap index"),
    ("F107", 50, 999.9, "sfu", "F10.7 radio flux"),
]
META = {c[0]: {"unit": c[3], "label": c[4]} for c in COLUMNS}

# Well-documented geomagnetic storms (UTC). The UI computes the minimum Dst from the data itself.
EVENTS = [
    {"id": "1989-03", "name": "March 1989 storm (Québec blackout)", "start": "1989-03-12T00:00Z", "end": "1989-03-16T00:00Z"},
    {"id": "2001-03", "name": "March 2001 storm", "start": "2001-03-30T12:00Z", "end": "2001-04-02T00:00Z"},
    {"id": "2003-10", "name": "Halloween storms 2003", "start": "2003-10-28T00:00Z", "end": "2003-11-02T00:00Z"},
    {"id": "2003-11", "name": "November 2003 storm", "start": "2003-11-19T00:00Z", "end": "2003-11-22T00:00Z"},
    {"id": "2004-11", "name": "November 2004 storms", "start": "2004-11-07T00:00Z", "end": "2004-11-12T00:00Z"},
    {"id": "2015-03", "name": "St. Patrick's Day storm 2015", "start": "2015-03-16T12:00Z", "end": "2015-03-20T00:00Z"},
    {"id": "2015-06", "name": "June 2015 storm", "start": "2015-06-21T12:00Z", "end": "2015-06-25T00:00Z"},
    {"id": "2017-09", "name": "September 2017 storm", "start": "2017-09-06T12:00Z", "end": "2017-09-10T00:00Z"},
    {"id": "2024-05", "name": "Gannon storm, May 2024", "start": "2024-05-10T00:00Z", "end": "2024-05-14T00:00Z"},
    {"id": "2024-10", "name": "October 2024 storm", "start": "2024-10-09T12:00Z", "end": "2024-10-13T00:00Z"},
]


class FormatError(ValueError):
    pass


def parse_omni2(text: str) -> pd.DataFrame:
    """Parse an OMNI2 hourly ASCII file into a DataFrame indexed by UTC time."""
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    rows = [r for r in rows if len(r) >= 51]
    if not rows:
        raise FormatError("No OMNI2 records found (expected ≥ 51 whitespace-separated fields per line)")
    a = np.array([[float(x) for x in r[:55]] if len(r) >= 55 else [float(x) for x in r] + [np.nan] * (55 - len(r))
                  for r in rows], dtype=np.float64)
    year, doy, hour = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int)
    t = pd.to_datetime(year.astype(str), format="%Y", utc=True) + pd.to_timedelta(doy - 1, "D") + pd.to_timedelta(hour, "h")
    df = pd.DataFrame(index=pd.DatetimeIndex(t, name="time"))
    for name, col, fill, *_ in COLUMNS:
        v = a[:, col].copy()
        v[v >= fill * 0.999] = np.nan
        df[name] = v
    df["Kp"] = df["Kp"] / 10.0  # OMNI2 stores Kp*10
    _sanity(df)
    return df


def _sanity(df: pd.DataFrame) -> None:
    """Guard against a format change.

    1. typical solar-wind values must come out, and
    2. physics must hold between independently stored columns: OMNI2 derives
       E = -V*Bz(GSM)*1e-3 mV/m and Pdyn = 2e-6*N*V^2 nPa. If a column index were off,
       these identities break immediately.
    """
    v, d = df["V"].median(), df["Dst"].median()
    if not (200 < v < 900) or not (-80 < d < 40):
        raise FormatError(f"Unexpected values after parsing (median V={v:.0f} km/s, Dst={d:.0f} nT). "
                          "The OMNI2 format may have changed; please report this.")
    ok = df[["V", "Bz", "E", "N", "Pdyn"]].notna().all(axis=1)
    if ok.sum() >= 10:
        x = df[ok]
        e_err = np.median(np.abs(x["E"] + x["V"] * x["Bz"] / 1000.0))
        p_rel = np.median(np.abs(x["Pdyn"] - 2e-6 * x["N"] * x["V"] ** 2) / np.maximum(x["Pdyn"], 0.05))
        if e_err > 0.05 or p_rel > 0.25:
            raise FormatError(f"OMNI2 columns are inconsistent (E vs -V*Bz off by {e_err:.2f} mV/m, "
                              f"Pdyn vs 2e-6*N*V^2 off by {p_rel:.0%}). The file format may have changed.")


def dst_status(year: int) -> str:
    """Dst quality level as documented by OMNIWeb (final / provisional / quick-look)."""
    if year <= 2020:
        return "final"
    if year <= 2024:
        return "provisional"
    return "provisional or quick-look"


def cache_path(year: int):
    return SW_DIR / f"omni2_{year}.npz"


def cached_years() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in SW_DIR.glob("omni2_*.npz")) if SW_DIR.exists() else []


def save_year(year: int, df: pd.DataFrame) -> None:
    SW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(year).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=(df.index.view("int64") // 10**9).astype(np.int64),
                        **{c: df[c].to_numpy(np.float32) for c in df.columns})
    tmp.replace(cache_path(year))


def fetch_year(year: int) -> int:
    """Download, parse and cache one year. Returns the record count (0 = already cached)."""
    if cache_path(year).exists():
        return 0
    raw = SW_DIR / f"omni2_{year}.dat"
    prov = fetch(BASE_URL.format(year=year), raw, source="NASA SPDF OMNIWeb", timeout=60, retries=3)
    try:
        df = parse_omni2(raw.read_text(errors="replace"))
    finally:
        raw.unlink(missing_ok=True)
        raw.with_name(raw.name + ".source.json").unlink(missing_ok=True)
    save_year(year, df)
    write_json(SW_DIR / f"omni2_{year}.source.json", {
        **prov, "file": f"omni2_{year}.dat", "records": int(len(df)), "parsed_at": now_iso(),
        "first": df.index[0].isoformat(), "last": df.index[-1].isoformat(), "dst_status": dst_status(year),
        "cache_sha256": sha256_file(cache_path(year))})
    return len(df)


def import_file(path, year: int | None = None) -> int:
    """Import an OMNI2 hourly file you downloaded yourself (offline machines, mirrors)."""
    from pathlib import Path
    p = Path(path)
    df = parse_omni2(p.read_text(errors="replace"))
    years = sorted({t.year for t in df.index}) if year is None else [year]
    for y in years:
        part = df[df.index.year == y]
        save_year(y, part)
        write_json(SW_DIR / f"omni2_{y}.source.json", {
            "source": "NASA SPDF OMNIWeb (imported file)", "url": None, "file": p.name,
            "sha256": sha256_file(p), "retrieved_at": now_iso(), "records": int(len(part)),
            "first": part.index[0].isoformat(), "last": part.index[-1].isoformat(), "dst_status": dst_status(y),
            "cache_sha256": sha256_file(cache_path(y))})
    return len(df)


def provenance(years: list[int]) -> list[dict]:
    out = []
    for y in years:
        rec = read_json(SW_DIR / f"omni2_{y}.source.json") or {"file": f"omni2_{y}.dat", "note": "provenance not recorded"}
        if cache_path(y).exists() and "cache_sha256" not in rec:
            rec["cache_sha256"] = sha256_file(cache_path(y))
        out.append({"year": y, "dst_status": dst_status(y), **rec})
    return out


def load(start: str | None = None, end: str | None = None, years: list[int] | None = None) -> pd.DataFrame:
    yrs = years or cached_years()
    if start:
        y0 = ts(start).year
        y1 = ts(end).year if end else y0
        yrs = [y for y in yrs if y0 <= y <= y1]
    frames = []
    for y in yrs:
        p = cache_path(y)
        if not p.exists():
            continue
        z = np.load(p)
        df = pd.DataFrame({c: z[c] for c in z.files if c != "t"},
                          index=pd.to_datetime(z["t"], unit="s", utc=True))
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No OMNIWeb data cached for that period. Run: python -m spaceml.spaceweather.data download")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    if start:
        df = df[df.index >= ts(start)]
    if end:
        df = df[df.index <= ts(end)]
    return df


def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def events_with_stats() -> list[dict]:
    have = set(cached_years())
    out = []
    for e in EVENTS:
        rec = dict(e)
        y = int(e["start"][:4])
        rec["cached"] = y in have
        if rec["cached"]:
            try:
                d = load(e["start"], e["end"])
                if d["Dst"].notna().any():
                    rec["min_dst"] = float(d["Dst"].min())
                    rec["min_dst_time"] = d["Dst"].idxmin().isoformat()
                    rec["max_kp"] = float(d["Kp"].max())
                    rec["max_v"] = float(d["V"].max())
            except FileNotFoundError:
                rec["cached"] = False
        out.append(rec)
    return out


def download_years(years: list[int]) -> None:
    from ..common.progress import report, step
    SW_DIR.mkdir(parents=True, exist_ok=True)
    step(f"Downloading OMNI2 hourly data for {len(years)} year(s) from NASA SPDF")
    errs = []
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fetch_year, y): y for y in years}
        for i, (f, y) in enumerate(((f, futs[f]) for f in futs), 1):
            try:
                n = f.result()
                print(f"  {y}: {'cached' if n == 0 else f'{n:,} hourly records'}", flush=True)
            except Exception as e:  # noqa: BLE001
                errs.append(y)
                print(f"  {y}: failed - {e}", flush=True)
            report(i, len(years), f"{y} done")
    print(f"Cached years: {cached_years()}")
    if errs and len(errs) == len(years):
        from ..common.progress import fail
        fail("source_unavailable: no OMNI2 file could be downloaded from NASA SPDF "
             "(https://spdf.gsfc.nasa.gov). Check the internet connection, or download omni2_YYYY.dat "
             "elsewhere and upload them on the Mission Replay page (Upload omni2_YYYY.dat files).")


def main() -> None:
    p = argparse.ArgumentParser(description="NASA OMNIWeb OMNI2 hourly data")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    i = sub.add_parser("import", help="import omni2_YYYY.dat files you downloaded yourself")
    i.add_argument("files", nargs="+")
    sub.add_parser("status")
    a = p.parse_args()
    if a.cmd == "download":
        download_years(sorted(set(a.years)))
    elif a.cmd == "import":
        for f in a.files:
            print(f"{f}: {import_file(f):,} hourly records")
    else:
        print({"cached_years": cached_years(), "dir": str(SW_DIR)})


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
