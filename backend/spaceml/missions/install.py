"""Install a mission's data from the UI: download it from the official source, or use files the user uploaded,
then run the validated importer (prepare). Used by the "Install data" job and by the optional automatic install
at startup (SMC_AUTO_INSTALL). With no file names, Hubble and Webb fetch the newest public data from MAST.

    python -m spaceml.missions.install <mission> --download [--files a b ...]
    python -m spaceml.missions.install <mission> --source <uploaded folder>

What each source supports is described by SPECS (also shown in the UI).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ..common.progress import fail, report, step
from ..common.zipsupport import enable_deflate64
from ..config import DATA_DIR
from . import catalog

enable_deflate64()                 # uploaded archives may use Deflate64 (ZIP method 9)

SRC_ROOT = DATA_DIR / "missions-source"
MAST = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:{coll}/product/{name}"
MAST_DL = "https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
AUTO_N = {"hubble": 12, "webb": 10}
UA = {"User-Agent": "space-ml-mission-control (+https://github.com/LoganthP)"}


def auto_source(mission: str) -> dict | None:
    """What one click on "Install data" fetches when no file names are given (shown in the UI)."""
    return {
        "opssat": {"label": "Download the newest OPSSAT-AD release (v2) from Zenodo", "size": "≈ 20 MB", "automatic": True},
        "hubble": {"label": f"Download the {AUTO_N['hubble']} newest public Hubble jitter logs from MAST", "size": "≈ 5-20 MB", "automatic": True},
        "webb": {"label": f"Download the newest public Webb fine-guidance data ({AUTO_N['webb']} visits) from MAST", "size": "≈ 5-50 MB", "automatic": True},
    }.get(mission)


SPECS: dict[str, dict] = {
    "opssat": {"download": {"label": "Download OPSSAT-AD v2 from Zenodo", "size": "≈ 20 MB", "fields": []},
               "upload": {"accept": ".csv,.zip", "hint": "dataset.csv and segments.csv from the Zenodo record (or a zip of both)"}},
    "hubble": {"download": {"label": "Download from MAST", "size": "≈ 0.1-2 MB per exposure",
                            "fields": [{"name": "files", "label": "HST observation-log files or rootnames",
                                        "placeholder": "e.g. abcd01xyj_jit.fits abcd01xyj_jif.fits  (or just the rootname)",
                                        "help": "Find them in the MAST Portal: search your HST observation, open its products and filter for 'jit' / 'jif'."}]},
               "upload": {"accept": ".fits,.fit,.zip", "hint": "*_jit.fits and matching *_jif.fits files (or a zip)"}},
    "webb": {"download": {"label": "Download from MAST", "size": "≈ 0.05-5 MB per guider file",
                          "fields": [{"name": "files", "label": "JWST guider product file names",
                                      "placeholder": "e.g. jw01234001001_gs-fg_2022163160927_cal.fits",
                                      "help": "In the MAST Portal open a JWST visit's products and filter for 'gs-fg' or 'gs-track' cal files."}]},
             "upload": {"accept": ".fits,.fit,.zip", "hint": "*_gs-fg_*_cal.fits / *_gs-track_*_cal.fits files (or a zip)"}},
}


def _fetch(url: str, out: Path, timeout: int = 120, headers: dict | None = None, label: str = "") -> Path:
    """Resumable download (keeps .part between attempts)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    for attempt in range(4):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={**(headers or {}), **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length") or 0) + (have if r.status == 206 else 0)
                mode = "ab" if r.status == 206 else "wb"
                done = have if r.status == 206 else 0
                with open(part, mode) as f:
                    t = time.time()
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t > 2:
                            report(done, total or done, f"downloading {label or out.name}: {done / 1e6:.0f}"
                                   + (f" / {total / 1e6:.0f} MB" if total else " MB"))
                            t = time.time()
            part.replace(out)
            return out
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise ValueError(f"{label or url}: HTTP {e.code} ({'not found - check the name' if e.code == 404 else 'access denied'})")
            last = e
        except Exception as e:  # noqa: BLE001 - network hiccup: retry
            if attempt == 0 and isinstance(e, urllib.error.URLError) and "Name or service not known" in str(e):
                raise ValueError(f"source_unavailable: no internet connection (could not resolve {urllib.parse.urlparse(url).netloc})")
            last = e
        time.sleep(3 * (attempt + 1))
    raise ValueError(f"source_unavailable: could not download {label or url}: {last}")


def _extract_zips(folder: Path) -> None:
    for z in list(folder.rglob("*.zip")):
        if z.parent.name == "channels":         # ESA channel files are zip-pickles, not archives to unpack
            continue
        with zipfile.ZipFile(z) as f:
            for m in f.infolist():
                target = (z.parent / m.filename).resolve()
                if not str(target).startswith(str(z.parent.resolve())):
                    raise ValueError(f"{z.name}: unsafe path {m.filename}")
            step(f"Unpacking {z.name}")
            f.extractall(z.parent)
        z.unlink()


def _find(folder: Path, mission: str) -> Path:
    """The importer input inside an uploaded / downloaded folder."""
    if mission == "opssat":
        hit = next((p.parent for p in folder.rglob("dataset.csv") if (p.parent / "segments.csv").exists()), None)
        if not hit:
            raise ValueError("needs dataset.csv and segments.csv (both files from the OPSSAT-AD record)")
        return hit
    if mission in ("hubble", "webb"):
        if not any(folder.rglob("*.fits")):
            raise ValueError("no .fits files found")
        return folder
    raise ValueError(f"no installer for {mission}")


def _mast(request: dict) -> list[dict]:
    """One MAST portal API call (POST /api/v0/invoke); returns the data rows."""
    body = urllib.parse.urlencode({"request": json.dumps({"format": "json", "pagesize": 5000, "page": 1, **request})}).encode()
    for attempt in range(6):
        req = urllib.request.Request(MAST_INVOKE, data=body, headers={**UA, "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        status = str(out.get("status", "COMPLETE")).upper()
        if status == "COMPLETE":
            return out.get("data") or []
        if status == "ERROR":
            raise ValueError(f"MAST rejected the query: {out.get('msg') or out}")
        time.sleep(2 + attempt * 2)                           # EXECUTING: poll again
    raise ValueError("source_unavailable: MAST did not finish the query - try again later")


def _mjd_now() -> float:
    return time.time() / 86400.0 + 40587.0


def _download_some(cands: list[tuple[str, str]], dest: Path, want: int, what: str, group=lambda n: n) -> int:
    """Download candidates (url, name) newest first, skipping files that are not public yet (401/403)."""
    got, seen, denied = 0, set(), 0
    for i, (url, name) in enumerate(cands):
        if got >= want:
            break
        g = group(name)
        if g in seen:
            continue
        report(got, want, f"downloading {name} ({got + 1}/{want})")
        try:
            _fetch(url, dest / name, label=name)
        except ValueError as e:
            if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                denied += 1                                   # still in its exclusive-access period
                continue
            raise
        seen.add(g)
        got += 1
    if not got:
        raise ValueError(f"no public {what} could be downloaded ({denied} recent files are not public yet)"
                         if denied else f"MAST returned no {what} for the search window")
    return got


def _webb_latest(dest: Path, want: int) -> None:
    """Newest public JWST fine-guidance (gs-fg) cal products, one per visit (MAST Mast.Jwst.Filtered.GuideStar)."""
    now = _mjd_now()
    rows: list[dict] = []
    for days in (120, 400, 900):
        filt = [{"paramName": "expstart", "values": [{"min": now - days, "max": now}]}]
        try:
            rows = _mast({"service": "Mast.Jwst.Filtered.GuideStar",
                          "params": {"columns": "*", "filters": filt + [{"paramName": "filename", "values": [],
                                                                          "freeText": "%gs-fg%cal.fits"}]}})
        except ValueError:
            rows = _mast({"service": "Mast.Jwst.Filtered.GuideStar", "params": {"columns": "*", "filters": filt}})
        rows = [r for r in rows if re.search(r"_gs-fg_\d+_cal\.fits$", str(r.get("filename") or r.get("productFilename") or ""))]
        if len(rows) >= want:
            break
    rows.sort(key=lambda r: -float(r.get("expstart") or 0))
    names = [str(r.get("filename") or r.get("productFilename")) for r in rows]
    step(f"MAST lists {len(names)} recent Webb fine-guidance files; downloading the newest {want} public visits")
    _download_some([(MAST.format(coll="JWST", name=urllib.parse.quote(n)), n) for n in names], dest, want,
                   "Webb guider files", group=lambda n: n.split("_gs-")[0])


def _hubble_latest(dest: Path, want: int) -> None:
    """Newest public HST observations that have jitter logs (_jit + _jif), via MAST CAOM."""
    now = _mjd_now()
    obs: list[dict] = []
    for days in (60, 200, 600):
        obs = _mast({"service": "Mast.Caom.Filtered", "params": {
            "columns": "obsid,obs_id,t_min,instrument_name,target_name,dataRights",
            "filters": [{"paramName": "obs_collection", "values": ["HST"]},
                        {"paramName": "dataproduct_type", "values": ["image"]},
                        {"paramName": "dataRights", "values": ["PUBLIC"]},
                        {"paramName": "t_min", "values": [{"min": now - days, "max": now}]}]}})
        if len(obs) >= want * 2:
            break
    obs.sort(key=lambda r: -float(r.get("t_min") or 0))
    pairs: list[tuple[str, str]] = []
    for i in range(0, min(len(obs), want * 8), 50):
        ids = ",".join(str(o["obsid"]) for o in obs[i:i + 50])
        prods = _mast({"service": "Mast.Caom.Products", "params": {"obsid": ids}})
        for pr in prods:
            sub = str(pr.get("productSubGroupDescription") or "").upper()
            fn = str(pr.get("productFilename") or "")
            if sub in ("JIT", "JIF") and str(pr.get("dataRights", "PUBLIC")).upper() == "PUBLIC" and fn.endswith(".fits"):
                pairs.append((MAST_DL.format(uri=urllib.parse.quote(str(pr["dataURI"]), safe=":/")), fn))
        roots = {fn.rsplit("_", 1)[0] for _, fn in pairs}
        if len(roots) >= want:
            break
    by_root: dict[str, list[tuple[str, str]]] = {}
    for url, fn in pairs:
        by_root.setdefault(fn.rsplit("_", 1)[0], []).append((url, fn))
    complete = [v for v in by_root.values() if {f.rsplit("_", 1)[1] for _, f in v} >= {"jit.fits", "jif.fits"}]
    step(f"MAST lists {len(complete)} recent public Hubble exposures with jitter logs; downloading {min(want, len(complete))}")
    flat = [x for v in complete[:want] for x in sorted(v, key=lambda t: t[1])]
    _download_some(flat, dest, want * 2, "Hubble jitter logs")


def _download(mission: str, dest: Path, files: list[str]) -> Path:
    if mission == "opssat":
        from . import opssat
        return opssat.download(dest, "v2")
    if not files:
        (_webb_latest if mission == "webb" else _hubble_latest)(dest, AUTO_N[mission])
        return dest
    coll = "HST" if mission == "hubble" else "JWST"
    names = []
    for f in files:
        f = f.strip()
        if not f:
            continue
        if mission == "hubble" and not f.endswith(".fits"):
            names += [f"{f.lower()}_jit.fits", f"{f.lower()}_jif.fits"]
        else:
            names.append(f if f.endswith(".fits") else f + ".fits")
    for i, n in enumerate(names):
        report(i, len(names), f"downloading {n} from MAST")
        _fetch(MAST.format(coll=coll, name=urllib.parse.quote(n)), dest / n, label=n)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mission", choices=list(catalog.MISSIONS))
    p.add_argument("--source", help="folder with uploaded files")
    p.add_argument("--download", action="store_true")
    p.add_argument("--files", nargs="*", default=[])
    a = p.parse_args()
    meta = catalog.get(a.mission)
    try:
        if a.download:
            dest = SRC_ROOT / a.mission / "download"        # stable folder: re-runs reuse it, never pile up
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            step(f"1/2 Download · {meta['name']} from {meta['source']['title']}")
            got = _download(a.mission, dest, a.files)
            folder = got if got.is_dir() else got.parent
        elif a.source:
            folder = Path(a.source)
            if not folder.exists():
                raise ValueError(f"upload folder {folder} is gone - upload again")
            _extract_zips(folder)
        else:
            raise ValueError("choose Download or upload files")
        src = _find(folder, a.mission)
        step(f"2/2 Import · validating and publishing {meta['name']}")
        if a.mission == "opssat":
            from . import opssat
            man = opssat.prepare(src)
        else:
            from . import pointing
            man = pointing.prepare(a.mission, src)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError, OSError) as e:
        msg = str(e)
        if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
            msg = (f"source_unavailable: {meta['source']['title']} is not reachable from this machine ({e.reason}). "
                   "Check the internet connection and press Install again - nothing was changed.")
        fail(msg if msg.startswith("source_unavailable") else f"{meta['name']}: {msg}")
    print(f"Installed {meta['name']}: {man.get('scope')} · {man.get('items')} items", flush=True)
    if a.source:
        shutil.rmtree(a.source, ignore_errors=True)     # the prepared store keeps what is needed


if __name__ == "__main__":
    main()
