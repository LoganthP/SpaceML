"""Kepler DR24 TCE table + MAST light curves -> global/local folded views.

    python -m spaceml.exoplanet.data tce                        # TCE catalog (~20k signals)
    python -m spaceml.exoplanet.data lightcurves --n-stars 300  # download + cache folded views
    python -m spaceml.exoplanet.data status                     # what is cached

Each star is downloaded once from MAST (lightkurve), stitched and saved as a compact
.npz together with its stellar header (RA/Dec, Teff, radius, logg). Raw FITS files are
deleted afterwards. Downloads retry on network errors and the command is resumable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import urllib.parse
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.progress import fail, report, step
from ..common.utils import download, read_json, write_json
from ..config import EXOPLANET_DIR, SEED

TCE_PATH = EXOPLANET_DIR / "q1_q17_dr24_tce.csv"
LC_DIR = EXOPLANET_DIR / "lightcurves"
VIEW_DIR = EXOPLANET_DIR / "views"
FAILED_PATH = EXOPLANET_DIR / "failed_stars.json"

# Identification, ephemeris and label. Autovetter predictions (av_pred_class, av_pp_*)
# are never downloaded so they cannot leak into any model.
COLUMNS = ["kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration",
           "tce_depth", "tce_model_snr", "tce_prad", "av_training_set"]
# Optional geometry / stellar columns used only for the 3D views (never as model inputs).
EXTRA = ["ra", "dec", "tce_steff", "tce_sradius", "tce_slogg", "tce_sma", "tce_dor",
         "tce_ror", "tce_impact", "tce_eqt", "tce_insol"]
ARCHIVE = "https://exoplanetarchive.ipac.caltech.edu"


def _tce_urls(cols: list[str]) -> list[str]:
    q = f"select {','.join(cols)} from q1_q17_dr24_tce"
    return [f"{ARCHIVE}/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"}),
            f"{ARCHIVE}/cgi-bin/nstedAPI/nph-nstedAPI?"
            + urllib.parse.urlencode({"table": "q1_q17_dr24_tce", "select": ",".join(cols), "format": "csv"})]


LABELS = ("PC", "AFP", "NTP")
GLOBAL_BINS, LOCAL_BINS = 2001, 201
VIEW_VERSION = 2   # bump when the cached view format changes
KEPLER_90 = 11442793
FAMOUS = {KEPLER_90: "Kepler-90 (8-planet system, Kepler-90i found by a CNN)"}


# --------------------------------------------------------------------------- TCE table
def _looks_like_csv(path: Path) -> bool:
    head = path.read_text(errors="replace")[:400].lower()
    return "kepid" in head and "<html" not in head and "error" not in head.split("\n")[0]


def download_tce(csv: str | None = None, if_missing: bool = False) -> pd.DataFrame:
    EXOPLANET_DIR.mkdir(parents=True, exist_ok=True)
    if if_missing and TCE_PATH.exists() and catalog_info()["extended"]:
        print(f"Catalog already present: {TCE_PATH}")
        return tce_table()
    if csv:
        shutil.copy(csv, TCE_PATH)
    else:
        tmp = TCE_PATH.with_suffix(".download")
        ok = False
        for cols, label in ((COLUMNS + EXTRA, "with stellar/geometry columns"), (COLUMNS, "basic columns")):
            for url in _tce_urls(cols):
                for attempt in range(3):
                    try:
                        step(f"Downloading DR24 TCE table ({label}) from the NASA Exoplanet Archive")
                        download(url, tmp, timeout=180)
                        if not _looks_like_csv(tmp):
                            raise ValueError("archive returned an error page instead of CSV")
                        ok = True
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"  attempt {attempt + 1} failed: {e}")
                        if isinstance(e, ValueError):
                            break  # bad query (e.g. column missing) - try the next variant
                        time.sleep(3 * (attempt + 1))
                if ok:
                    break
            if ok:
                break
        if not ok:
            fail("source_unavailable: could not download the Kepler TCE table from the NASA Exoplanet Archive "
                 "(https://exoplanetarchive.ipac.caltech.edu). Check the internet connection and press Retry.")
        tmp.replace(TCE_PATH)
    load_tce.cache_clear()
    df = tce_table()
    info = catalog_info()
    print(f"TCE table: {len(df)} signals; labels: {df.av_training_set.value_counts().to_dict()}; "
          f"sky positions: {'yes' if info['has_sky'] else 'no'}")
    return df


@lru_cache(maxsize=2)
def load_tce(mtime: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(TCE_PATH, comment="#", low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TCE table is missing columns {missing}. Re-download it: "
                         "python -m spaceml.exoplanet.data tce")
    for c in EXTRA:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS + EXTRA].copy()
    if df["ra"].notna().any() and df["ra"].max() <= 24.0001:  # RA given in hours
        df["ra"] = df["ra"] * 15
    df["av_training_set"] = df["av_training_set"].fillna("UNK").astype(str).str.strip()
    df["tce_id"] = df.kepid.astype(str) + "-" + df.tce_plnt_num.astype(str)
    df["planet"] = (df.av_training_set == "PC").astype(int)
    return df


def tce_table(labeled_only: bool = False) -> pd.DataFrame:
    if not TCE_PATH.exists():
        raise FileNotFoundError("The Kepler TCE catalog is not installed. Press Download on the Kepler card in Data & Training. Run: python -m spaceml.exoplanet.data tce")
    df = load_tce(TCE_PATH.stat().st_mtime)
    return df[df.av_training_set.isin(LABELS)] if labeled_only else df


def catalog_info() -> dict:
    if not TCE_PATH.exists():
        return {"present": False, "extended": False, "has_sky": False}
    df = tce_table()
    return {"present": True, "has_sky": bool(df["ra"].notna().any()),
            "extended": bool(df["tce_steff"].notna().any() or df["ra"].notna().any())}


def tce_row(kepid: int, plnt: int) -> pd.Series:
    df = tce_table()
    r = df[(df.kepid == kepid) & (df.tce_plnt_num == plnt)]
    if r.empty:
        raise ValueError(f"No DR24 TCE for KIC {kepid} planet {plnt}")
    return r.iloc[0]


# --------------------------------------------------------------------------- light curves
def lc_path(kepid: int) -> Path:
    return LC_DIR / f"{int(kepid)}.npz"


def view_path(kepid: int, plnt: int) -> Path:
    return VIEW_DIR / f"{int(kepid)}_{int(plnt)}.npz"


def _save_npz(path: Path, **arrays) -> None:
    """Atomic write so concurrent runs never read a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


STAR_KEYS = {"RA_OBJ": "ra", "DEC_OBJ": "dec", "TEFF": "teff", "RADIUS": "radius", "LOGG": "logg",
             "KEPMAG": "kepmag", "FEH": "feh"}


def load_lightcurve(kepid: int) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(lc_path(kepid))
    star = {k: float(z[k]) for k in STAR_KEYS.values() if k in z.files and np.isfinite(z[k])}
    return z["time"], z["flux"], star


QUARTER_DIR = LC_DIR / "quarters"
NET_TIMEOUT = 60          # seconds per MAST request (astropy default is shorter and can hang on slow links)


def fetch_lightcurve(kepid: int, retries: int = 3, max_quarters: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Download Kepler long-cadence quarters for a star, stitch and cache it.

    Checkpointed per quarter: each quarter is downloaded into a persistent folder
    (lightcurves/quarters/<kepid>/), so a failed attempt or an app restart keeps the quarters
    that already arrived and only fetches the missing ones. Each quarter has its own bounded
    retries; one unavailable quarter does not fail the star as long as enough others arrive.
    ``max_quarters`` > 0 keeps that many quarters spread evenly over the mission (bulk
    downloads use 8). 0 downloads all quarters (when you open a single star in the dashboard).
    """
    if lc_path(kepid).exists():
        return load_lightcurve(kepid)
    import lightkurve as lk  # heavy import, only when needed
    try:
        from astropy.utils.data import conf as _aconf
        _aconf.remote_timeout = NET_TIMEOUT
    except Exception:  # noqa: BLE001
        pass

    search = None
    last: Exception | None = None
    for attempt in range(retries):
        try:
            search = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler", author="Kepler", cadence="long")
            break
        except Exception as e:  # noqa: BLE001 - MAST hiccup: bounded retry
            last = e
            time.sleep(2 + 4 * attempt)
    if search is None:
        raise RuntimeError(f"MAST search failed after {retries} attempts: {last}")
    if len(search) == 0:
        raise LookupError(f"MAST has no Kepler long-cadence light curves for KIC {kepid}")
    if max_quarters and len(search) > max_quarters:
        keep = sorted(set(np.linspace(0, len(search) - 1, max_quarters).round().astype(int).tolist()))
        search = search[keep]

    qdir = QUARTER_DIR / str(int(kepid))
    qdir.mkdir(parents=True, exist_ok=True)
    lcs, failed_q = [], 0
    for qi in range(len(search)):
        lc_q = None
        for attempt in range(retries):
            try:
                # lightkurve/astroquery reuse a complete file already in download_dir (= resume)
                lc_q = search[qi].download(download_dir=str(qdir), flux_column="pdcsap_flux")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1 + 3 * attempt)
        if lc_q is None:
            failed_q += 1
            print(f"  KIC {kepid}: quarter {qi + 1}/{len(search)} unavailable ({str(last)[:120]})", flush=True)
            continue
        lcs.append(lc_q)
    need = max(2, (len(search) + 1) // 2)
    if len(lcs) < need:
        raise RuntimeError(f"only {len(lcs)} of {len(search)} quarters arrived (need {need}); the ones that did are "
                           f"kept and the rest is retried next run. Last error: {last}")
    star = {}
    for key, name in STAR_KEYS.items():
        v = lcs[0].meta.get(key)
        try:
            if v is not None and np.isfinite(float(v)):
                star[name] = float(v)
        except (TypeError, ValueError):
            pass
    lc = lk.LightCurveCollection(lcs).stitch().remove_nans()  # stitch() normalises each quarter
    t = np.asarray(lc.time.value, dtype=np.float64)
    f = np.asarray(lc.flux.value, dtype=np.float32)
    if len(t) < 500:
        raise LookupError(f"only {len(t)} valid points")
    _save_npz(lc_path(kepid), time=t, flux=f, quarters=np.int64(len(lcs)), quarters_missing=np.int64(failed_q),
              **{k: np.float64(v) for k, v in star.items()})
    shutil.rmtree(qdir, ignore_errors=True)        # the stitched light curve replaces the raw quarters
    return t, f, star


def _binned(x: np.ndarray, y: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """Median of y in n equal bins over [lo, hi); empty bins are interpolated."""
    sel = (x >= lo) & (x < hi)
    x, y = x[sel], y[sel]
    out = np.full(n, np.nan)
    if len(x):
        idx = np.minimum(((x - lo) / (hi - lo) * n).astype(int), n - 1)
        order = np.argsort(idx, kind="stable")
        idx_s, y_s = idx[order], y[order]
        bounds = np.flatnonzero(np.diff(idx_s)) + 1
        for b, seg in zip(idx_s[np.r_[0, bounds]], np.split(y_s, bounds)):
            out[b] = np.median(seg)
    good = np.isfinite(out)
    if good.sum() == 0:
        return np.zeros(n)
    if (~good).any():
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out


def _normalise(v: np.ndarray) -> np.ndarray:
    v = v - np.median(v)
    m = abs(v.min())
    return v / m if m > 0 else v


def flatten(time: np.ndarray, flux: np.ndarray, period: float, t0: float, dur_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Remove stellar variability with transits masked. Window ~8.2 d whatever the cadence."""
    import lightkurve as lk

    cadence = float(np.median(np.diff(time))) if len(time) > 1 else 0.0204
    win = int(round(8.2 / max(cadence, 1e-4)))
    win = max(51, min(win, len(time) - 1 if len(time) % 2 == 0 else len(time) - 2))
    win += (win + 1) % 2  # must be odd
    lc = lk.LightCurve(time=time, flux=flux)
    mask = lc.create_transit_mask(period=period, transit_time=t0, duration=2 * dur_days)
    flat = lc.flatten(window_length=win, mask=mask)
    t, f = np.asarray(flat.time.value), np.asarray(flat.flux.value, dtype=np.float64)
    ok = np.isfinite(f)
    return t[ok], f[ok]


def diagnostics(phase: np.ndarray, epoch_no: np.ndarray, f: np.ndarray, period: float, dur: float) -> dict:
    """False-positive checks: odd/even depth difference and a secondary eclipse at phase 0.5."""
    out_tr = np.abs(phase) > dur
    base = np.median(f[out_tr]) if out_tr.any() else 1.0
    noise = float(np.std(f[out_tr])) if out_tr.sum() > 10 else float("nan")
    in_tr = np.abs(phase) < dur / 2

    def depth(sel):
        return float((base - np.median(f[sel])) * 1e6) if sel.sum() >= 3 else float("nan")

    odd, even = depth(in_tr & (epoch_no % 2 == 1)), depth(in_tr & (epoch_no % 2 == 0))
    n_odd, n_even = int((in_tr & (epoch_no % 2 == 1)).sum()), int((in_tr & (epoch_no % 2 == 0)).sum())
    sec_phase = np.abs(np.abs(phase) - period / 2) < dur / 2
    sec = depth(sec_phase)
    err = lambda n: noise / np.sqrt(max(n, 1)) * 1e6  # noqa: E731
    oe_sigma = abs(odd - even) / np.hypot(err(n_odd), err(n_even)) if np.isfinite(odd + even) else float("nan")
    sec_sigma = sec / err(int(sec_phase.sum())) if np.isfinite(sec) else float("nan")
    return {"odd_depth_ppm": odd, "even_depth_ppm": even, "odd_even_sigma": float(oe_sigma),
            "secondary_depth_ppm": sec, "secondary_sigma": float(sec_sigma), "noise_ppm": noise * 1e6}


def make_views(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration_hr: float) -> dict:
    dur = duration_hr / 24.0  # TCE table: duration in hours, period/epoch in days
    t, f = flatten(time, flux, period, t0, dur)
    phase = (t - t0 + period / 2) % period - period / 2  # transit at phase 0, in days
    epoch_no = np.floor((t - t0 + period / 2) / period).astype(int)
    glob = _normalise(_binned(phase, f, -period / 2, period / 2, GLOBAL_BINS))
    half = min(4 * dur, period / 2)
    loc = _normalise(_binned(phase, f, -half, half, LOCAL_BINS))
    in_tr = np.abs(phase) < dur / 2
    depth_ppm = float((1 - np.median(f[in_tr])) * 1e6) if in_tr.any() else 0.0
    diag = diagnostics(phase, epoch_no, f, period, dur)

    rng = np.random.default_rng(SEED)
    near = np.flatnonzero(np.abs(phase) < half)
    near = near if len(near) <= 3000 else np.sort(rng.choice(near, 3000, replace=False))
    every = np.arange(len(phase)) if len(phase) <= 5000 else np.sort(rng.choice(len(phase), 5000, replace=False))
    return {
        "global": glob.astype(np.float32), "local": loc.astype(np.float32),
        "measured_depth_ppm": depth_ppm, "n_points": len(t),
        "n_transits": int(len(np.unique(epoch_no[in_tr]))) if in_tr.any() else 0,
        "local_half_width_days": half, "view_version": VIEW_VERSION,
        **{f"diag_{k}": v for k, v in diag.items()},
        "scatter_local_phase": phase[near].astype(np.float32), "scatter_local_flux": f[near].astype(np.float32),
        "scatter_global_phase": phase[every].astype(np.float32), "scatter_global_flux": f[every].astype(np.float32),
    }


def _read_views(p: Path) -> dict:
    z = np.load(p)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def get_views(kepid: int, plnt: int, allow_download: bool = True) -> dict:
    p = view_path(kepid, plnt)
    if p.exists():
        v = _read_views(p)
        if v.get("view_version", 1) >= VIEW_VERSION or not lc_path(kepid).exists():
            return v
    elif not allow_download and not lc_path(kepid).exists():
        raise FileNotFoundError(f"Light curve for KIC {kepid} is not cached")
    row = tce_row(kepid, plnt)
    t, f, _ = fetch_lightcurve(kepid)
    v = make_views(t, f, float(row.tce_period), float(row.tce_time0bk), float(row.tce_duration))
    _save_npz(p, **v)
    return v


def cached_view_ids() -> set[tuple[int, int]]:
    if not VIEW_DIR.exists():
        return set()
    out = set()
    for p in VIEW_DIR.glob("*.npz"):
        try:
            a, b = p.stem.split("_")
            out.add((int(a), int(b)))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- bulk download
def _failed() -> dict:
    return read_json(FAILED_PATH, {}) or {}


def _star_job(kepid: int, tces: list[tuple[int, float, float, float]], max_quarters: int = 8) -> tuple[int, int]:
    """Runs in a worker process: fetch one star and build the views of all its TCEs."""
    t, f, _ = fetch_lightcurve(kepid, max_quarters=max_quarters)
    n = 0
    for plnt, period, t0, dur in tces:
        p = view_path(kepid, plnt)
        if p.exists() and _read_views(p).get("view_version", 1) >= VIEW_VERSION:
            n += 1
            continue
        try:
            _save_npz(p, **make_views(t, f, period, t0, dur))
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad TCE should not drop the star
            print(f"  KIC {kepid} TCE {plnt}: view failed ({e})", flush=True)
    return kepid, n


def choose_stars(n_stars: int, include: list[int], balance: bool) -> list[int]:
    """Random stars; with balance, half are stars that host a planet candidate so that a
    few hundred downloads already give both classes enough examples to train on."""
    df = tce_table(labeled_only=True)
    rng = np.random.default_rng(SEED)
    pc_stars = df[df.planet == 1].kepid.unique()
    other = np.setdiff1d(df.kepid.unique(), pc_stars)
    if balance:
        a = rng.permutation(pc_stars)[: n_stars // 2]
        b = rng.permutation(other)[: n_stars - len(a)]
        chosen = rng.permutation(np.concatenate([a, b])).tolist()
    else:
        chosen = rng.permutation(df.kepid.unique()).tolist()[:n_stars]
    already = {k for k, _ in cached_view_ids()}
    # stars cached earlier always stay in the set so re-runs only add new ones
    return list(dict.fromkeys(include + sorted(already) + chosen))


def _star_job_tuple(args: tuple) -> tuple[int, int]:
    return _star_job(*args)


def download_lightcurves(n_stars: int, workers: int, include: list[int], balance: bool = True,
                         max_quarters: int = 8) -> None:
    df = tce_table(labeled_only=True)
    failed = _failed()
    chosen = choose_stars(n_stars, include, balance)
    # permanent failures (no data at MAST) are skipped; network failures are retried
    todo = [k for k in chosen if failed.get(str(k), {}).get("kind") != "no_data"]
    groups = {k: [(int(r.tce_plnt_num), float(r.tce_period), float(r.tce_time0bk), float(r.tce_duration))
                  for r in g.itertuples()] for k, g in df[df.kepid.isin(todo)].groupby("kepid")}
    todo = [k for k in todo if k in groups]
    step(f"Caching light curves for {len(todo)} stars with {workers} parallel downloads, "
         f"{max_quarters or 'all'} quarters per star (re-running resumes where it stopped)")
    st = {"ok": 0, "bad": 0, "views": 0, "i": 0}

    def got(k, res) -> None:
        st["ok"] += 1
        st["views"] += res[1]
        failed.pop(str(k), None)
        tick()

    def err(k, e) -> None:
        st["bad"] += 1
        kind = "no_data" if isinstance(e, LookupError) else "network"
        failed[str(int(k))] = {"kind": kind, "error": str(e)[:300]}
        print(f"  KIC {k} skipped ({kind}): {str(e)[:160]}", flush=True)
        tick()

    def tick() -> None:
        st["i"] += 1
        report(st["i"], len(todo), f"{st['ok']} stars cached, {st['bad']} skipped")
        if st["i"] % 10 == 0:
            write_json(FAILED_PATH, failed)

    from ..common import parallel
    from spaceml.exoplanet.data import _star_job_tuple  # importable by name for spawned workers
    left = parallel.run(_star_job_tuple, [(int(k), groups[k], max_quarters) for k in todo], workers,
                        on_result=lambda it, r: got(it[0], r), on_error=lambda it, e: err(it[0], e),
                        stall_seconds=6 * 60)
    for it in left:   # the pool stalled on these (hung connections): mark for a later retry
        err(it[0], TimeoutError("MAST download stalled; will be retried on the next run"))
    ok, bad = st["ok"], st["bad"]
    write_json(FAILED_PATH, failed)
    total = cached_view_ids()
    print(f"Done: {ok} stars ok, {bad} skipped. Cached views: {len(total)} TCEs on "
          f"{len({k for k, _ in total})} stars in {VIEW_DIR}")
    if ok == 0 and todo:
        fail("No light curve could be downloaded. Check that https://mast.stsci.edu is reachable "
             "(firewall/proxy) and try again with fewer workers: --workers 2")


MIN_PER_CLASS = 15


def cached_training_rows():
    catalog = tce_table(labeled_only=True)
    cached = cached_view_ids()
    return catalog[[(int(k), int(p)) in cached for k, p in zip(catalog.kepid, catalog.tce_plnt_num)]].reset_index(drop=True)


def status() -> dict:
    cached = cached_view_ids()
    info = {"catalog": catalog_info(), "cached_views": len(cached), "cached_stars": len({k for k, _ in cached})}
    if TCE_PATH.exists():
        df = tce_table(labeled_only=True)
        keys = set(zip(df.kepid, df.tce_plnt_num))
        lab = df.set_index(["kepid", "tce_plnt_num"]).planet
        usable = [c for c in cached if c in keys]
        info["labeled_cached"] = len(usable)
        info["planets_cached"] = int(sum(lab.get(c, 0) for c in usable))
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Kepler DR24 data")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tce", help="download the DR24 TCE catalog")
    t.add_argument("--csv", help="use a manually exported CSV instead of downloading")
    t.add_argument("--if-missing", action="store_true", help="skip if an up-to-date catalog exists")
    l = sub.add_parser("lightcurves", help="download light curves and build views")
    l.add_argument("--n-stars", type=int, default=300)
    l.add_argument("--workers", type=int, default=6, help="parallel downloads (network-bound, up to 12)")
    l.add_argument("--max-quarters", type=int, default=8, help="quarters per star, spread evenly (0 = all 17)")
    l.add_argument("--no-balance", action="store_true", help="pick stars uniformly at random")
    sub.add_parser("status", help="show what is cached")
    args = p.parse_args()
    if args.cmd == "tce":
        download_tce(args.csv, args.if_missing)
    elif args.cmd == "status":
        print(status())
    else:
        if not TCE_PATH.exists():
            download_tce()
        download_lightcurves(args.n_stars, max(1, min(args.workers, 12)),
                             include=[KEPLER_90], balance=not args.no_balance, max_quarters=max(0, args.max_quarters))


if __name__ == "__main__":
    main()
