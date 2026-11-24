"""Hubble (HST observation logs) and Webb (FGS POINTING tables): import, Isolation Forest training, scoring.

Hubble (per STScI's observation-log definition and the HST "exception report" notebook):
  *_jit.fits  one BINTABLE extension per exposure (header EXPNAME); 3-second averages with columns
              Seconds, SI_V2_AVG, SI_V3_AVG, SI_V2_RMS, SI_V3_RMS (+ P2P, RA/DEC/Roll, flags when present)
  *_jif.fits  matching extensions (same EXPNAME) with guiding keywords GUIDEACT, GSACQ, GSFAIL, NLOSSES,
              V2_RMS, V3_RMS; -32767 marks a missing / non-physical value
Webb (JWST "Fine Guide Stability" page):
  jw<pppppooovvv>_gs-<mode>_<yyyydddhhmmss>_(cal|uncal).fits with a POINTING table: Time [ms since start of
  the file], Jitter [mas], Delta DDC RA/Dec, Delta Aper PA, Delta V/J frame RA/Dec/PA [mas], HGA motion;
  POINTING header JITTRAVG/JITTRMIN/JITTRMAX. Segmented files of one visit are concatenated in time order.

Detectors: a rolling median/MAD baseline (no training) and one Isolation Forest per telescope, trained on
windows of training observations only (observations/visits are split BEFORE windowing - no leakage).
No verified fault labels exist, so results are review candidates; no classification metrics are published.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import detect, store

SENTINEL = -32767.0
HST_COLS = ["SI_V2_AVG", "SI_V3_AVG", "SI_V2_RMS", "SI_V3_RMS", "SI_V2_P2P", "SI_V3_P2P", "RA", "DEC", "Roll"]
HST_FLAG_COLS = ["Recenter", "TakeData", "SlewFlag", "DayNight", "FGS_flags"]
HST_GUIDING_KEYS = ["GUIDEACT", "GSACQ", "GSFAIL", "NLOSSES", "V2_RMS", "V3_RMS", "GSSEPRMS", "ACTGSSEP"]
HST_DETECT = ["SI_V2_AVG", "SI_V3_AVG", "SI_V2_RMS", "SI_V3_RMS"]

JW_CANON = {"time": "time", "jitter": "jitter", "deltaddcra": "delta_ddc_ra", "deltaddcdec": "delta_ddc_dec",
            "deltaaperpa": "delta_aper_pa", "deltavframera": "delta_v_frame_ra", "deltavframedec": "delta_v_frame_dec",
            "deltavframepa": "delta_v_frame_pa", "deltajframera": "delta_j_frame_ra", "deltajframedec": "delta_j_frame_dec",
            "deltajframepa": "delta_j_frame_pa", "hgamotion": "hga_motion"}
JW_UNITS = {"jitter": "mas", "delta_ddc_ra": "mas", "delta_ddc_dec": "mas", "delta_aper_pa": "mas",
            "delta_v_frame_ra": "mas", "delta_v_frame_dec": "mas", "delta_v_frame_pa": "mas", "delta_j_frame_ra": "mas",
            "delta_j_frame_dec": "mas", "delta_j_frame_pa": "mas", "hga_motion": "0=moving 1=finished 2=offline"}
JW_DETECT = ["jitter", "delta_ddc_ra", "delta_ddc_dec"]
JW_NAME = re.compile(r"jw(\d{5})(\d{3})(\d{3})_gs-([a-z0-9]+)_(\d{13})_(cal|uncal)\.fits$", re.I)


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", c.lower())


def _val(h, k):
    v = h.get(k)
    if isinstance(v, (int, float)) and v <= SENTINEL:
        return None
    return v if v not in ("", None) else None


# ------------------------------------------------------------------ Hubble
def _hst_files(folder: Path) -> list[tuple[Path, Path | None]]:
    jits = sorted(folder.rglob("*_jit.fits"))
    out = []
    for j in jits:
        jif = j.with_name(j.name.replace("_jit.fits", "_jif.fits"))
        out.append((j, jif if jif.exists() else None))
    return out


def prepare_hubble(folder: Path) -> dict:
    from astropy.io import fits
    files = _hst_files(folder)
    if not files:
        raise ValueError(f"no *_jit.fits files in {folder}")
    w = store.Writer("hubble")
    units: dict[str, str | None] = {}
    sums, skipped, unmatched = {}, [], 0
    t_min, t_max = None, None
    for jit, jif in files:
        sums[jit.name] = _file_sum(jit)
        if jif:
            sums[jif.name] = _file_sum(jif)
        jif_ext = {}
        if jif:
            with fits.open(jif) as hj:
                p0 = hj[0].header
                for h in hj[1:]:
                    if h.header.get("EXPNAME"):
                        jif_ext[str(h.header["EXPNAME"]).strip().upper()] = h.header
        else:
            p0 = {}
        with fits.open(jit) as hdul:
            prim = hdul[0].header
            for h in hdul[1:]:
                if not hasattr(h, "columns"):
                    continue
                exp = str(h.header.get("EXPNAME", "")).strip().upper() or f"{jit.stem}-{h.name}"
                names = list(h.columns.names)
                if h.data is None or not len(h.data):
                    skipped.append(f"{jit.name}[{exp}]: empty jitter table")
                    continue
                if "Seconds" not in names:
                    skipped.append(f"{jit.name}[{exp}]: no Seconds column")
                    continue
                data = h.data
                t = np.asarray(data["Seconds"], dtype=np.float64)
                cols, flags = {}, {}
                for c in HST_COLS:
                    if c in names:
                        v = np.asarray(data[c], dtype=np.float64)
                        v[v <= SENTINEL] = np.nan
                        cols[c] = v
                        units[c] = (h.columns[c].unit or "").strip() or ("arcsec" if c.startswith("SI_") else None)
                for c in HST_FLAG_COLS:
                    if c in names:
                        try:
                            flags[c] = np.asarray(data[c], dtype=np.float64)
                        except (TypeError, ValueError):
                            pass
                if not cols:
                    skipped.append(f"{jit.name}[{exp}]: no pointing columns")
                    continue
                jh = jif_ext.get(exp)
                if jh is None:
                    unmatched += 1
                guiding = {k: _val(jh, k) for k in HST_GUIDING_KEYS if jh is not None and k in jh}
                warnings = [f"{k}={guiding[k]}" for k in ("GSFAIL",) if guiding.get(k)] + \
                    ([f"NLOSSES={guiding['NLOSSES']}"] if guiding.get("NLOSSES") else []) + \
                    ([f"{k} missing (-32767)" for k in ("V2_RMS", "V3_RMS") if jh is not None and k in jh and guiding.get(k) is None])
                start = _hst_start(h.header, jh, prim)
                if start is not None:
                    t_min = start if t_min is None else min(t_min, start)
                    t_max = start + float(t[-1]) if t_max is None else max(t_max, start + float(t[-1]))
                w.add({"id": exp, "label": f"{exp} · {prim.get('TARGNAME') or p0.get('TARGNAME') or ''}".strip(" ·"),
                       "rootname": jit.name.split("_jit")[0], "proposal": prim.get("PROPOSID") or p0.get("PROPOSID"),
                       "target": prim.get("TARGNAME") or p0.get("TARGNAME"), "start_utc": _iso(start),
                       "guiding": guiding, "warnings": warnings, "jif_matched": jh is not None,
                       "cadence_s": float(np.median(np.diff(t))) if len(t) > 2 else None,
                       "units": {c: units.get(c) for c in cols}, "flag_columns": list(flags)},
                      t, {**cols, **{f"flag:{k}": v for k, v in flags.items()}})
    if not w.items:
        raise ValueError(f"no usable exposures ({'; '.join(skipped[:3])})")
    return w.publish({
        "source": {"name": "HST observation logs (MAST)", "version": "as downloaded", "files": sums,
                   "license": "public MAST data", "attribution": "Data from the Mikulski Archive for Space Telescopes (MAST), STScI."},
        "scope": f"{len(w.items)} exposures from {len(files)} jitter files", "time_is_epoch": False,
        "time_basis": "seconds since exposure start (jit 'Seconds', 3-second averages)",
        "date_range": [_iso(t_min), _iso(t_max)], "counts": {"exposures": len(w.items), "jit_files": len(files),
                                                            "jif_unmatched_exposures": unmatched, "skipped": len(skipped)},
        "skipped": skipped[:20], "channels": [{"id": c, "name": c, "unit": units.get(c)} for c in HST_COLS if c in units],
        "detect_columns": [c for c in HST_DETECT if c in units]})


def _hst_start(eh, jh, prim) -> float | None:
    for h in (eh, jh, prim):
        if h is None:
            continue
        if h.get("EXPSTART"):
            try:
                from astropy.time import Time
                return float(Time(float(h["EXPSTART"]), format="mjd").unix)
            except Exception:  # noqa: BLE001
                pass
        if h.get("DATE-OBS") and h.get("TIME-OBS"):
            try:
                return datetime.fromisoformat(f"{h['DATE-OBS']}T{h['TIME-OBS']}").replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    return None


# ------------------------------------------------------------------ Webb
def _jw_end(stamp: str) -> float:
    """yyyydddhhmmss (end of data in the file) -> unix seconds."""
    y, ddd, hh, mm, ss = int(stamp[:4]), int(stamp[4:7]), int(stamp[7:9]), int(stamp[9:11]), int(stamp[11:13])
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() + (ddd - 1) * 86400 + hh * 3600 + mm * 60 + ss


MJD_UNIX0 = 40587.0       # MJD of 1970-01-01


def jw_times(tms: np.ndarray, unit: str | None, end: float) -> tuple[np.ndarray, str]:
    """POINTING 'Time' column -> unix seconds. The unit is taken from TUNIT when present, otherwise
    inferred: values around 40000-100000 with sub-second steps are MJD days (the FGS samples at
    ~15.6 Hz, i.e. ~7e-7 d), otherwise the step size tells seconds from milliseconds. Relative
    clocks are anchored to the file's end-of-data stamp. Returns (times, basis)."""
    t = np.asarray(tms, dtype=np.float64)
    u = (unit or "").strip().lower()
    finite = t[np.isfinite(t)]
    step = float(np.median(np.diff(finite))) if len(finite) > 2 else 0.0
    looks_mjd = len(finite) > 0 and 40000 < float(np.median(finite)) < 100000 and 0 < step < 1e-3
    if u in ("d", "day", "days", "mjd") or (not u and looks_mjd):
        abs_t = (t - MJD_UNIX0) * 86400.0
        if len(finite) and abs(float(abs_t[np.isfinite(abs_t)][-1]) - end) < 2 * 86400:
            return abs_t, "MJD days"
        return end - (t[-1] - t) * 86400.0, "days (anchored to file stamp)"
    if u in ("s", "sec", "second", "seconds") or (not u and step >= 0.005 and step < 5):
        return end - (t[-1] - t), "seconds (anchored to file stamp)"
    return end - (t[-1] - t) / 1000.0, "milliseconds (anchored to file stamp)"


def prepare_webb(folder: Path) -> dict:
    from astropy.io import fits
    files = [p for p in sorted(folder.rglob("*.fits")) if JW_NAME.search(p.name)]
    if not files:
        raise ValueError(f"no JWST guider files (jw<pppppooovvv>_gs-<mode>_<yyyydddhhmmss>_cal.fits) in {folder}")
    by_visit: dict[tuple, list] = {}
    sums, skipped = {}, []
    for p in files:
        m = JW_NAME.search(p.name)
        prop, obs, visit, mode, stamp, level = m.groups()
        if level.lower() == "uncal" and p.with_name(p.name.replace("_uncal", "_cal")).exists():
            continue                     # the cal product carries the same POINTING table
        sums[p.name] = _file_sum(p)
        with fits.open(p) as hdul:
            ext = next((h for h in hdul[1:] if str(h.header.get("EXTNAME", "")).upper() == "POINTING"), None) or \
                next((h for h in hdul[1:] if hasattr(h, "columns") and {"time", "jitter"} <= {_norm(c) for c in h.columns.names}), None)
            if ext is None or ext.data is None or not len(ext.data):
                skipped.append(f"{p.name}: no populated POINTING table")
                continue
            names = {_norm(c): c for c in ext.columns.names}
            cols = {}
            for k, canon in JW_CANON.items():
                if k in names:
                    values = np.asarray(ext.data[names[k]])
                    if canon == "hga_motion" and values.dtype.kind in "SU":
                        cols[canon] = pd.to_numeric(values.astype(str), errors="coerce")
                    else:
                        cols[canon] = values.astype(np.float64)
            if "time" not in cols:
                skipped.append(f"{p.name}: POINTING table without Time")
                continue
            time_unit = ext.columns[names["time"]].unit
            hdr = {k: ext.header.get(k) for k in ("JITTRAVG", "JITTRMIN", "JITTRMAX") if k in ext.header}
            units = {canon: ((ext.columns[names[k]].unit or "").strip() or JW_UNITS.get(canon)) for k, canon in JW_CANON.items()
                     if k in names and canon != "time"}
        tms = cols.pop("time")
        end = _jw_end(stamp)
        t, basis = jw_times(tms, time_unit, end)
        if len(t) > 100 and float(t[-1] - t[0]) < 1.0:
            raise ValueError(f"{p.name}: {len(t)} samples collapse into {float(t[-1] - t[0]):.6f} s - the POINTING "
                             f"'Time' unit was not recognised ({basis}); nothing was installed")
        by_visit.setdefault((prop, obs, visit), []).append({"file": p.name, "mode": mode.lower(), "t": t, "cols": cols,
                                                            "hdr": hdr, "units": units, "end": end, "basis": basis})
    w = store.Writer("webb")
    t_min, t_max = None, None
    for (prop, obs, visit), segs in sorted(by_visit.items()):
        segs.sort(key=lambda s: s["end"])
        keys = sorted(set().union(*[s["cols"].keys() for s in segs]))
        t = np.concatenate([s["t"] for s in segs])
        cols = {k: np.concatenate([s["cols"].get(k, np.full(len(s["t"]), np.nan)) for s in segs]) for k in keys}
        vid = f"jw{prop}{obs}{visit}"
        cad = float(np.median(np.diff(t))) if len(t) > 2 else None
        t_min = float(t[0]) if t_min is None else min(t_min, float(t[0]))
        t_max = float(t[-1]) if t_max is None else max(t_max, float(t[-1]))
        w.add({"id": vid, "label": f"{vid} · program {int(prop)} obs {int(obs)} visit {int(visit)}", "program": int(prop),
               "observation": int(obs), "visit": int(visit), "guiding_modes": sorted({s["mode"] for s in segs}),
               "files": [s["file"] for s in segs], "segments": len(segs), "start_utc": _iso(float(t[0])), "end_utc": _iso(float(t[-1])),
               "duration_s": round(float(t[-1] - t[0]), 3), "time_unit": segs[0]["basis"],
               "cadence_s": cad, "pointing_header": [s["hdr"] for s in segs], "units": segs[0]["units"]}, t, cols)
    if not w.items:
        raise ValueError(f"no usable POINTING tables ({'; '.join(skipped[:3])})")
    all_units = {}
    for it in w.items:
        all_units.update(it.get("units") or {})
    return w.publish({
        "source": {"name": "JWST FGS guider products (MAST)", "version": "as downloaded", "files": sums,
                   "license": "public MAST data", "attribution": "Data from the Mikulski Archive for Space Telescopes (MAST), STScI."},
        "scope": f"{len(w.items)} visits from {len(sums)} guider files", "time_is_epoch": True,
        "time_basis": "UTC from the POINTING 'Time' column (MJD days when so stored; relative clocks anchored to each file's end-of-data stamp)",
        "date_range": [_iso(t_min), _iso(t_max)], "counts": {"visits": len(w.items), "files": len(sums), "skipped": len(skipped)},
        "skipped": skipped[:20], "channels": [{"id": k, "name": k, "unit": u} for k, u in sorted(all_units.items())],
        "detect_columns": [c for c in JW_DETECT if c in all_units]})


def prepare(mission: str, folder: Path) -> dict:
    return prepare_hubble(folder) if mission == "hubble" else prepare_webb(folder)


# ------------------------------------------------------------------ detection / training
def _win(mission: str, cadence: float | None) -> tuple[int, int]:
    if mission == "hubble":
        return 20, 10                       # 20 × 3 s = 1 minute windows, 30 s step
    c = cadence or 0.064
    size = int(max(16, min(1024, round(8.0 / c))))   # ~8 s of guider samples
    return size, max(4, size // 2)


def _detect_cols(mission: str) -> list[str]:
    man = store.manifest(mission) or {}
    return man.get("detect_columns") or (HST_DETECT if mission == "hubble" else JW_DETECT)


def item_windows(mission: str, it: dict, d: dict) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    size, step = _win(mission, it.get("cadence_s"))
    cols = {c: d["cols"][c] for c in _detect_cols(mission) if c in d["cols"]}
    spans = detect.windows(len(d["t"]), size, step)
    X, names = detect.window_features(d["t"], cols, spans)
    return X, spans, names


def baseline(mission: str, d: dict) -> list[dict]:
    """Rolling median/MAD on the pointing magnitude (no training)."""
    cols = [d["cols"][c] for c in _detect_cols(mission) if c in d["cols"]]
    if not cols:
        return []
    mag = np.sqrt(np.nansum(np.vstack([np.nan_to_num(c) ** 2 for c in cols[:2]]), axis=0))
    _, flags = detect.rolling_mad(mag, window=61, k=6.0)
    return detect.intervals(d["t"], flags, merge_gap=0, min_len=2)


def split_items(mission: str, seed: int = 2137) -> tuple[list[str], list[str]]:
    ids = sorted(str(i["id"]) for i in store.items(mission))
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(ids))
    n_test = max(1, round(len(ids) * 0.3)) if len(ids) > 1 else 0
    return sorted(perm[n_test:]), sorted(perm[:n_test])


def train(mission: str, out: Path, quick: bool = True, seed: int = 2137, n_jobs: int = 1, progress=None) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    import joblib
    from ..common.utils import write_json
    t0 = time.time()
    tr_ids, te_ids = split_items(mission, seed)
    if not tr_ids:
        raise ValueError("need at least 2 observations/visits to train and evaluate")
    if set(tr_ids) & set(te_ids):
        raise ValueError("observation split overlaps")
    progress and progress(0, 3, f"windowing {len(tr_ids)} training {'exposures' if mission == 'hubble' else 'visits'}")
    Xs = []
    for i in tr_ids:
        it, d = store.item(mission, i), store.load(mission, i)
        X, _, names = item_windows(mission, it, d)
        Xs.append(X)
    X = np.vstack([x for x in Xs if len(x)]) if any(len(x) for x in Xs) else np.zeros((0, 1))
    if len(X) < 20:
        raise ValueError(f"only {len(X)} training windows - not enough to fit a detector")
    cap = 20000 if quick else 200000
    rng = np.random.default_rng(seed)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    scaler = RobustScaler().fit(X)
    progress and progress(1, 3, f"fitting Isolation Forest on {len(X)} windows")
    iso = IsolationForest(n_estimators=100 if quick else 300, random_state=seed, n_jobs=n_jobs).fit(scaler.transform(X))
    train_scores = -iso.score_samples(scaler.transform(X))
    thr = float(np.quantile(train_scores, 0.995))      # threshold from training windows only
    progress and progress(2, 3, f"scoring {len(te_ids)} held-out {'exposures' if mission == 'hubble' else 'visits'}")
    preds, totals = [], {"model_alerts": 0, "baseline_alerts": 0, "overlapping_alerts": 0, "windows": 0}
    for i in te_ids:
        it, d = store.item(mission, i), store.load(mission, i)
        r = score_item(mission, it, d, {"iso": iso, "scaler": scaler, "threshold": thr})
        totals["model_alerts"] += len(r["model_intervals"])
        totals["baseline_alerts"] += len(r["baseline_intervals"])
        totals["overlapping_alerts"] += r["overlap"]
        totals["windows"] += r["windows"]
        preds.append({"item": i, **{k: r[k] for k in ("model_intervals", "baseline_intervals", "windows", "overlap")},
                      "context": {"guiding": it.get("guiding"), "warnings": it.get("warnings"),
                                  "guiding_modes": it.get("guiding_modes")}})
    joblib.dump({"iso": iso, "scaler": scaler, "threshold": thr, "features": names}, out / "model.joblib")
    metrics = {
        "headline": {"metric": "alerts on held-out observations (model vs rolling-MAD baseline)",
                     "model": totals["model_alerts"], "baseline": totals["baseline_alerts"]},
        "eval_split": f"{len(te_ids)} held-out {'exposures' if mission == 'hubble' else 'visits'} (split before windowing)",
        "train_items": tr_ids, "test_items": te_ids, "threshold": thr,
        "threshold_source": "99.5th percentile of training-window anomaly scores (single windows alert: transients are short)",
        "alerts": totals,
        "labels": "none verified - alerts are review candidates; no fault-classification metrics are published",
        "model": {"name": f"IsolationForest ({iso.n_estimators} trees) per telescope", "windows_trained": int(len(X))},
        "baseline": {"name": "rolling median/MAD (61 samples, |z| > 6) on pointing magnitude"}}
    write_json(out / "metrics.json", metrics)
    write_json(out / "predictions.json", preds)
    man = store.manifest(mission) or {}
    write_json(out / "manifest.json", {"mission": mission, "model_name": metrics["model"]["name"],
                                       "run_kind": "quick" if quick else "full", "source": man.get("source", {}).get("name"),
                                       "dataset_fingerprint": man.get("fingerprint"), "trained_seconds": round(time.time() - t0, 2)})
    progress and progress(3, 3, f"done: {totals['model_alerts']} model alerts, {totals['baseline_alerts']} baseline alerts")
    return metrics


def score_item(mission: str, it: dict, d: dict, model: dict | None) -> dict:
    base = baseline(mission, d)
    out = {"baseline_intervals": base, "model_intervals": [], "windows": 0, "overlap": 0, "scores": None}
    if model is None:
        return out
    X, spans, _ = item_windows(mission, it, d)
    out["windows"] = int(len(spans))
    if not len(X):
        return out
    s = -model["iso"].score_samples(model["scaler"].transform(X))
    keep = s > model["threshold"]           # single windows count: guider transients are short
    flags = np.zeros(len(d["t"]), dtype=bool)
    for (a, b), k in zip(spans, keep):
        if k:
            flags[a:b] = True
    ivs = detect.intervals(d["t"], flags)
    for iv in ivs:
        sel = [sc for (a, b), sc in zip(spans, s) if a <= iv["i1"] and b - 1 >= iv["i0"]]
        iv["score"] = round(float(max(sel)), 4) if sel else None
    out["model_intervals"] = ivs
    out["overlap"] = int(sum(any(m["start"] <= b["end"] and m["end"] >= b["start"] for b in base) for m in ivs))
    out["scores"] = {"threshold": model["threshold"], "max": float(s.max()), "windows_over": int((s > model["threshold"]).sum())}
    return out


# ------------------------------------------------------------------ utils
def _file_sum(p: Path) -> dict:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def _iso(x: float | None) -> str | None:
    return None if x is None else datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat()


def webb_time_problem(items: list[dict]) -> list[str] | None:
    """Data imported by releases before 3.1.4 read the POINTING 'Time' column (MJD days) as milliseconds,
    which squeezes each visit into a few microseconds. Returns the source file names to re-download,
    or None when the store is fine."""
    def collapsed(it: dict) -> bool:
        if (it.get("n") or 0) <= 100:
            return False
        cad = it.get("cadence_s")
        if cad is not None and float(cad) < 1e-3:          # FGS samples every ~64 ms
            return True
        return it.get("t0") is not None and it.get("t1") is not None and float(it["t1"]) - float(it["t0"]) < 1.0
    bad = [it for it in items if collapsed(it)]
    if not bad:
        return None
    return sorted({f for it in items for f in (it.get("files") or [])})
