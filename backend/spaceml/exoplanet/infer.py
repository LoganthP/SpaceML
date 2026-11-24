"""Exoplanet inference shared by Kepler catalog signals and uploaded light curves."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch

from ..common.utils import clean_floats, read_json
from . import data, system
from .model import TwoBranchCNN


def baseline_features(period: float, duration_hr: float, depth_ppm: float) -> np.ndarray:
    X = np.array([[period, duration_hr, depth_ppm]], dtype=float)
    return np.log10(np.clip(X, 1e-6, None))


@lru_cache(maxsize=4)
def _load(version_dir: str):
    vd = Path(version_dir)
    model = TwoBranchCNN()
    ck = torch.load(vd / "cnn.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.calibration = ck.get("calibration")  # Platt scaling fitted on validation stars (absent in old models)
    base = joblib.load(vd / "baseline_gb.joblib")
    split = read_json(vd / "split.json", {}) or {}
    where = {int(k): s for s in ("train", "val", "test") for k in split.get(s, [])}
    return model, base, where


def model_ready(version_dir: Path | None) -> bool:
    return bool(version_dir and (version_dir / "cnn.pt").exists() and (version_dir / "baseline_gb.joblib").exists())


def score(views: dict, period: float, duration_hr: float, depth_ppm: float, version_dir: Path | None):
    if not model_ready(version_dir):
        return None, None, {}
    model, base, where = _load(str(version_dir))
    with torch.no_grad():
        g = torch.from_numpy(np.nan_to_num(views["global"]).astype(np.float32))[None]
        l = torch.from_numpy(np.nan_to_num(views["local"]).astype(np.float32))[None]
        logit = float(model(g, l)[0])
    cal = getattr(model, "calibration", None) or {}
    s = float(1.0 / (1.0 + np.exp(-(cal.get("a", 1.0) * logit + cal.get("b", 0.0)))))
    b = float(base.predict_proba(baseline_features(period, duration_hr, depth_ppm))[0, 1])
    return s, b, where


def view_payload(views: dict, period: float) -> dict:
    half = float(views["local_half_width_days"])
    diag = {k[5:]: (None if not np.isfinite(float(v)) else float(v)) for k, v in views.items() if k.startswith("diag_")}
    return {
        "measured_depth_ppm": float(views["measured_depth_ppm"]),
        "n_points": int(views["n_points"]), "n_transits": int(views["n_transits"]),
        "diagnostics": diag or None,
        "global_view": {"x": clean_floats(np.linspace(-0.5, 0.5, data.GLOBAL_BINS), 5), "y": clean_floats(views["global"], 4)},
        "local_view": {"x": clean_floats(np.linspace(-half, half, data.LOCAL_BINS) * 24, 4), "y": clean_floats(views["local"], 4)},
        "scatter_global": {"x": clean_floats(views["scatter_global_phase"] / period, 5),
                           "y": clean_floats(views["scatter_global_flux"], 6)},
        "scatter_local": {"x": clean_floats(views["scatter_local_phase"] * 24, 4),
                          "y": clean_floats(views["scatter_local_flux"], 6)},
    }


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def kepler_system(kepid: int, selected: int, star_meta: dict) -> tuple[dict, dict | None]:
    df = data.tce_table()
    rows = df[df.kepid == kepid].sort_values("tce_period")
    first = rows.iloc[0]
    st = system.star(teff=_f(first.tce_steff) or star_meta.get("teff"),
                     radius=_f(first.tce_sradius) or star_meta.get("radius"),
                     logg=_f(first.tce_slogg) or star_meta.get("logg"))
    planets = []
    for r in rows.itertuples():
        pl = system.planet(float(r.tce_period), float(r.tce_depth), float(r.tce_duration), st,
                           sma_au=_f(r.tce_sma), dor=_f(r.tce_dor), ror=_f(r.tce_ror), impact=_f(r.tce_impact))
        planets.append({"id": f"TCE {int(r.tce_plnt_num)}", "label": r.av_training_set,
                        "selected": int(r.tce_plnt_num) == selected, **pl})
    ra = _f(first.ra) or star_meta.get("ra")
    dec = _f(first.dec) or star_meta.get("dec")
    sky = {"ra": ra, "dec": dec} if ra is not None and dec is not None else None
    return {"star": st, "planets": planets}, sky


def run(kepid: int, plnt: int, version_dir: Path | None) -> dict:
    row = data.tce_row(kepid, plnt)
    views = data.get_views(kepid, plnt)  # cached, or downloaded live from MAST
    star_meta = data.load_lightcurve(kepid)[2] if data.lc_path(kepid).exists() else {}
    period, dur_hr = float(row.tce_period), float(row.tce_duration)
    cnn, base, where = score(views, period, dur_hr, float(row.tce_depth), version_dir)
    sys_, sky = kepler_system(kepid, plnt, star_meta)
    return {
        "source": "kepler", "kepid": kepid, "tce_plnt_num": plnt,
        "target_name": f"KIC {kepid}", "name": data.FAMOUS.get(kepid),
        "candidate_score": cnn, "baseline_score": base, "model_available": cnn is not None,
        "catalog_label": row.av_training_set, "split": where.get(kepid, "unseen") if cnn is not None else "no model",
        "ephemeris_source": "Kepler DR24 catalog",
        "tce": {"period_days": period, "epoch": float(row.tce_time0bk), "epoch_system": "BKJD",
                "duration_hours": dur_hr, "depth_ppm": float(row.tce_depth), "model_snr": _f(row.tce_model_snr),
                "planet_radius_earth": _f(row.tce_prad)},
        **view_payload(views, period),
        "system": sys_, "sky": sky,
    }


def sky_catalog(limit: int = 20000) -> dict:
    df = data.tce_table()
    if df["ra"].isna().all():
        return {"available": False, "stars": []}
    stars = df.dropna(subset=["ra", "dec"]).sort_values("planet", ascending=False).drop_duplicates("kepid")
    if len(stars) > limit:
        stars = stars.sample(limit, random_state=0)
    cached = {k for k, _ in data.cached_view_ids()}
    lab = {"PC": 2, "AFP": 1, "NTP": 0, "UNK": 0}
    return {"available": True, "fields": ["kepid", "ra", "dec", "label", "cached"],
            "stars": [[int(r.kepid), round(float(r.ra), 4), round(float(r.dec), 4),
                       lab.get(r.av_training_set, 0), int(r.kepid in cached)] for r in stars.itertuples()]}

