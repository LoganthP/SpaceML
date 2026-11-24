"""Mission Replay: step through a real storm as if forecasting live.

For every hourly issuance time t in the event window the trained models forecast
Dst(t + h) from data up to and including hour t only (features are causal; the replay
also checks that no input row after t was used). The persistence baseline issues
Dst(t + h) = Dst(t). Each forecast is then scored against the observed Dst at t + h.

The result answers three questions on one screen:
  * What happened?               observed Dst + solar wind, timeline detected from the data
  * What could the model know?   forecasts as issued at each hour, inputs available at that hour
  * Where did the baseline win?  per-hour verdict model vs persistence, per-horizon scores
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..common.quality import series_quality
from ..common.utils import read_json
from . import data
from .forecast import INPUTS, STORM, features

LOOKBACK_H = 24          # longest lag / rolling window used by the features
TIE_NT = 2.0             # |error| differences below this count as a tie
THRESHOLDS = [(-50, "Moderate storm level (Dst ≤ −50 nT)"), (-100, "Intense storm level (Dst ≤ −100 nT)"),
              (-250, "Superstorm level (Dst ≤ −250 nT)")]
PLOT_VARS = ["Dst", "V", "Bz", "N", "Pdyn", "Kp"]


class InsufficientData(ValueError):
    pass


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _vals(s: pd.Series, digits: int = 2) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in s.to_numpy(float)]


def _first(mask: pd.Series):
    idx = mask[mask].index
    return idx[0] if len(idx) else None


def timeline(df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> list[dict]:
    """Key moments detected from the observations (not hand-entered)."""
    w = df[(df.index >= s) & (df.index <= e)]
    out: list[dict] = []
    if w.empty:
        return out
    dp = w["Pdyn"].diff()
    if dp.notna().any() and dp.max() >= 2.0:
        t = dp.idxmax()
        i = w.index.get_loc(t)
        p0, p1 = w["Pdyn"].iloc[max(i - 1, 0)], w["Pdyn"].iloc[i]
        v0 = w["V"].iloc[max(i - 2, 0):i].min()
        v1 = w["V"].iloc[i:i + 2].max()
        detail = f"Flow pressure {p0:.1f} → {p1:.1f} nPa"
        if np.isfinite(v0) and np.isfinite(v1):
            detail += f", speed {v0:.0f} → {v1:.0f} km/s"
        out.append({"time": _iso(t), "kind": "shock", "label": "Solar-wind shock / pressure jump reaches Earth",
                    "detail": detail + " (OMNI is time-shifted to the bow-shock nose)"})
    south = w["Bz"].rolling(3, min_periods=3).max() <= -10
    t = _first(south)
    if t is not None:
        t0 = t - pd.Timedelta(hours=2)
        out.append({"time": _iso(t0), "kind": "imf", "label": "Strong southward IMF begins (Bz ≤ −10 nT for 3 h)",
                    "detail": f"Min Bz in window {w['Bz'].min():.1f} nT"})
    dst = w["Dst"]
    for thr, label in THRESHOLDS:
        t = _first(dst <= thr)
        if t is not None:
            out.append({"time": _iso(t), "kind": f"dst{-thr}", "label": label, "detail": f"Dst {dst[t]:.0f} nT"})
    if dst.notna().any():
        tmin = dst.idxmin()
        out.append({"time": _iso(tmin), "kind": "min", "label": "Storm peak (minimum Dst)",
                    "detail": f"Dst {dst.min():.0f} nT, Kp max {w['Kp'].max():.1f}"})
        after = dst[dst.index > tmin]
        for thr in (-100, -50):
            if dst.min() <= thr:
                t = _first(after > thr)
                if t is not None:
                    out.append({"time": _iso(t), "kind": f"rec{-thr}", "label": f"Recovery above {thr} nT",
                                "detail": f"{(t - tmin).total_seconds() / 3600:.0f} h after the peak"})
    return sorted(out, key=lambda x: x["time"])


def _lead(issued: pd.DatetimeIndex, fc: np.ndarray, onset) -> dict | None:
    """Earliest issuance whose forecast reached storm level, relative to the observed onset."""
    if onset is None:
        return None
    hit = np.flatnonzero(np.nan_to_num(fc, nan=0.0) <= STORM)
    if not len(hit):
        return {"issued": None, "lead_hours": None}
    t = issued[hit[0]]
    return {"issued": _iso(t), "lead_hours": round((onset - t).total_seconds() / 3600, 1)}


def run(version_dir, start: str, end: str, horizons: list[int] | None = None) -> dict:
    man = read_json(version_dir / "manifest.json")
    hs = sorted(set(horizons or man["horizons"]) & set(man["horizons"]))
    if not hs:
        raise ValueError(f"No model for the requested horizons (available: {man['horizons']})")
    s, e = data.ts(start), data.ts(end)
    if e <= s:
        raise ValueError("End must be after start")
    if (e - s) > pd.Timedelta(days=21):
        raise ValueError("Replays are limited to 21 days")
    hmax = max(hs)
    years = sorted({s.year, e.year, (e + pd.Timedelta(hours=hmax)).year, (s - pd.Timedelta(hours=LOOKBACK_H + 24)).year})
    missing = [y for y in years if y not in data.cached_years()]
    if [y for y in missing if y in (s.year, e.year)]:
        raise FileNotFoundError(f"OMNI2 data for {missing} is not downloaded yet. "
                                f"Run: python -m spaceml.spaceweather.data download --years {' '.join(map(str, missing))}")
    df = data.load((s - pd.Timedelta(hours=LOOKBACK_H + 24)).isoformat(), (e + pd.Timedelta(hours=hmax)).isoformat())
    win = df[(df.index >= s) & (df.index <= e)]
    q = series_quality(win, ["Dst", "V", "Bz", "N", "Pdyn"], required=["Dst", "V", "Bz"], continuous={"V", "N", "Pdyn"})
    if q["status"] == "insufficient_data":
        raise InsufficientData("insufficient_data: " + "; ".join(q["findings"][:3]))

    F = features(df)
    issued = win.index
    persist = df["Dst"].reindex(issued).to_numpy(float)
    inputs_ok = win[["V", "Bz", "N"]].notna().all(axis=1).to_numpy()
    obs_dst = df["Dst"]
    onset = _first(obs_dst[(obs_dst.index >= s) & (obs_dst.index <= e + pd.Timedelta(hours=hmax))] <= STORM)

    # Leakage guard: the forecast issued at t must not change if every row after t is removed.
    probe_t = issued[len(issued) // 2]
    m0 = joblib.load(version_dir / f"dst_h{hs[0]}.joblib")
    full = float(m0.predict(F.loc[[probe_t]])[0])
    cut = features(df[df.index <= probe_t])
    trunc = float(m0.predict(cut.loc[[probe_t]])[0])
    causal_ok = abs(full - trunc) < 1e-6

    per_h, fc_out, verdict_out, truth_out = {}, {}, {}, {}
    for h in hs:
        model = m0 if h == hs[0] else joblib.load(version_dir / f"dst_h{h}.joblib")
        pred = pd.Series(model.predict(F), index=df.index).reindex(issued).to_numpy(float)
        truth = obs_dst.reindex(issued + pd.Timedelta(hours=h)).to_numpy(float)
        em, ep = np.abs(pred - truth), np.abs(persist - truth)
        ok = np.isfinite(em) & np.isfinite(ep)
        v = np.zeros(len(issued), dtype=float)
        v[ok & (em + TIE_NT < ep)] = 1
        v[ok & (ep + TIE_NT < em)] = -1
        v[~ok] = np.nan
        storm = ok & (truth <= STORM)
        rm = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2))) if ok.any() else None
        rp = float(np.sqrt(np.mean((persist[ok] - truth[ok]) ** 2))) if ok.any() else None
        rec = {
            "horizon": h, "n": int(ok.sum()), "rmse": rm, "rmse_persistence": rp,
            "skill": (1 - rm / rp) if rm is not None and rp else None,
            "mae": float(np.mean(em[ok])) if ok.any() else None,
            "mae_persistence": float(np.mean(ep[ok])) if ok.any() else None,
            "better_hours": int((v == 1).sum()), "worse_hours": int((v == -1).sum()), "tie_hours": int((v == 0).sum()),
            "storm_hours": int(storm.sum()),
            "storm_rmse": float(np.sqrt(np.mean((pred[storm] - truth[storm]) ** 2))) if storm.any() else None,
            "storm_rmse_persistence": float(np.sqrt(np.mean((persist[storm] - truth[storm]) ** 2))) if storm.any() else None,
            "forecast_min": float(np.nanmin(pred)) if np.isfinite(pred).any() else None,
            "observed_min": float(np.nanmin(truth)) if np.isfinite(truth).any() else None,
            "first_warning": _lead(issued, pred, onset),
            "persistence_first_warning": _lead(issued, persist, onset),
            "degraded_issuances": int((~inputs_ok).sum()),
        }
        if rec["forecast_min"] is not None and rec["observed_min"] is not None:
            rec["peak_error"] = rec["forecast_min"] - rec["observed_min"]
        per_h[str(h)] = rec
        fc_out[str(h)] = [None if not np.isfinite(x) else round(float(x), 1) for x in pred]
        truth_out[str(h)] = [None if not np.isfinite(x) else float(x) for x in truth]
        verdict_out[str(h)] = [None if not np.isfinite(x) else int(x) for x in v]

    shown = df[(df.index >= s - pd.Timedelta(hours=LOOKBACK_H)) & (df.index <= e + pd.Timedelta(hours=hmax))]
    in_sample = sorted({s.year, e.year} & set(man["train_years"]))
    return {
        "window": {"start": _iso(s), "end": _iso(e)},
        "horizons": hs, "lookback_hours": LOOKBACK_H, "tie_nt": TIE_NT, "storm_threshold": STORM,
        "issued": [_iso(t) for t in issued],
        "inputs_complete": [bool(x) for x in inputs_ok],
        "persistence": [None if not np.isfinite(x) else float(x) for x in persist],
        "forecast": fc_out, "truth": truth_out, "verdict": verdict_out, "per_horizon": per_h,
        "observed": {"time": [_iso(t) for t in shown.index], **{c: _vals(shown[c]) for c in PLOT_VARS}},
        "timeline": timeline(df, s, e + pd.Timedelta(hours=hmax)),
        "onset": _iso(onset) if onset is not None else None,
        "quality": q,
        "causality_check": {"passed": causal_ok, "probe_issued": _iso(probe_t),
                            "detail": "Forecast at the probe hour is identical with all later data removed"
                            if causal_ok else f"Forecast changed by {abs(full - trunc):.3g} nT when later data was removed"},
        "model": {"version": version_dir.name, "train_years": man["train_years"], "test_years": man["test_years"],
                  "inputs": man.get("inputs", INPUTS), "created_at": man.get("created_at")},
        "in_sample_years": in_sample,
        "data_provenance": data.provenance(sorted({t.year for t in shown.index})),
        "baseline": "Persistence: Dst(t + h) = Dst(t)",
    }
