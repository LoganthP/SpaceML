"""Dst forecasting with a chronological split, compared with a persistence baseline.

    python -m spaceml.spaceweather.forecast --train-years 2010-2016 --test-years 2017 --horizons 1 3 6 12 24

At issue time t the features use only data up to and including hour t (solar wind at L1,
IMF, past Dst/Kp and their recent statistics). The target is Dst(t + h). One gradient-boosting
model per horizon. Baseline: persistence, Dst(t + h) = Dst(t).
Metrics on the held-out test years: RMSE / MAE overall and during storms (true Dst <= -50 nT),
skill = 1 - RMSE_model / RMSE_persistence, and storm-onset detection (Dst crosses -50 nT).
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..common.progress import fail, report, step
from ..common.utils import new_version, now_iso, read_json, write_json
from ..config import MODELS_DIR, SEED
from . import data

STORM = -50.0
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]
INPUTS = ["V", "Bz", "By", "N", "Pdyn", "E", "B", "Dst", "Kp"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features at each hour (uses rows <= t only)."""
    x = df[INPUTS].interpolate(limit=3, limit_direction="forward")
    f = pd.DataFrame(index=df.index)
    for c in INPUTS:
        f[c] = x[c]
    for lag in (1, 2, 3, 6, 12, 24):
        f[f"Dst_l{lag}"] = x["Dst"].shift(lag)
    for lag in (1, 2, 3, 6):
        f[f"Bz_l{lag}"] = x["Bz"].shift(lag)
        f[f"V_l{lag}"] = x["V"].shift(lag)
    for w in (3, 6, 12, 24):
        f[f"Bz_mean{w}"] = x["Bz"].rolling(w, min_periods=1).mean()
        f[f"Bz_min{w}"] = x["Bz"].rolling(w, min_periods=1).min()
        f[f"E_mean{w}"] = x["E"].rolling(w, min_periods=1).mean()
        f[f"Pdyn_max{w}"] = x["Pdyn"].rolling(w, min_periods=1).max()
    f["dDst1"] = x["Dst"] - x["Dst"].shift(1)
    f["dDst3"] = x["Dst"] - x["Dst"].shift(3)
    f["VBs"] = x["V"] * np.clip(-x["Bz"], 0, None) / 1000.0  # coupling: speed x southward IMF
    f["VBs_mean6"] = f["VBs"].rolling(6, min_periods=1).mean()
    doy = df.index.dayofyear.to_numpy()
    f["season_sin"], f["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f


def target(df: pd.DataFrame, h: int) -> pd.Series:
    return df["Dst"].shift(-h)


def _years(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _metrics(y, p, base) -> dict:
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(base)
    y, p, base = y[ok], p[ok], base[ok]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    rb = float(np.sqrt(np.mean((base - y) ** 2)))
    st = y <= STORM
    out = {"n": int(len(y)), "rmse": rmse, "mae": float(np.mean(np.abs(p - y))),
           "rmse_persistence": rb, "mae_persistence": float(np.mean(np.abs(base - y))),
           "skill": 1 - rmse / rb if rb > 0 else 0.0, "corr": float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 else 0.0,
           "storm_hours": int(st.sum())}
    if st.sum() >= 5:
        out["rmse_storm"] = float(np.sqrt(np.mean((p[st] - y[st]) ** 2)))
        out["rmse_storm_persistence"] = float(np.sqrt(np.mean((base[st] - y[st]) ** 2)))
    return out


def _onsets(y: np.ndarray) -> np.ndarray:
    below = y <= STORM
    return np.flatnonzero(below[1:] & ~below[:-1]) + 1


def _detect(y, p, h) -> dict:
    """Was each storm onset forecast (forecast <= -50 nT within ±h hours of the true onset)?"""
    on = _onsets(np.nan_to_num(y, nan=0))
    hits = sum(bool(np.any(p[max(0, i - h):i + h + 1] <= STORM)) for i in on)
    fo = _onsets(np.nan_to_num(p, nan=0))
    false = sum(not np.any(y[max(0, i - 2 * h - 3):i + 2 * h + 4] <= STORM) for i in fo)
    return {"storm_onsets": int(len(on)), "forecast_hits": int(hits), "false_onsets": int(false)}


def train(train_years: list[int], test_years: list[int], horizons: list[int], version: str | None = None) -> dict:
    missing = [y for y in train_years + test_years if y not in data.cached_years()]
    if missing:
        fail(f"OMNI2 years not downloaded: {missing}. Press 'Train forecaster' on Data & Training again with "
             "internet access - it downloads the missing years first.")
    if set(train_years) & set(test_years):
        fail("Train and test years overlap - the split must be chronological and disjoint")
    t0 = time.time()
    tr_df, te_df = data.load(years=train_years), data.load(years=test_years)
    Ftr, Fte = features(tr_df), features(te_df)
    version = version or new_version()
    final = MODELS_DIR / "spaceweather" / version
    if final.exists():
        fail(f"Model version {version} already exists - choose another --version")
    out = MODELS_DIR / "spaceweather" / f".staging-{version}"   # published atomically at the end
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    step(f"Training {len(horizons)} horizon models on {train_years[0]}-{train_years[-1]} "
         f"({len(tr_df):,} h), testing on {test_years} ({len(te_df):,} h)")
    rows, per_h, preds = [], {}, {"t": (te_df.index.view("int64") // 10**9).astype(np.int64),
                                  "dst": te_df["Dst"].to_numpy(np.float32)}
    for i, h in enumerate(horizons, 1):
        ytr = target(tr_df, h)
        ok = ytr.notna() & Ftr["Dst"].notna()
        w = np.where(ytr[ok] <= STORM, 4.0, 1.0)  # storms are rare: weight them up
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=SEED)
        model.fit(Ftr[ok], ytr[ok], sample_weight=w)
        joblib.dump(model, out / f"dst_h{h}.joblib")
        yte = target(te_df, h).to_numpy()
        p = model.predict(Fte.fillna(np.nan))
        base = te_df["Dst"].to_numpy()
        m = _metrics(yte, p, base)
        m.update(_detect(yte, p, h))
        per_h[h] = m
        preds[f"h{h}"] = p.astype(np.float32)
        report(i, len(horizons), f"h={h} h: RMSE {m['rmse']:.1f} nT vs persistence {m['rmse_persistence']:.1f}")
        print(f"  +{h:2d} h  RMSE {m['rmse']:.2f}  persistence {m['rmse_persistence']:.2f}  skill {m['skill']:+.2f}", flush=True)
    np.savez_compressed(out / "test_predictions.npz", **preds)

    for h in horizons:
        m = per_h[h]
        metrics = [{"name": "RMSE (nT)", "model": m["rmse"], "baseline": m["rmse_persistence"], "lower_is_better": True},
                   {"name": "MAE (nT)", "model": m["mae"], "baseline": m["mae_persistence"], "lower_is_better": True}]
        if "rmse_storm" in m:
            metrics.append({"name": "Storm RMSE (nT)", "model": m["rmse_storm"], "baseline": m["rmse_storm_persistence"],
                            "lower_is_better": True})
        metrics.append({"name": "Storm onsets forecast", "model": f"{m['forecast_hits']}/{m['storm_onsets']}",
                        "baseline": f"{_persist_hits(te_df, h)}/{m['storm_onsets']}"})
        rows.append({"group": f"+{h} h ahead", "metrics": metrics})
    mid = horizons[len(horizons) // 2]
    write_json(out / "manifest.json", {
        "module": "spaceweather", "version": version, "created_at": now_iso(),
        "model_name": "Gradient boosting per horizon on causal L1 solar-wind + index features",
        "train_years": train_years, "test_years": test_years, "horizons": horizons, "inputs": INPUTS,
        "train_hours": int(len(tr_df)), "seconds": round(time.time() - t0, 1),
        "data": "NASA OMNIWeb OMNI2 hourly (SPDF)", "data_provenance": data.provenance(train_years + test_years),
        "environment": environment(), "seed": SEED})
    write_json(out / "metrics.json", {
        "module": "spaceweather", "version": version, "created_at": now_iso(),
        "model_name": "Gradient-boosting Dst forecaster", "baseline_name": "Persistence: Dst(t+h) = Dst(t)",
        "eval_split": f"Chronological: trained on {train_years[0]}-{train_years[-1]}, tested on {', '.join(map(str, test_years))}",
        "headline": {"metric": f"Skill vs persistence (+{mid} h)", "model": per_h[mid]["skill"], "baseline": 0.0},
        "rows": rows, "per_horizon": {str(k): v for k, v in per_h.items()},
        "struggles": [f"+{h} h: storm-time RMSE {per_h[h].get('rmse_storm', float('nan')):.1f} nT"
                      for h in horizons if "rmse_storm" in per_h[h]][:5],
    })
    out.replace(final)
    print(f"Saved {final}")
    return per_h


def environment() -> dict:
    import platform

    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__}


def _persist_hits(df: pd.DataFrame, h: int) -> int:
    y = target(df, h).to_numpy()
    return _detect(y, df["Dst"].to_numpy(), h)["forecast_hits"]


def latest_dir():
    root = MODELS_DIR / "spaceweather"
    dirs = [d for d in root.glob("*") if not d.name.startswith(".") and (d / "manifest.json").exists()] if root.exists() else []
    return max(dirs, key=lambda d: read_json(d / "manifest.json")["created_at"]) if dirs else None


def predict_window(version_dir, start: str, end: str, horizon: int) -> dict:
    """Forecasts issued every hour in [start, end] for `horizon` hours ahead (valid at t + h)."""
    man = read_json(version_dir / "manifest.json")
    if horizon not in man["horizons"]:
        raise ValueError(f"No model for +{horizon} h (available: {man['horizons']})")
    pad_start = (data.ts(start) - pd.Timedelta(hours=48)).isoformat()
    df = data.load(pad_start, (data.ts(end) + pd.Timedelta(hours=horizon)).isoformat())
    F = features(df)
    model = joblib.load(version_dir / f"dst_h{horizon}.joblib")
    p = pd.Series(model.predict(F), index=df.index)
    issued = df.index[(df.index >= data.ts(start)) & (df.index <= data.ts(end))]
    valid = issued + pd.Timedelta(hours=horizon)
    truth = df["Dst"].reindex(valid).to_numpy()
    in_train = sorted({t.year for t in issued} & set(man["train_years"]))
    return {"horizon": horizon, "issued": [t.isoformat() for t in issued], "valid": [t.isoformat() for t in valid],
            "forecast": [None if not np.isfinite(v) else round(float(v), 2) for v in p.reindex(issued)],
            "persistence": [None if not np.isfinite(v) else float(v) for v in df["Dst"].reindex(issued)],
            "truth": [None if not np.isfinite(v) else float(v) for v in truth],
            "in_sample_years": in_train, "version": version_dir.name}


def main() -> None:
    p = argparse.ArgumentParser(description="Train Dst forecasters (chronological split)")
    p.add_argument("--train-years", default="2010-2016")
    p.add_argument("--test-years", default="2017")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    p.add_argument("--version", default=None)
    p.add_argument("--if-missing", action="store_true", help="skip when a trained forecaster is already installed")
    a = p.parse_args()
    if a.if_missing and latest_dir() is not None:
        print(f"Forecaster already installed ({latest_dir().name}) - skipping training")
        return
    train(_years(a.train_years), _years(a.test_years), sorted(set(a.horizons)), a.version)


if __name__ == "__main__":
    main()
