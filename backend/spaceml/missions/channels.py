"""Channel telemetry (ESA-AD missions, FUNcube-1, SatNOGS): rolling detection and, where events are
annotated (ESA-AD), bounded Isolation Forest training with chronological, event-level evaluation.

Leakage rules: robust scaling and the threshold are fitted on the training period only; windows are built
separately inside the training and the test period of each channel, so no window crosses the split; windows
that overlap annotated training events are left out of the (unsupervised) fit.
Scoring is event-level and labeled as a local baseline (not the official ESA-ADB time-aware metrics).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import detect, store

WIN, STEP = 60, 30            # 60 samples (30 min at 30 s), 50 % overlap


def _items(mission: str) -> list[dict]:
    return [i for i in store.items(mission)]


def _split_idx(t: np.ndarray, split_at: float) -> int:
    return int(np.searchsorted(t, split_at, "right"))


def _feats(t, y, a, b):
    spans = [(a + s0, a + s1) for s0, s1 in detect.windows(b - a, WIN, STEP)]
    X, _ = detect.window_features(t, {"v": y}, spans)
    return X, spans


def train(mission: str, out: Path, quick: bool = True, seed: int = 2137, n_jobs: int = 1, progress=None) -> dict:
    from sklearn.ensemble import IsolationForest
    import joblib
    from ..common.utils import write_json
    t0 = time.time()
    man = store.manifest(mission) or {}
    split_at = man.get("split_at")
    if split_at is None:
        raise ValueError("manifest has no split boundary - re-run prepare")
    evs = store.events(mission)
    its = _items(mission)
    Xtr, stats = [], {}
    for k, it in enumerate(its):
        d = store.load(mission, it["id"])
        t, y = d["t"], d["cols"]["value"].astype(np.float64)
        s = _split_idx(t, split_at)
        ytr = y[:s][np.isfinite(y[:s])]
        if len(ytr) < WIN * 2:
            continue
        med = float(np.median(ytr))
        iqr = float(np.subtract(*np.percentile(ytr, [75, 25]))) or float(np.std(ytr)) or 1.0
        stats[it["id"]] = (med, iqr)
        z = (y - med) / iqr
        X, spans = _feats(t, z, 0, s)
        ev = [e for e in evs if e["channel"] == it["id"] and e["category"] != "Communication Gap"]
        keep = [not any(t[a] <= e["end"] and t[b - 1] >= e["start"] for e in ev) for a, b in spans]
        Xtr.append(X[np.asarray(keep, dtype=bool)] if len(X) else X)
        progress and progress(k + 1, len(its) + 2, f"windows for {it['id']}")
    X = np.vstack([x for x in Xtr if len(x)]) if any(len(x) for x in Xtr) else np.zeros((0, 1))
    if len(X) < 50:
        raise ValueError(f"only {len(X)} nominal training windows before {man.get('split_at_utc')} - install a longer period")
    cap = 30000 if quick else 300000
    rng = np.random.default_rng(seed)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    iso = IsolationForest(n_estimators=100 if quick else 300, random_state=seed, n_jobs=n_jobs).fit(X)
    thr = float(np.quantile(-iso.score_samples(X), 0.999))
    model = {"iso": iso, "threshold": thr, "stats": stats}
    per, all_m, all_b, all_ev = {}, [], [], []
    for it in its:
        if it["id"] not in stats:
            continue
        d = store.load(mission, it["id"])
        r = score_item(mission, it, d, model, test_from=split_at)
        ev = [e for e in evs if e["channel"] == it["id"]]
        tt = d["t"][_split_idx(d["t"], split_at):]
        if not len(tt):
            continue
        per[it["id"]] = {"model": detect.event_scores(r["model_intervals"], ev, float(tt[0]), float(tt[-1])),
                         "baseline": detect.event_scores(r["baseline_intervals"], ev, float(tt[0]), float(tt[-1]))}
        all_m += r["model_intervals"]
        all_b += r["baseline_intervals"]
        all_ev += [e for e in ev if e["end"] >= tt[0]]
    progress and progress(len(its) + 1, len(its) + 2, "event-level scoring")

    def agg(key):
        tot = {"events": 0, "detected_events": 0, "alarms": 0, "false_alarms": 0}
        for v in per.values():
            for k in tot:
                tot[k] += v[key][k]
        tot["event_recall"] = tot["detected_events"] / tot["events"] if tot["events"] else None
        tot["alarm_precision"] = (tot["alarms"] - tot["false_alarms"]) / tot["alarms"] if tot["alarms"] else None
        return tot
    m, b = agg("model"), agg("baseline")
    preds = []
    for it in its:
        if it["id"] in stats:
            d = store.load(mission, it["id"])
            r = score_item(mission, it, d, model, test_from=split_at)
            preds.append({"item": it["id"], "model_intervals": r["model_intervals"][:200],
                          "baseline_intervals": r["baseline_intervals"][:200]})
    joblib.dump(model, out / "model.joblib")
    metrics = {"headline": {"metric": "event recall on the test period (local baseline scoring)",
                            "model": m["event_recall"], "baseline": b["event_recall"]},
               "eval_split": f"{man.get('split_protocol')}; test from {man.get('split_at_utc')}",
               "protocol": "local baseline: an annotated event counts as detected if any alarm overlaps it; "
                           "NOT the official ESA-ADB time-aware metrics",
               "model": {"name": f"IsolationForest ({iso.n_estimators} trees) on 30-min window features", **m,
                         "threshold": thr, "threshold_source": "99.9th percentile of nominal training-window scores; alarm needs >= 2 consecutive windows",
                         "windows_trained": int(len(X))},
               "baseline": {"name": "rolling median/MAD (61 samples, |z| > 6)", **b},
               "per_channel": per, "test_events": len(all_ev)}
    write_json(out / "metrics.json", metrics)
    write_json(out / "predictions.json", preds)
    write_json(out / "manifest.json", {"mission": mission, "model_name": metrics["model"]["name"],
                                       "run_kind": "quick" if quick else "full", "source": (man.get("source") or {}).get("name"),
                                       "dataset_fingerprint": man.get("fingerprint"), "split_at_utc": man.get("split_at_utc"),
                                       "trained_seconds": round(time.time() - t0, 2)})
    progress and progress(len(its) + 2, len(its) + 2, "done")
    return metrics


def score_item(mission: str, it: dict, d: dict, model: dict | None, test_from: float | None = None,
               column: str | None = None) -> dict:
    """Rolling MAD baseline (always) and the trained detector (when given) on one channel/field.
    With test_from, only the test period is scored."""
    t = d["t"]
    col = column or ("value" if "value" in d["cols"] else next(iter(d["cols"])))
    y = d["cols"][col].astype(np.float64)
    a = _split_idx(t, test_from) if test_from is not None else 0
    tt, yy = t[a:], y[a:]
    med_dt = float(np.median(np.diff(tt))) if len(tt) > 2 else 0.0
    _, flags = detect.rolling_mad(yy, window=61, k=6.0)
    base = detect.intervals(tt, flags, merge_gap=10 * med_dt, min_len=3)
    out = {"column": col, "baseline_intervals": base, "model_intervals": []}
    if model is None or it["id"] not in model.get("stats", {}):
        return out
    med, iqr = model["stats"][it["id"]]
    z = (y - med) / iqr
    X, spans = _feats(t, z, a, len(t))
    if not len(X):
        return out
    s = -model["iso"].score_samples(X)
    keep = detect.persistent(s > model["threshold"], 2)
    f = np.zeros(len(t), dtype=bool)
    for (p, q), k in zip(spans, keep):
        if k:
            f[p:q] = True
    ivs = detect.intervals(t[a:], f[a:], merge_gap=10 * med_dt)
    out["model_intervals"] = ivs
    return out
