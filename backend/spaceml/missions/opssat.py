"""ESA OPS-SAT (OPSSAT-AD): import, random-forest training and segment-level evaluation.

Source: OPSSAT-AD (Ruszczak et al.). dataset.csv = one row per labeled segment with 18 published features;
segments.csv = the raw samples. Verified join: dataset.segment == segments.segment (1:1); channel, anomaly,
train flag, sampling and sample count agree for every segment. The official split is the `train` column.

Model inputs: the 18 published features only. Excluded: segment (record id), anomaly (label), train (split
marker) and channel (an identifier; using it would let the model learn per-channel base rates).
Threshold and preprocessing are fitted on training segments only (out-of-bag probabilities).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import store

FEATURES = ["duration", "len", "mean", "var", "std", "kurtosis", "skew", "n_peaks", "smooth10_n_peaks",
            "smooth20_n_peaks", "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "gaps_squared",
            "len_weighted", "var_div_duration", "var_div_len"]
EXCLUDED = ["segment", "anomaly", "train", "channel"]
DATASET_COLS = ["segment", "anomaly", "train", "channel", "sampling", *FEATURES]
SEGMENT_COLS = ["channel", "timestamp", "value", "label", "sampling", "anomaly", "segment", "train"]

# Published checksums (Zenodo). v1 = record 12588359, v2 = record 15108715.
KNOWN = {
    "c12fb83863afa4463dec07303a20bb14": ("dataset.csv", "v1", "10.5281/zenodo.12588359"),
    "1f4b6ffd0359efac96a6698ff557e0f9": ("segments.csv", "v1", "10.5281/zenodo.12588359"),
    "5246fdc5e4630a4cecbf7fb6bc8b795e": ("dataset.csv", "v2", "10.5281/zenodo.15108715"),
    "72f109630abb933a386106897a631188": ("segments.csv", "v2", "10.5281/zenodo.15108715"),
}
URLS = {"v2": {"dataset.csv": "https://zenodo.org/records/15108715/files/dataset.csv?download=1",
               "segments.csv": "https://zenodo.org/records/15108715/files/segments.csv?download=1"}}


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(dest: Path, version: str = "v2") -> Path:
    """Fetch the published files from Zenodo (release preparation only; never at app startup)."""
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in URLS[version].items():
        out = dest / name
        if not out.exists():
            with urllib.request.urlopen(url, timeout=120) as r, open(out.with_suffix(".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            out.with_suffix(".part").replace(out)
    return dest


def validate_sources(folder: Path) -> dict:
    """Schema, checksum and join validation. Raises ValueError with the specific problem."""
    dp, sp = folder / "dataset.csv", folder / "segments.csv"
    for p in (dp, sp):
        if not p.exists():
            raise ValueError(f"{p} not found")
    sums = {p.name: {"md5": _md5(p), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in (dp, sp)}
    versions = {KNOWN.get(v["md5"], (None, None, None))[1] for v in sums.values()}
    version = versions.pop() if len(versions) == 1 else None
    if version is None:
        raise ValueError(f"checksums {[(k, v['md5']) for k, v in sums.items()]} match no published OPSSAT-AD release "
                         f"(known: {sorted({v[1] for v in KNOWN.values()})}); refusing to import unknown data")
    d = pd.read_csv(dp)
    s = pd.read_csv(sp)
    miss = [c for c in DATASET_COLS if c not in d.columns] + [c for c in SEGMENT_COLS if c not in s.columns]
    if miss:
        raise ValueError(f"missing columns {miss}")
    if not d.segment.is_unique:
        raise ValueError("dataset.csv: segment ids are not unique")
    g = s.groupby("segment").agg(channel=("channel", "first"), nch=("channel", "nunique"), anomaly=("anomaly", "first"),
                                 train=("train", "first"), sampling=("sampling", "first"), n=("value", "size"),
                                 label=("label", "first"), nlab=("label", "nunique"))
    bad = {"multi_channel_segments": int((g.nch > 1).sum()), "mixed_label_segments": int((g.nlab > 1).sum())}
    m = d.merge(g, left_on="segment", right_index=True, how="outer", suffixes=("_d", "_s"), indicator=True)
    bad["dataset_only"] = int((m._merge == "left_only").sum())
    bad["segments_only"] = int((m._merge == "right_only").sum())
    both = m[m._merge == "both"]
    for c in ("channel", "anomaly", "train", "sampling"):
        bad[f"{c}_mismatch"] = int((both[f"{c}_d"] != both[f"{c}_s"]).sum())
    bad["len_vs_samples_mismatch"] = int((both["len"] != both["n"]).sum())
    ts = pd.to_datetime(s.timestamp, utc=True, errors="coerce")
    bad["unparsable_timestamps"] = int(ts.isna().sum())
    bad["nan_values"] = int(s.value.isna().sum())
    if any(bad[k] for k in bad if k != "nan_values"):
        raise ValueError(f"OPS-SAT join/schema validation failed: { {k: v for k, v in bad.items() if v} }")
    return {"version": version, "doi": KNOWN[sums['dataset.csv']['md5']][2], "files": sums, "checks": bad,
            "dataset": d, "segments": s.assign(ts=ts), "seg_meta": g}


def prepare(folder: Path) -> dict:
    v = validate_sources(Path(folder))
    d, s, g = v["dataset"], v["segments"], v["seg_meta"]
    w = store.Writer("opssat")
    s = s.sort_values(["segment", "ts"]).reset_index(drop=True)
    epoch = (s.ts.astype("int64") // 10**9).to_numpy(np.float64)
    vals = s.value.to_numpy(np.float64)
    dd = d.set_index("segment")
    for seg, sub in s.groupby("segment", sort=True):
        row, pos = dd.loc[seg], sub.index.to_numpy()
        w.add({"id": str(int(seg)), "label": f"Segment {int(seg)} · {row.channel}", "channel": row.channel,
               "split": "train" if int(row.train) == 1 else "test", "anomaly": int(row.anomaly),
               "published_label": str(g.loc[seg, "label"]), "sampling_s": int(row.sampling),
               "units": {"value": None}},
              epoch[pos], {"value": vals[pos]}, label=np.full(len(pos), int(row.anomaly), np.int8))
    import shutil
    (Path(w.stage) / "source").mkdir(exist_ok=True)
    for name in ("dataset.csv", "segments.csv"):          # the published files, byte-for-byte
        shutil.copy2(Path(folder) / name, Path(w.stage) / "source" / name)
    chans = sorted(d.channel.unique())
    man = w.publish({
        "source": {"name": "OPSSAT-AD", "version": v["version"], "doi": v["doi"],
                   "files": {k: {kk: vv for kk, vv in f.items()} for k, f in v["files"].items()},
                   "license": "CC BY 4.0", "attribution": "Ruszczak B., Kotowski K., Evans D., Nalepa J. "
                   "OPSSAT-AD - anomaly detection dataset for satellite telemetry (Zenodo)."},
        "scope": "complete OPSSAT-AD release (all segments, all channels)",
        "time_basis": "UTC timestamp of each sample", "time_is_epoch": True,
        "date_range": [str(s.ts.min()), str(s.ts.max())],
        "counts": {"segments": int(len(d)), "samples": int(len(s)), "channels": len(chans),
                   "train_segments": int((d.train == 1).sum()), "test_segments": int((d.train == 0).sum()),
                   "train_anomalies": int(d[d.train == 1].anomaly.sum()), "test_anomalies": int(d[d.train == 0].anomaly.sum())},
        "channels": [{"id": c, "name": c, "unit": None, "segments": int((d.channel == c).sum())} for c in chans],
        "join": "dataset.segment = segments.segment (1:1); channel, anomaly, train, sampling and len verified equal",
        "validation": v["checks"], "features": FEATURES, "excluded_from_inputs": EXCLUDED,
        "source_copies": "source/dataset.csv and source/segments.csv are the published files (checksums above)"})
    return man


def features() -> pd.DataFrame:
    p = store.mdir("opssat") / "source" / "dataset.csv"
    if not p.exists():
        raise FileNotFoundError("OPS-SAT is not installed (source/dataset.csv missing)")
    return pd.read_csv(p)


# ------------------------------------------------------------------ training + evaluation
def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out = {"confusion_matrix": {"labels": ["nominal", "anomaly"], "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
           "per_class": {"nominal": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(sup[0])},
                         "anomaly": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(sup[1])}},
           "false_alarms": int(fp), "missed_anomalies": int(fn), "detected_anomalies": int(tp),
           "accuracy": float((tp + tn) / max(len(y), 1)), "f1_anomaly": float(f[1])}
    if prob is not None and len(set(y.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out


def train(version_dir: Path, quick: bool = True, seed: int = 2137, n_jobs: int = 1, progress=None) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    import joblib
    t0 = time.time()
    d = features()
    man = store.manifest("opssat") or {}
    tr, te = d[d.train == 1].reset_index(drop=True), d[d.train == 0].reset_index(drop=True)
    leak = set(tr.segment) & set(te.segment)
    if leak:
        raise ValueError(f"{len(leak)} segments in both splits")
    X, y = tr[FEATURES].to_numpy(np.float64), tr.anomaly.to_numpy(int)
    Xt, yt = te[FEATURES].to_numpy(np.float64), te.anomaly.to_numpy(int)
    if not np.isfinite(X).all() or not np.isfinite(Xt).all():
        raise ValueError("non-finite feature values")
    progress and progress(1, 4, "fitting random forest on the official training split")
    rf = RandomForestClassifier(n_estimators=150 if quick else 500, min_samples_leaf=1, class_weight="balanced",
                                oob_score=True, random_state=seed, n_jobs=n_jobs).fit(X, y)
    # threshold from out-of-bag probabilities of TRAINING segments only
    oob = rf.oob_decision_function_[:, 1]
    oob = np.nan_to_num(oob, nan=0.5)
    grid = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y, (oob >= g).astype(int), zero_division=0) for g in grid]
    thr = float(grid[int(np.argmax(f1s))])
    progress and progress(2, 4, f"threshold {thr:.2f} chosen on out-of-bag training probabilities")
    # simple baseline: one feature, one threshold, both chosen on training data only
    best = (0.0, None, None, None)
    for fi, name in enumerate(FEATURES):
        col = X[:, fi]
        for q in np.unique(np.quantile(col, np.linspace(0.02, 0.98, 49))):
            for sign in (1, -1):
                f = f1_score(y, (sign * col >= sign * q).astype(int), zero_division=0)
                if f > best[0]:
                    best = (f, name, float(q), sign)
    _, bname, bthr, bsign = best
    progress and progress(3, 4, "scoring the held-out test segments")
    prob = rf.predict_proba(Xt)[:, 1]
    pred = (prob >= thr).astype(int)
    bcol = te[bname].to_numpy(np.float64)
    bpred = (bsign * bcol >= bsign * bthr).astype(int)
    m_model, m_base = _metrics(yt, pred, prob), _metrics(yt, bpred, None)
    preds = [{"segment": int(s), "channel": c, "label": int(l), "prob": round(float(p), 4), "pred": int(pp),
              "baseline_pred": int(bp), "outcome": ("TP" if l and pp else "FN" if l else "FP" if pp else "TN")}
             for s, c, l, p, pp, bp in zip(te.segment, te.channel, yt, prob, pred, bpred)]
    per_channel = {}
    for c in sorted(te.channel.unique()):
        msk = (te.channel == c).to_numpy()
        per_channel[c] = {"segments": int(msk.sum()), "anomalies": int(yt[msk].sum()),
                          "false_alarms": int(((pred == 1) & (yt == 0) & msk).sum()),
                          "missed": int(((pred == 0) & (yt == 1) & msk).sum())}
    version_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": FEATURES, "threshold": thr}, version_dir / "model.joblib")
    metrics = {
        "headline": {"metric": "F1 (anomaly), held-out segments", "model": m_model["f1_anomaly"], "baseline": m_base["f1_anomaly"]},
        "eval_split": f"official OPSSAT-AD test split ({len(te)} segments, {int(yt.sum())} anomalies), scored per whole segment",
        "model": {"name": f"RandomForest ({rf.n_estimators} trees, class_weight=balanced)", "threshold": thr,
                  "threshold_source": "out-of-bag probabilities on training segments (max F1)", **m_model,
                  "oob_score": float(rf.oob_score_)},
        "baseline": {"name": f"single-feature threshold: {bname} {'>=' if bsign > 0 else '<='} {bthr:.4g} (chosen on training data)",
                     **m_base},
        "majority_class": {"accuracy": float((yt == 0).mean()), "f1_anomaly": 0.0},
        "per_channel": per_channel,
        "feature_importance": dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)), key=lambda kv: -kv[1])),
        "inputs": FEATURES, "excluded_from_inputs": EXCLUDED, "train_segments": int(len(tr)),
        "test_segments": int(len(te)), "split_overlap": 0, "seed": seed}
    from ..common.utils import write_json
    write_json(version_dir / "metrics.json", metrics)
    write_json(version_dir / "predictions.json", preds)
    write_json(version_dir / "manifest.json", {
        "mission": "opssat", "model_name": metrics["model"]["name"], "run_kind": "quick" if quick else "full",
        "source_version": (man.get("source") or {}).get("version"), "source_doi": (man.get("source") or {}).get("doi"),
        "dataset_fingerprint": man.get("fingerprint"), "trained_seconds": round(time.time() - t0, 2),
        "train_segments": int(len(tr)), "test_segments": int(len(te))})
    progress and progress(4, 4, f"done: F1 {m_model['f1_anomaly']:.3f} vs baseline {m_base['f1_anomaly']:.3f}")
    return metrics
