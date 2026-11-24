"""Labeled multivariate time-series samples (one labeled window per record).

Typical source: the SWAN-SF solar-flare benchmark used by the IEEE BigData 2020 cup
(``train_partition*_data.json``). That file is JSON Lines; each line is

    {"<id>": {"label": "Q|A|B|C|M|X", "values": {"TOTUSJH": {"0": v, ..., "59": v}, ...}}}

i.e. 33 SHARP magnetic-field parameters of one active region, 60 steps at 12-minute cadence
(a 12-hour observation window), labeled with the largest GOES flare class in the following
24 hours (Q = flare-quiet). Records of the form {"id": .., "label": .., "values": {param: [..]}}
are accepted too, so other labeled-window datasets work the same way.

Analysis ``samples.classify`` turns each window into summary features (last, mean, std, min,
max, slope, change per parameter) and compares a random forest with the classic single-feature
threshold baseline, using the scores the flare-forecasting literature reports (TSS, HSS).
When the upload has several files (e.g. partition 1 and partition 2), the test set is always a
different file than the training set, because windows of one active region overlap in time and
a random split leaks information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .store import WorkbenchError, dataset_dir, get

LABEL_KEYS = ("label", "class", "classnum", "flare_class", "y", "target")
FLARE_ORDER = ["Q", "A", "B", "C", "M", "X"]
SHARP = {"TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "R_VALUE", "MEANPOT"}
MAX_SAMPLES = 200_000
MAX_CELLS = 250_000_000
FEATURES = ["last", "mean", "std", "min", "max", "slope", "change"]
PARAM_INFO = {
    "TOTUSJH": "total unsigned current helicity", "TOTBSQ": "total magnetic energy proxy (sum B²)",
    "TOTPOT": "total photospheric free-energy density", "TOTUSJZ": "total unsigned vertical current",
    "ABSNJZH": "absolute net current helicity", "SAVNCPP": "sum of net current per polarity",
    "USFLUX": "total unsigned magnetic flux", "R_VALUE": "flux near polarity-inversion lines",
    "MEANPOT": "mean free-energy density", "TOTFZ": "total vertical Lorentz force",
    "SHRGT45": "area with shear angle > 45°", "MEANSHR": "mean shear angle",
}


def clean_floats(o, nd: int = 6):
    """Recursively round floats and turn NaN/inf into None (JSON-safe)."""
    if isinstance(o, dict):
        return {k: clean_floats(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_floats(v, nd) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return round(f, nd) if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------ parsing
def _unwrap(rec: dict) -> tuple[str | None, dict] | None:
    """Return (id, body) when the record looks like a labeled multivariate window."""
    if not isinstance(rec, dict):
        return None
    if "values" in rec and isinstance(rec["values"], dict):
        rid = rec.get("id") or rec.get("ID") or rec.get("sample_id")
        return (str(rid) if rid is not None else None), rec
    if len(rec) == 1:
        (k, v), = rec.items()
        if isinstance(v, dict) and isinstance(v.get("values"), dict):
            return str(k), v
    return None


def _label(body: dict) -> str | None:
    for k, v in body.items():
        if k.lower() in LABEL_KEYS and v is not None and not isinstance(v, (dict, list)):
            return str(v).strip()
    return None


def _series(v) -> np.ndarray:
    if isinstance(v, dict):
        items = sorted(((int(k) if str(k).lstrip("-").isdigit() else i, x) for i, (k, x) in enumerate(v.items())))
        vals = [x for _, x in items]
    elif isinstance(v, list):
        vals = v
    else:
        raise ValueError("series must be an object {step: value} or a list")
    out = np.full(len(vals), np.nan, dtype=np.float32)
    for i, x in enumerate(vals):
        try:
            out[i] = float(x)
        except (TypeError, ValueError):
            pass
    return out


def _records(p: Path):
    """Yield (line_no, record) from a JSON Lines file or a JSON array / object of records."""
    with open(p, encoding="utf-8", errors="replace") as fh:
        head = fh.read(2048).lstrip()
    if head.startswith("["):
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        for i, r in enumerate(data, 1):
            yield i, r
        return
    with open(p, encoding="utf-8", errors="replace") as fh:
        first = True
        for i, line in enumerate(fh, 1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if first:  # a single pretty-printed JSON object holding all records
                    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                    for j, (k, v) in enumerate(obj.items(), 1):
                        yield j, {k: v}
                    return
                raise WorkbenchError(f"{p.name}, line {i}: not valid JSON")
            first = False
            if _unwrap(obj) is None and isinstance(obj, dict) and len(obj) > 1 and all(
                    isinstance(v, dict) and "values" in v for v in list(obj.values())[:5]):
                for k, v in obj.items():
                    yield i, {k: v}
            else:
                yield i, obj


def detect(files: list[Path]) -> list[Path]:
    """JSON/JSONL files whose first record is a labeled multivariate window."""
    hits = []
    for f in files:
        if f.suffix.lower() not in (".json", ".jsonl", ".ndjson"):
            continue
        try:
            for _, rec in _records(f):
                u = _unwrap(rec)
                if u and _label(u[1]) is not None and u[1]["values"]:
                    hits.append(f)
                break
        except Exception:  # noqa: BLE001 - not this format
            continue
    return hits


def create(files: list[Path], out: Path) -> dict:
    """Parse every record into X[n, params, steps] (+ labels, ids, source file) and save it."""
    params: list[str] | None = None
    xs, labels, ids, groups, skipped = [], [], [], [], 0
    group_names = []
    for g, f in enumerate(sorted(files)):
        group_names.append(f.name)
        for line, rec in _records(f):
            u = _unwrap(rec)
            if u is None:
                skipped += 1
                continue
            rid, body = u
            lab = _label(body)
            if lab is None:
                skipped += 1
                continue
            vals = body["values"]
            if params is None:
                params = [str(k) for k in vals.keys()]
                if not params:
                    raise WorkbenchError(f"{f.name}: the first record has no parameters")
            try:
                rows = [_series(vals.get(k, [])) for k in params]
            except ValueError as e:
                raise WorkbenchError(f"{f.name}, record {rid or line}: {e}") from e
            xs.append(rows)
            labels.append(lab)
            ids.append(rid or f"{f.stem}:{line}")
            groups.append(g)
            if len(xs) > MAX_SAMPLES:
                raise WorkbenchError(f"More than {MAX_SAMPLES:,} samples - split the upload")
    if not xs or params is None:
        raise WorkbenchError("No labeled samples found. Each record needs a label and a 'values' object "
                             "of parameter -> series.")
    n_steps = max(len(r) for x in xs for r in x)
    if len(xs) * len(params) * n_steps > MAX_CELLS:
        raise WorkbenchError("The upload is too large to analyse in one go - split it into smaller files")
    X = np.full((len(xs), len(params), n_steps), np.nan, dtype=np.float32)
    for i, x in enumerate(xs):
        for j, r in enumerate(x):
            X[i, j, :len(r)] = r
    labels_a = np.array(labels)
    np.savez_compressed(out / "samples.npz", X=X, labels=labels_a, ids=np.array(ids), groups=np.array(groups))
    classes = sorted(set(labels), key=lambda c: (FLARE_ORDER.index(c) if c in FLARE_ORDER else 99, c))
    counts = {c: int((labels_a == c).sum()) for c in classes}
    flare = set(classes) <= set(FLARE_ORDER) and (bool(SHARP & set(params)) or len(classes) >= 3)
    missing = float(np.isnan(X).mean())
    meta = {
        "samples": {
            "n": len(xs), "params": params, "n_steps": int(n_steps), "classes": classes, "class_counts": counts,
            "files": group_names, "missing_fraction": round(missing, 4), "skipped_records": skipped,
            "flare": flare,
            "context": ("SWAN-SF / IEEE BigData 2020 flare benchmark layout: SHARP parameters of one solar active "
                        "region over a 12-hour window (12-minute cadence); the label is the largest GOES flare "
                        "class in the next 24 hours (Q = quiet, then B < C < M < X).") if flare else None,
            "separation": _separation(X, labels_a, params, classes, flare),
        },
        "sha256": None,
        "summary": f"{len(xs):,} labeled samples · {len(params)} parameters × {n_steps} steps · "
                   + ", ".join(f"{c} {n}" for c, n in counts.items()),
    }
    return meta


def _positive(labels: np.ndarray, classes: list[str], flare: bool, target: str = "auto") -> tuple[np.ndarray, str]:
    if flare:
        ge_m = np.isin(labels, ["M", "X"])
        ge_c = np.isin(labels, ["C", "M", "X"])
        if target == "ge_M" or (target == "auto" and ge_m.sum() >= 10):
            return ge_m, "M- or X-class flare (≥M) vs everything else"
        if target in ("auto", "ge_C"):
            return ge_c, "C, M or X flare (≥C) vs B-class and quiet"
    counts = {c: int((labels == c).sum()) for c in classes}
    usable = {c: n for c, n in counts.items() if n >= 3} or counts
    pos = min(usable, key=usable.get) if target in ("auto", "minority") else target
    if pos not in counts:
        raise WorkbenchError(f"Unknown class {pos!r}")
    return labels == pos, f"class {pos} vs the rest"


def _separation(X, labels, params, classes, flare) -> list[dict]:
    """Rank parameters by how well their last value separates the positive class (Cohen's d)."""
    try:
        pos, _ = _positive(labels, classes, flare)
    except WorkbenchError:
        return []
    if pos.sum() < 2 or (~pos).sum() < 2:
        return []
    last = _last(X).astype(np.float64)
    out = []
    for j, name in enumerate(params):
        a, b = last[pos, j], last[~pos, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1.0
        out.append({"param": name, "d": round(float((a.mean() - b.mean()) / s), 3),
                    "pos_median": float(np.median(a)), "neg_median": float(np.median(b))})
    out.sort(key=lambda r: -abs(r["d"]))
    return clean_floats(out[:12], 6)


def _last(X: np.ndarray) -> np.ndarray:
    """Last finite value of every series -> (n, params)."""
    finite = np.isfinite(X)
    idx = X.shape[2] - 1 - np.argmax(finite[:, :, ::-1], axis=2)
    out = np.take_along_axis(X, idx[..., None], axis=2)[..., 0]
    out[~finite.any(axis=2)] = np.nan
    return out


# ------------------------------------------------------------------ features + preview
def features(X: np.ndarray, params: list[str]) -> tuple[np.ndarray, list[str]]:
    import warnings
    X = X.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = np.arange(X.shape[2], dtype=np.float64)
        mean = np.nanmean(X, axis=2)
        Xc = X - mean[..., None]
        tc = t - t.mean()
        w = np.isfinite(X)
        num = np.nansum(Xc * tc, axis=2)
        den = np.sum(np.where(w, tc ** 2, 0), axis=2)
        slope = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        first = np.take_along_axis(X, np.argmax(w, axis=2)[..., None], axis=2)[..., 0]
        last = _last(X)
        F = np.stack([last, mean, np.nanstd(X, axis=2), np.nanmin(X, axis=2), np.nanmax(X, axis=2),
                      slope, last - first], axis=2)          # (n, params, 7)
    names = [f"{p}:{f}" for p in params for f in FEATURES]
    return F.reshape(len(X), -1).astype(np.float64), names


def load(dataset_id: str) -> tuple[dict, dict]:
    meta = get(dataset_id)
    if meta.get("kind") != "samples":
        raise WorkbenchError("This dataset is not a labeled-sample dataset")
    z = np.load(dataset_dir(dataset_id) / "samples.npz", allow_pickle=False)
    return meta, {k: z[k] for k in z.files}


def class_profile(dataset_id: str, param: str) -> dict:
    """Median and inter-quartile band of one parameter over the window, per class (preview chart)."""
    meta, d = load(dataset_id)
    s = meta["samples"]
    if param not in s["params"]:
        raise WorkbenchError(f"Unknown parameter {param!r}")
    j = s["params"].index(param)
    out = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in s["classes"]:
            x = d["X"][d["labels"] == c, j, :].astype(np.float64)
            out.append({"class": c, "n": int(len(x)), "median": np.nanmedian(x, axis=0).tolist(),
                        "q25": np.nanpercentile(x, 25, axis=0).tolist(), "q75": np.nanpercentile(x, 75, axis=0).tolist()})
    return clean_floats({"param": param, "info": PARAM_INFO.get(param), "steps": s["n_steps"], "classes": out}, 6)


def sample_evidence(dataset_id: str, sample_id: str, params: list[str]) -> dict:
    """One labeled window next to the per-class medians (review evidence)."""
    meta, d = load(dataset_id)
    s = meta["samples"]
    idx = np.where(d["ids"] == sample_id)[0]
    if not len(idx):
        raise KeyError(f"sample {sample_id} not found")
    i = int(idx[0])
    params = [p for p in params if p in s["params"]] or [r["param"] for r in s.get("separation", [])[:3]] or s["params"][:3]
    steps = s["n_steps"]
    x = [round((k - steps + 1) * 12 / 60, 3) for k in range(steps)] if s.get("flare") else list(range(steps))
    panels = []
    import warnings
    for p in params[:3]:
        j = s["params"].index(p)
        lines = [{"name": f"this window ({d['labels'][i]})", "y": d["X"][i, j, :].astype(np.float64).tolist(), "bold": True}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in s["classes"]:
                sel = d["X"][d["labels"] == c, j, :].astype(np.float64)
                if len(sel):
                    lines.append({"name": f"median {c}", "y": np.nanmedian(sel, axis=0).tolist(), "dash": True})
        panels.append({"name": p + (f" - {PARAM_INFO[p]}" if p in PARAM_INFO else ""), "lines": lines})
    return clean_floats({"type": "series", "x_label": "hours before the end of the window" if s.get("flare") else "step",
                         "x": x, "panels": panels}, 6)


# ------------------------------------------------------------------ analysis
def _scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    pod = tp / (tp + fn) if tp + fn else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    n = tp + fp + fn + tn
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n if n else 0
    hss = ((tp + tn) - exp) / (n - exp) if n - exp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tss": round(pod - far, 4), "hss": round(hss, 4),
            "recall": round(pod, 4), "false_alarm_rate": round(far, 4), "precision": round(prec, 4),
            "f1": round(2 * prec * pod / (prec + pod), 4) if prec + pod else 0.0}


def _threshold_baseline(Ftr, ytr, Fte, names):
    """Best single feature + threshold on the training part (max TSS) - the classic flare baseline."""
    best = (-2.0, 0, 0.0, 1)
    for j in range(Ftr.shape[1]):
        if not names[j].endswith(":last"):
            continue
        x = Ftr[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 5:
            continue
        for q in np.unique(np.nanpercentile(x[ok], np.linspace(5, 95, 19))):
            for sign in (1, -1):
                pred = (sign * x > sign * q) & ok
                s = _scores(ytr[ok], pred[ok])["tss"]
                if s > best[0]:
                    best = (s, j, float(q), sign)
    _, j, q, sign = best
    x = Fte[:, j]
    return (sign * np.nan_to_num(x, nan=-np.inf * sign) > sign * q), names[j], q, sign


def run(dataset_id: str, p: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    meta, d = load(dataset_id)
    s = meta["samples"]
    X, labels, ids, groups = d["X"], d["labels"], d["ids"], d["groups"]
    y, target_desc = _positive(labels, s["classes"], s["flare"], p.get("target", "auto"))
    npos = int(y.sum())
    if npos < 3 or (~y).sum() < 3:
        raise WorkbenchError(f"Only {npos} positive sample(s) for '{target_desc}' - need at least 3 of each side. "
                             "Choose another target or upload more samples.")
    F, names = features(X, s["params"])
    med = np.nanmedian(F, axis=0)
    Fi = np.where(np.isfinite(F), F, np.where(np.isfinite(med), med, 0.0))

    ugroups = np.unique(groups)
    split_desc: str
    if len(ugroups) >= 2:
        folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in ugroups]
        split_desc = f"leave-one-file-out across {len(ugroups)} files (train on the others, test on each file)"
    else:
        k = int(max(2, min(int(p.get("folds", 5)), npos, int((~y).sum()))))
        folds = list(StratifiedKFold(k, shuffle=True, random_state=0).split(Fi, y))
        split_desc = (f"stratified {k}-fold cross-validation within one file - windows of the same active region "
                      "can overlap in time, so these scores are optimistic; upload a second partition for an "
                      "out-of-sample test")
    prob = np.full(len(y), np.nan)
    auto_thr = p.get("threshold_mode", "auto") == "auto"
    thr_fold = np.full(len(y), float(p.get("threshold", 0.5)))
    base_pred = np.zeros(len(y), dtype=bool)
    imp = np.zeros(F.shape[1])
    base_feats: list[str] = []
    for tr, te in folds:
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        clf = RandomForestClassifier(n_estimators=int(p.get("trees", 300)), min_samples_leaf=2, oob_score=True,
                                     class_weight="balanced_subsample", n_jobs=2, random_state=0)
        clf.fit(Fi[tr], y[tr])
        prob[te] = clf.predict_proba(Fi[te])[:, 1]
        if auto_thr:   # cut-off chosen on the training part only (out-of-bag votes), never on the test fold
            oob = np.nan_to_num(clf.oob_decision_function_[:, 1], nan=0.0)
            cands = np.unique(np.round(oob, 3))
            best = max(cands, key=lambda c: (_scores(y[tr], oob >= c)["tss"], -c)) if len(cands) else 0.5
            thr_fold[te] = best
        imp += clf.feature_importances_
        bp, bname, _, _ = _threshold_baseline(F[tr], y[tr], F[te], names)
        base_pred[te] = bp
        base_feats.append(bname)
    tested = np.isfinite(prob)
    pred = (prob >= thr_fold) & tested
    thr = float(np.median(thr_fold[tested])) if tested.any() else float(p.get("threshold", 0.5))
    m_scores = _scores(y[tested], pred[tested])
    b_scores = _scores(y[tested], base_pred[tested])
    clim = _scores(y[tested], np.zeros(tested.sum(), dtype=bool))
    try:
        auc = round(float(roc_auc_score(y[tested], prob[tested])), 4)
    except ValueError:
        auc = None
    imp = imp / max(1, len(folds))
    order = np.argsort(-imp)[:15]
    top_params = list(dict.fromkeys(names[i].split(":")[0] for i in order))[:4]
    base_feat = max(set(base_feats), key=base_feats.count) if base_feats else None

    rows = []
    for i in np.where(tested)[0]:
        rows.append({"id": str(ids[i]), "label": str(labels[i]), "positive": bool(y[i]), "prob": round(float(prob[i]), 4),
                     "predicted": bool(pred[i]), "file": s["files"][int(groups[i])],
                     "outcome": ("hit" if y[i] and pred[i] else "missed" if y[i] else "false alarm" if pred[i] else "correct quiet")})
    rows.sort(key=lambda r: -r["prob"])
    miss = [r for r in rows if r["outcome"] == "missed"]
    fa = [r for r in rows if r["outcome"] == "false alarm"]

    insights = []
    better = m_scores["tss"] - b_scores["tss"]
    insights.append({"kind": "warn" if better < -0.05 else "info",
                     "title": f"Model TSS {m_scores['tss']:.2f} vs single-threshold baseline {b_scores['tss']:.2f}",
                     "text": f"Target: {target_desc}. TSS = hit rate − false-alarm rate (0 = no skill, 1 = perfect); "
                               f"it does not depend on how rare flares are. Baseline: one threshold on "
                               f"{(base_feat or '?').replace(':', ' ')}."})
    insights.append({"kind": "info", "title": f"{npos} positive of {len(y)} samples ({npos / len(y):.0%})",
                     "text": f"Always predicting 'no flare' would be {1 - npos / len(y):.0%} accurate with TSS 0 - "
                               "which is why accuracy is not reported."})
    if top_params:
        insights.append({"kind": "info", "title": "Most informative parameters: " + ", ".join(top_params),
                         "text": "; ".join(f"{t}: {PARAM_INFO[t]}" for t in top_params if t in PARAM_INFO) or
                                   "Ranked by random-forest importance averaged over the folds."})
    if miss:
        insights.append({"kind": "warn", "title": f"{len(miss)} flare window(s) missed",
                         "text": "Lowest-scored: " + ", ".join(f"{r['id']} ({r['label']}, p={r['prob']:.2f})" for r in miss[-3:])
                                   + ". They are in Review Studio."})
    if fa:
        insights.append({"kind": "warn", "title": f"{len(fa)} false alarm(s)",
                         "text": "Highest-scored: " + ", ".join(f"{r['id']} ({r['label']}, p={r['prob']:.2f})" for r in fa[:3])})
    insights.append({"kind": "info", "title": "How it was tested", "text": split_desc})

    profiles = [class_profile(dataset_id, t) for t in top_params[:3]]
    return clean_floats({
        "viewer": "classification", "dataset_id": dataset_id, "target": target_desc, "split": split_desc,
        "n": int(len(y)), "n_tested": int(tested.sum()), "n_positive": npos, "threshold": thr,
        "threshold_mode": "chosen on training data (max TSS, out-of-bag)" if auto_thr else "fixed",
        "class_counts": s["class_counts"], "flare": s["flare"],
        "model": {"name": "Random forest on per-parameter window features", **m_scores, "auc": auc},
        "baseline": {"name": f"Single threshold on {(base_feat or '?').replace(':', ' ')}", **b_scores},
        "climatology": {"name": "Always 'no flare'", **clim},
        "importance": [{"feature": names[i], "importance": round(float(imp[i]), 5)} for i in order],
        "profiles": profiles, "samples": rows[:2000], "insights": insights,
    }, 6)
