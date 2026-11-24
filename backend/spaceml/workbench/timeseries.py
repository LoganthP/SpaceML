"""Anomaly flags for any uploaded multichannel time series.

Two detectors run on the same mapped columns:

* **z-score baseline** - mean/std from the first ``train_fraction`` of the series (assumed
  nominal, chronological split), flag where |x - mean| > k * std.
* **rolling robust detector** - deviation from a centred rolling median, scaled by the
  rolling MAD (median absolute deviation) of the residuals. Adapts to slow drifts and level changes that a
  global z-score treats as anomalies.

Flags from all channels are merged into intervals (gaps shorter than ``merge_gap`` steps are
bridged; intervals shorter than ``min_length`` dropped). If the user maps a 0/1 label column
or types labeled intervals, event-level precision/recall is reported for both detectors.
Everything is chronological and nothing looks at the labels while flagging.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..common.utils import clean_floats
from ..telemetry.detection import prf, score_events, to_sequences
from .store import WorkbenchError, load_table

MAX_POINTS = 4000


def _intervals(flags: np.ndarray, merge_gap: int, min_length: int) -> list[list[int]]:
    seqs = to_sequences(flags)
    merged: list[list[int]] = []
    for a, b in seqs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [s for s in merged if s[1] - s[0] + 1 >= min_length]


def _fill(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return s.interpolate(limit_direction="both").to_numpy() if s.notna().any() else np.zeros(len(x))


def zscore_scores(x: np.ndarray, n_train: int) -> tuple[np.ndarray, dict]:
    ref = x[:n_train]
    ref = ref[np.isfinite(ref)]
    mu, sd = (float(ref.mean()), float(ref.std())) if ref.size else (0.0, 1.0)
    sd = sd if sd > 1e-12 else 1e-12
    return np.abs(x - mu) / sd, {"mean": mu, "std": sd}


def robust_scores(x: np.ndarray, window: int) -> np.ndarray:
    """|x - centred rolling median| / (1.4826 * rolling median absolute residual).

    The residual scale uses a 5x wider window so a short anomaly cannot inflate its own
    noise estimate. Centred windows are fine here: this is offline analysis of a recording.
    """
    s = pd.Series(x)
    mp = max(5, window // 4)
    med = s.rolling(window, center=True, min_periods=mp).median()
    resid = s - med
    scale = 1.4826 * resid.abs().rolling(window * 5, center=True, min_periods=mp).median()
    g = np.nanmedian(scale.to_numpy())
    g = g if np.isfinite(g) and g > 0 else 1e-9
    scale = np.maximum(scale.fillna(g).to_numpy(), max(g * 0.1, 1e-12))
    return np.nan_to_num(np.abs(resid.to_numpy()) / scale)


def _decimate(n: int) -> np.ndarray:
    """Indices that keep min/max of each bucket, so spikes survive downsampling."""
    if n <= MAX_POINTS:
        return np.arange(n)
    edges = np.linspace(0, n, MAX_POINTS // 2 + 1).astype(int)
    return np.unique(np.concatenate([edges[:-1], np.minimum(edges[1:] - 1, n - 1)]))


def parse_intervals(text: str, n: int, x: np.ndarray, is_time: bool) -> list[list[int]]:
    """'100-250, 900-1000' (steps or x values) -> index intervals."""
    out = []
    for part in re.split(r"[;,\n]+", text or ""):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(.+?)\s*(?:-|to|–|\.\.)\s*(.+)", part)
        if not m:
            raise WorkbenchError(f"Cannot read labeled interval {part!r}; use start-end")
        a, b = m.group(1).strip(), m.group(2).strip()
        try:
            if is_time and not re.fullmatch(r"[\d.]+", a):
                va = pd.Timestamp(a, tz="UTC").timestamp()
                vb = pd.Timestamp(b, tz="UTC").timestamp()
            else:
                va, vb = float(a), float(b)
        except Exception as e:  # noqa: BLE001
            raise WorkbenchError(f"Cannot read labeled interval {part!r}: {e}") from e
        ia, ib = (int(np.searchsorted(x, va)), int(np.searchsorted(x, vb, side="right") - 1)) \
            if x is not None else (int(va), int(vb))
        ia, ib = max(0, min(ia, n - 1)), max(0, min(ib, n - 1))
        if ib >= ia:
            out.append([ia, ib])
    return out


def run(dataset_id: str, p: dict) -> dict:
    entry, arrays = load_table(dataset_id, p["table"])
    sensors = p["sensor_cols"]
    if not sensors:
        raise WorkbenchError("Select at least one sensor column")
    cols = {c["name"]: c for c in entry["columns"]}
    for c in sensors + [p.get("time_col"), p.get("label_col")]:
        if c and c not in cols:
            raise WorkbenchError(f"Column {c!r} not found")
    n = entry["rows"]

    # x axis: time column (sorted), else step index
    x, x_kind = np.arange(n, dtype=np.float64), "step"
    order = np.arange(n)
    if p.get("time_col"):
        key = ("d::" if cols[p["time_col"]]["type"] == "datetime" else "n::") + p["time_col"]
        if key not in arrays:
            raise WorkbenchError(f"Time column {p['time_col']!r} is not numeric or a date")
        tx = arrays[key]
        if np.isfinite(tx).sum() < n * 0.9:
            raise WorkbenchError(f"Time column {p['time_col']!r} has too many unreadable values")
        order = np.argsort(np.where(np.isfinite(tx), tx, np.inf), kind="stable")
        x = tx[order]
        x_kind = "datetime" if key.startswith("d::") else "numeric"

    n_train = max(10, int(n * float(p["train_fraction"])))
    k, rk = float(p["z_threshold"]), float(p["robust_threshold"])
    gap, minlen = int(p["merge_gap"]), int(p["min_length"])
    window = int(p["window"])

    flags_z = np.zeros(n, bool)
    flags_r = np.zeros(n, bool)
    channels = []
    keep = _decimate(n)
    for c in sensors:
        if f"n::{c}" not in arrays:
            raise WorkbenchError(f"Column {c!r} is not numeric")
        raw = arrays[f"n::{c}"][order]
        miss = float(np.mean(~np.isfinite(raw)))
        if miss > 0.5:
            raise WorkbenchError(f"Column {c!r} is {miss:.0%} missing")
        v = _fill(raw)
        if p.get("smoothing", 1) > 1:
            v = pd.Series(v).ewm(span=int(p["smoothing"])).mean().to_numpy()
        zs, st = zscore_scores(v, n_train)
        rs = robust_scores(v, window)
        fz, fr = zs > k, rs > rk
        flags_z |= fz
        flags_r |= fr
        channels.append({
            "name": c, "unit": (p.get("units") or {}).get(c, ""), "missing_fraction": miss, "train_mean": st["mean"], "train_std": st["std"],
            "upper": st["mean"] + k * st["std"], "lower": st["mean"] - k * st["std"],
            "values": clean_floats(v[keep], 6),
            "flags_zscore": len(_intervals(fz, gap, minlen)), "flags_robust": len(_intervals(fr, gap, minlen)),
        })

    iz, ir = _intervals(flags_z, gap, minlen), _intervals(flags_r, gap, minlen)

    labels: list[list[int]] = []
    if p.get("label_col"):
        lab = arrays.get(f"n::{p['label_col']}")
        if lab is None:
            raise WorkbenchError("Label column must be 0/1")
        labels = to_sequences(np.nan_to_num(lab[order]) > 0.5)
    if p.get("labeled_intervals"):
        labels += parse_intervals(p["labeled_intervals"], n, x if x_kind != "step" else None, x_kind == "datetime")
    labels = sorted(labels)

    def xval(i: int):
        v = float(x[i])
        if x_kind == "datetime":
            return datetime.fromtimestamp(v, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        return v

    def describe(seqs, scores_fn):
        return [{"start": a, "end": b, "x0": xval(a), "x1": xval(b), "length": b - a + 1,
                 "channels": scores_fn(a, b)} for a, b in seqs]

    def chans_z(a, b):
        out = []
        for ch in channels:
            sl = (np.abs(_fill(arrays[f"n::{ch['name']}"][order][a:b + 1]) - ch["train_mean"]) / ch["train_std"])
            if sl.max() > k:
                out.append(ch["name"])
        return out

    metrics = None
    if labels:
        sz, sr = score_events(iz, labels), score_events(ir, labels)
        mz, mr = prf(sz["tp"], sz["fp"], sz["fn"]), prf(sr["tp"], sr["fp"], sr["fn"])
        metrics = {"zscore": mz, "robust": mr, "n_labeled": len(labels),
                   "found_zscore": sz["found"], "found_robust": sr["found"]}

    xs = [xval(int(i)) for i in keep]
    return {
        "dataset_id": dataset_id, "table": entry["id"], "file": entry["file"], "n_rows": n,
        "x_kind": x_kind, "x_label": p.get("time_col") or "step", "x": xs,
        "train_end": xval(n_train - 1), "settings": p,
        "channels": channels,
        "zscore": {"name": f"z-score > {k:g} (first {float(p['train_fraction']):.0%} as reference)",
                   "intervals": describe(iz, chans_z)},
        "robust": {"name": f"Rolling median/MAD > {rk:g} (window {window})",
                   "intervals": describe(ir, lambda a, b: [])},
        "labeled": [{"start": a, "end": b, "x0": xval(a), "x1": xval(b)} for a, b in labels],
        "metrics": metrics,
    }
