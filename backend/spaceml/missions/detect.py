"""Shared, CPU-light detectors and scoring for unlabeled or event-labeled telemetry.

* rolling_mad      robust baseline: rolling median / MAD z-score per sample (no training)
* windows          fixed-length windows that never cross an item (observation/visit) or split boundary
* intervals        boolean flags -> merged [start, end] intervals in time
* event_scores     event-level matching of alarm intervals against annotated events (local baseline scoring)
"""
from __future__ import annotations

import numpy as np


def rolling_mad(y: np.ndarray, window: int = 61, k: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Robust z = |y - rolling median| / (1.4826 * rolling MAD). Returns (z, flags). NaNs never flag."""
    import pandas as pd
    s = pd.Series(np.asarray(y, dtype=np.float64))
    w = max(5, int(window) | 1)
    med = s.rolling(w, center=True, min_periods=max(3, w // 4)).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=max(3, w // 4)).median()
    scale = 1.4826 * mad
    floor = max(float(np.nanmedian(scale.to_numpy())) * 0.1, 1e-12) if np.isfinite(scale).any() else 1e-12
    z = ((s - med).abs() / scale.clip(lower=floor)).to_numpy()
    z = np.where(np.isfinite(z), z, 0.0)
    return z, z > k


def windows(n: int, size: int, step: int) -> list[tuple[int, int]]:
    """[a, b) index windows inside ONE item. Callers window each item (and each split) separately, so no
    window ever crosses an observation, visit or train/test boundary."""
    if n < size:
        return [(0, n)] if n >= max(4, size // 2) else []
    return [(a, a + size) for a in range(0, n - size + 1, step)]


def window_features(t: np.ndarray, cols: dict[str, np.ndarray], spans: list[tuple[int, int]]) -> tuple[np.ndarray, list[str]]:
    names, rows = [], []
    for c in cols:
        names += [f"{c}:mean", f"{c}:std", f"{c}:absmax", f"{c}:p2p", f"{c}:slope", f"{c}:nan_frac"]
    for a, b in spans:
        r = []
        tt = t[a:b] - t[a]
        for c, y in cols.items():
            yy = np.asarray(y[a:b], dtype=np.float64)
            ok = np.isfinite(yy)
            frac = 1.0 - ok.mean() if len(yy) else 1.0
            if ok.sum() >= 3:
                v = yy[ok]
                slope = float(np.polyfit(tt[ok], v, 1)[0]) if np.ptp(tt[ok]) > 0 else 0.0
                r += [float(v.mean()), float(v.std()), float(np.abs(v).max()), float(np.ptp(v)), slope, float(frac)]
            else:
                r += [0.0, 0.0, 0.0, 0.0, 0.0, float(frac)]
        rows.append(r)
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def intervals(t: np.ndarray, flags: np.ndarray, merge_gap: float = 0.0, min_len: int = 1) -> list[dict]:
    """Contiguous flagged runs -> [{start, end, i0, i1, n}] in time units of t (merged when closer than merge_gap)."""
    f = np.asarray(flags, dtype=bool)
    out: list[dict] = []
    i, n = 0, len(f)
    while i < n:
        if f[i]:
            j = i
            while j + 1 < n and f[j + 1]:
                j += 1
            if out and merge_gap and t[i] - out[-1]["end"] <= merge_gap:
                out[-1].update(end=float(t[j]), i1=int(j), n=out[-1]["n"] + (j - i + 1))
            else:
                out.append({"start": float(t[i]), "end": float(t[j]), "i0": int(i), "i1": int(j), "n": int(j - i + 1)})
            i = j + 1
        else:
            i += 1
    return [o for o in out if o["n"] >= min_len]


def event_scores(alarms: list[dict], events: list[dict], t0: float, t1: float) -> dict:
    """Event-level matching inside [t0, t1]: an event is detected if any alarm overlaps it; an alarm is a false
    alarm if it overlaps no event. Local baseline scoring - NOT the ESA-ADB time-aware metrics."""
    ev = [e for e in events if e["end"] >= t0 and e["start"] <= t1]
    hit = [any(a["start"] <= e["end"] and a["end"] >= e["start"] for a in alarms) for e in ev]
    fa = [a for a in alarms if not any(a["start"] <= e["end"] and a["end"] >= e["start"] for e in ev)]
    by_cat: dict[str, dict] = {}
    for e, h in zip(ev, hit):
        c = by_cat.setdefault(e.get("category") or "event", {"events": 0, "detected": 0})
        c["events"] += 1
        c["detected"] += int(h)
    tp_alarms = len(alarms) - len(fa)
    return {"events": len(ev), "detected_events": int(sum(hit)), "missed_events": int(len(ev) - sum(hit)),
            "event_recall": float(sum(hit) / len(ev)) if ev else None,
            "alarms": len(alarms), "false_alarms": len(fa),
            "alarm_precision": float(tp_alarms / len(alarms)) if alarms else None, "by_category": by_cat,
            "protocol": "local baseline: event detected if any alarm overlaps it (not the official ESA-ADB metrics)"}


def persistent(over: np.ndarray, k: int = 2) -> np.ndarray:
    """Keep a window flag only when it belongs to a run of >= k consecutive over-threshold windows."""
    over = np.asarray(over, dtype=bool)
    if k <= 1 or not over.any():
        return over
    out = np.zeros_like(over)
    i, n = 0, len(over)
    while i < n:
        if over[i]:
            j = i
            while j + 1 < n and over[j + 1]:
                j += 1
            if j - i + 1 >= k:
                out[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return out
