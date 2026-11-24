"""Prepared, indexed mission data on disk (written once by prepare.py, read by the API, jobs and runs).

Layout (data/missions/<mission>/):
    manifest.json   source ids/version, file checksums, scope, counts, date range, channels with units,
                    validation results - the single record of what is installed
    items.json      selectable items (segments / exposures / visits / channels) with their metadata
    series/<item>.npz   full-resolution arrays: "t" (float64 seconds; epoch seconds when time_is_epoch) and
                        one float32 array per column "c__<name>"; optional "label" (int8 per sample)
    events.json     annotated intervals (anomalies / rare events) where the source publishes them

Queries read only the selected item and time range; display data is downsampled with a min/max envelope
(peaks survive), analysis always uses the full arrays.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..common.utils import now_iso, read_json, write_json
from ..config import DATA_DIR

ROOT = DATA_DIR / "missions"
VERS = ROOT / ".versions"          # earlier data versions kept when newer data is installed
KEEP_VERSIONS = 4


def mdir(mission: str) -> Path:
    return ROOT / mission


def safe(item: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(item))[:120]


def manifest(mission: str) -> dict | None:
    p = mdir(mission) / "manifest.json"
    return read_json(p) if p.exists() else None


def installed(mission: str) -> bool:
    m = manifest(mission)
    if not (m and m.get("items") and (mdir(mission) / "items.json").exists()):
        return False
    if mission in {"hubble", "webb"}:
        return any(item.get("n", 0) > 0 for item in items(mission))
    return True


def items(mission: str) -> list[dict]:
    p = mdir(mission) / "items.json"
    return read_json(p) if p.exists() else []


def events(mission: str) -> list[dict]:
    p = mdir(mission) / "events.json"
    return read_json(p) if p.exists() else []


def item(mission: str, item_id: str) -> dict:
    for it in items(mission):
        if str(it["id"]) == str(item_id):
            return it
    raise KeyError(f"{mission}: no item '{item_id}'")


def load(mission: str, item_id: str) -> dict:
    """Full-resolution arrays of one item: {'t': float64[n], 'cols': {name: float32[n]}, 'label': int8[n] | None}."""
    p = mdir(mission) / "series" / f"{safe(item_id)}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{mission}/{item_id}: series file missing ({p.name}) - re-run prepare")
    with np.load(p, allow_pickle=False) as z:
        cols = {k[3:]: z[k] for k in z.files if k.startswith("c__")}
        return {"t": z["t"], "cols": cols, "label": z["label"] if "label" in z.files else None}


# ------------------------------------------------------------------ writing (prepare)
class Writer:
    """Collects items and series for one mission and publishes them atomically (staging dir + rename)."""

    def __init__(self, mission: str):
        self.mission = mission
        self.stage = ROOT / f".{mission}.staging"
        import shutil
        shutil.rmtree(self.stage, ignore_errors=True)
        (self.stage / "series").mkdir(parents=True)
        self.items: list[dict] = []
        self.events: list[dict] = []

    def add(self, meta: dict, t: np.ndarray, cols: dict[str, np.ndarray], label: np.ndarray | None = None) -> dict:
        t = np.asarray(t, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        if not np.all(order == np.arange(len(t))):
            t = t[order]
            cols = {k: np.asarray(v)[order] for k, v in cols.items()}
            label = None if label is None else np.asarray(label)[order]
        arrays = {"t": t, **{f"c__{k}": np.asarray(v, dtype=np.float32) for k, v in cols.items()}}
        if label is not None:
            arrays["label"] = np.asarray(label, dtype=np.int8)
        np.savez_compressed(self.stage / "series" / f"{safe(meta['id'])}.npz", **arrays)
        g = gaps(t)
        meta = {**meta, "n": int(len(t)), "t0": float(t[0]) if len(t) else None, "t1": float(t[-1]) if len(t) else None,
                "columns": list(cols), "gaps": len(g),
                "missing": {k: int(np.isnan(np.asarray(v, dtype=np.float64)).sum()) for k, v in cols.items()}}
        self.items.append(meta)
        return meta

    def publish(self, man: dict) -> dict:
        write_json(self.stage / "items.json", self.items)
        write_json(self.stage / "events.json", self.events)
        files = sorted(p for p in self.stage.rglob("*") if p.is_file() and p.name != "manifest.json")
        man = {**man, "mission": self.mission, "items": len(self.items), "events": len(self.events),
               "prepared_at": man.get("prepared_at") or now_iso(),
               "prepared_files": len(files) + 1, "prepared_bytes": int(sum(p.stat().st_size for p in files)),
               "fingerprint": hashlib.sha256("".join(f"{p.name}:{p.stat().st_size}" for p in files).encode()).hexdigest()[:16]}
        write_json(self.stage / "manifest.json", man)
        import shutil
        dst = mdir(self.mission)
        old = ROOT / f".{self.mission}.old"
        shutil.rmtree(old, ignore_errors=True)
        prev = manifest(self.mission) if dst.exists() else None
        if dst.exists():
            dst.replace(old)
        self.stage.replace(dst)
        if prev and prev.get("fingerprint") != man["fingerprint"]:
            _archive(self.mission, old, prev)          # keep the earlier data as a selectable version
        shutil.rmtree(old, ignore_errors=True)
        return man


# ------------------------------------------------------------------ data versions
def _vid(man: dict) -> str:
    stamp = re.sub(r"[^0-9T]", "", str(man.get("prepared_at") or ""))[:15] or "bundled"
    return safe(f"{stamp}-{str(man.get('fingerprint') or 'x')[:8]}")


def version_label(man: dict) -> str:
    src = man.get("source") or {}
    dr = man.get("date_range") or []
    when = f"{str(dr[0])[:10]} → {str(dr[1])[:10]}" if len(dr) == 2 and dr[0] and dr[1] else ""
    ver = src.get("version") or ""
    return " · ".join(x for x in (when, f"source {ver}" if ver else "", man.get("scope") or "") if x)


def _archive(mission: str, folder: Path, man: dict) -> None:
    import shutil
    d = VERS / mission
    d.mkdir(parents=True, exist_ok=True)
    target = d / _vid(man)
    shutil.rmtree(target, ignore_errors=True)
    folder.replace(target)
    kept = sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in kept[KEEP_VERSIONS:]:
        shutil.rmtree(p, ignore_errors=True)


def versions(mission: str) -> list[dict]:
    """Installed data versions, the active one first."""
    out = []
    act = manifest(mission)
    if act:
        out.append({"id": _vid(act), "active": True, "label": version_label(act), "items": act.get("items"),
                    "date_range": act.get("date_range"), "prepared_at": act.get("prepared_at"),
                    "fingerprint": act.get("fingerprint")})
    d = VERS / mission
    for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []:
        m = read_json(p / "manifest.json") if (p / "manifest.json").exists() else None
        if m:
            out.append({"id": p.name, "active": False, "label": version_label(m), "items": m.get("items"),
                        "date_range": m.get("date_range"), "prepared_at": m.get("prepared_at"),
                        "fingerprint": m.get("fingerprint")})
    return out


def activate(mission: str, vid: str) -> dict:
    """Make an archived version the active one; the current one is archived (atomic renames)."""
    src = VERS / mission / safe(vid)
    if not (src / "manifest.json").exists():
        raise KeyError(f"{mission}: no data version '{vid}'")
    dst = mdir(mission)
    cur = manifest(mission)
    tmp = ROOT / f".{mission}.swap"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if dst.exists():
        dst.replace(tmp)
    src.replace(dst)
    if cur and tmp.exists():
        _archive(mission, tmp, cur)
    shutil.rmtree(tmp, ignore_errors=True)
    return manifest(mission) or {}


# ------------------------------------------------------------------ querying (API)
def gaps(t: np.ndarray, factor: float = 5.0, min_gap: float | None = None) -> list[dict]:
    """Packet / data gaps: intervals where the step exceeds `factor` × the median sampling step."""
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    if med <= 0:
        return []
    lim = max(factor * med, min_gap or 0.0)
    idx = np.nonzero(dt > lim)[0]
    return [{"start": float(t[i]), "end": float(t[i + 1]), "seconds": float(dt[i])} for i in idx]


def downsample_idx(y: np.ndarray, max_points: int) -> np.ndarray:
    """Indices of a min/max envelope: each bucket keeps its lowest and highest sample (spikes stay visible)."""
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    nb = max(1, max_points // 2)
    edges = np.linspace(0, n, nb + 1).astype(int)
    out = []
    yy = np.where(np.isnan(y), np.nanmedian(y) if np.isfinite(y).any() else 0.0, y)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = yy[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        out.extend(sorted({i, j}))
    return np.asarray(out, dtype=int)


def query(mission: str, item_id: str, columns: list[str] | None = None, start: float | None = None,
          end: float | None = None, max_points: int = 2000) -> dict:
    it = item(mission, item_id)
    d = load(mission, item_id)
    t = d["t"]
    lo = 0 if start is None else int(np.searchsorted(t, start, "left"))
    hi = len(t) if end is None else int(np.searchsorted(t, end, "right"))
    cols = {k: v for k, v in d["cols"].items() if not columns or k in columns}
    if columns:
        missing = [c for c in columns if c not in d["cols"]]
        if missing:
            raise KeyError(f"{mission}/{item_id}: unknown column(s) {missing}; available: {list(d['cols'])}")
    tt = t[lo:hi]
    primary = next(iter(cols.values()))[lo:hi] if cols else np.zeros(len(tt))
    idx = downsample_idx(primary.astype(np.float64), max_points)
    man = manifest(mission) or {}
    units = {c["id"]: c.get("unit") for c in man.get("channels", [])}

    def clean(a):
        a = np.asarray(a, dtype=np.float64)
        return [None if not np.isfinite(x) else round(float(x), 9) for x in a]

    out = {"mission": mission, "item": it, "n_total": int(hi - lo), "n_returned": int(len(idx)),
           "downsampled": bool(len(idx) < hi - lo), "time_basis": man.get("time_basis"),
           "time_is_epoch": bool(man.get("time_is_epoch")),
           "t": [float(x) for x in tt[idx]],
           "cols": {k: clean(v[lo:hi][idx]) for k, v in cols.items()},
           "units": {k: units.get(k) or it.get("units", {}).get(k) for k in cols},
           "missing": {k: int(np.isnan(np.asarray(v[lo:hi], dtype=np.float64)).sum()) for k, v in cols.items()},
           "gaps": gaps(tt)[:200]}
    if d["label"] is not None:
        lab = d["label"][lo:hi]
        out["label"] = [int(x) for x in lab[idx]]
        out["labeled_samples"] = int((lab > 0).sum())
    return out
