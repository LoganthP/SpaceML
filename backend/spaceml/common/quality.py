"""Data-quality checks shown before an analysis runs.

Every check returns a list of plain findings and an overall status:

    ready              enough good data to analyse
    degraded           usable, but with gaps/stale readings the user should know about
    insufficient_data  too little valid data - the analysis would not mean anything
    source_unavailable the data could not be fetched (set by the caller)

The point is that a failed fetch or a mostly-empty window can never look like a clean result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_ORDER = ["ready", "degraded", "insufficient_data", "source_unavailable"]


def worst(*statuses: str) -> str:
    return max(statuses, key=STATUS_ORDER.index) if statuses else "ready"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (inclusive) index pairs of True runs."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def stale_runs(v: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Runs of identical consecutive finite values of at least `min_len` samples (frozen sensor)."""
    if len(v) < min_len:
        return []
    same = np.zeros(len(v), bool)
    same[1:] = (v[1:] == v[:-1]) & np.isfinite(v[1:])
    out = []
    for a, b in _runs(same):
        start = max(a - 1, 0)
        if b - start + 1 >= min_len:
            out.append((start, b))
    return out


def series_quality(df: pd.DataFrame, columns: list[str], *, required: list[str] | None = None,
                   stale_min: int = 6, step_label: str = "h", min_valid: float = 0.7,
                   continuous: set[str] | None = None) -> dict:
    """Quality of a regularly sampled table (index = time or step)."""
    required = required or columns
    continuous = continuous if continuous is not None else set(columns)
    n = len(df)
    items, statuses = [], []
    for c in columns:
        v = df[c].to_numpy(float) if c in df else np.full(n, np.nan)
        ok = np.isfinite(v)
        gaps = _runs(~ok)
        longest = max((b - a + 1 for a, b in gaps), default=0)
        stale = stale_runs(v, stale_min) if c in continuous else []
        stale_n = sum(b - a + 1 for a, b in stale)
        valid = float(ok.mean()) if n else 0.0
        st = "ready"
        if c in required and valid < min_valid:
            st = "insufficient_data"
        elif valid < 0.98 or stale_n:
            st = "degraded"
        statuses.append(st)
        it = {"name": c, "valid_fraction": round(valid, 4), "missing": int((~ok).sum()), "gaps": len(gaps),
              "longest_gap": int(longest), "stale_samples": int(stale_n), "status": st}
        if gaps and isinstance(df.index, pd.DatetimeIndex):
            a, b = max(gaps, key=lambda g: g[1] - g[0])
            it["longest_gap_at"] = [df.index[a].isoformat(), df.index[b].isoformat()]
        items.append(it)
    overall = worst(*statuses) if items else "insufficient_data"
    if n == 0:
        overall = "insufficient_data"
    findings = [f"{it['name']}: only {it['valid_fraction']:.0%} valid - not enough to analyse"
                for it in items if it["status"] == "insufficient_data"]
    for it in items:
        if it["status"] == "insufficient_data":
            pass
        elif it["missing"]:
            findings.append(f"{it['name']}: {it['missing']} missing {step_label} in {it['gaps']} gap(s), "
                            f"longest {it['longest_gap']} {step_label}")
        if it["stale_samples"]:
            findings.append(f"{it['name']}: {it['stale_samples']} {step_label} of repeated identical values (possibly stale)")
    return {"status": overall, "samples": n, "columns": items, "findings": findings}
