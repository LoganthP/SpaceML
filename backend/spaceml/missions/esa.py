"""ESA Anomaly Dataset (ESA-AD) Missions 1-3: import a documented, reproducible subset.

Layout of an extracted ESA-MissionN folder (ESA-ADB paper, Supplementary Table 5; official prep script
notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py):
    channels/<channel>.zip   zip-compressed pickle: DataFrame indexed by timestamp, value column named after the channel
    labels.csv               ID, Channel, StartTime, EndTime          (one row per annotated channel interval)
    anomaly_types.csv        ID, Category (Anomaly | Rare Event | Communication Gap), Class, Subclass, ...
    channels.csv             Channel, Subsystem, Physical Unit, Group, Target  (column names vary - matched loosely)
    events.csv / telecommands.csv (optional)

Published channel IDs, timestamps (UTC), interval bounds and categories are kept verbatim. Values are put on
the same 30 s zero-order-hold grid the official prep uses, but holds never bridge gaps longer than 10 minutes
(those stay missing so gaps remain visible). Official benchmark split for Mission 1: test after 2007-01-01.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from ..common.zipsupport import enable_deflate64

enable_deflate64()     # channel files are zip-compressed pickles; the archives use Deflate64

OFFICIAL_TEST_SPLIT = {1: "2007-01-01"}
MAX_GAP = pd.Timedelta(minutes=10)


def _col(df: pd.DataFrame, *cands: str) -> str | None:
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for c in cands:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    return None


def read_channel(folder: Path, ch: str) -> pd.Series:
    p = folder / "channels" / f"{ch}.zip"
    if not p.exists():
        raise FileNotFoundError(f"channel file {p} missing")
    df = pd.read_pickle(p)
    if isinstance(df, pd.Series):
        s = df
    else:
        s = df[ch] if ch in df.columns else df.iloc[:, 0]
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(f"{ch}: index is {type(s.index).__name__}, expected timestamps")
    s = pd.to_numeric(s, errors="coerce")
    idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
    s = pd.Series(s.to_numpy(np.float64), index=idx).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _densest_year(labels: pd.DataFrame, types: pd.DataFrame, channels: list[str], idx: pd.DatetimeIndex):
    """The 1-year window (monthly steps) inside the channel history that holds the most annotated non-gap events,
    so the chronological split has events to learn around and to evaluate on."""
    cat = types.set_index("ID")["Category"] if "Category" in types.columns else pd.Series(dtype=str)
    ev = labels[labels.Channel.isin(channels)]
    ev = ev[[str(cat.get(i, "Anomaly")) != "Communication Gap" for i in ev.ID]]
    lo, hi = idx[0], idx[-1]
    if hi - lo <= pd.Timedelta(days=365) or ev.empty:
        return lo, hi + pd.Timedelta(seconds=1)
    best, best_t0 = -1, hi - pd.Timedelta(days=365)
    for t0 in pd.date_range(lo, hi - pd.Timedelta(days=365), freq="30D"):
        t1 = t0 + pd.Timedelta(days=365)
        inside = ((ev.StartTime >= t0) & (ev.EndTime < t1)).sum()
        # prefer windows whose events fall on both sides of the 70 % split
        cut = t0 + (t1 - t0) * 0.7
        both = int(((ev.StartTime >= t0) & (ev.EndTime < cut)).any()) + int(((ev.StartTime >= cut) & (ev.EndTime < t1)).any())
        score = inside + 1000 * both
        if score > best:
            best, best_t0 = score, t0
    return best_t0, best_t0 + pd.Timedelta(days=365)


def readiness(events: list[dict], split_at: float, items: int) -> dict:
    """Is the installed subset fit for Quick train + event-level evaluation?"""
    use = [e for e in events if e.get("category") != "Communication Gap"]
    train = sum(1 for e in use if e["end"] < split_at)
    test = sum(1 for e in use if e["start"] >= split_at)
    spanning = sum(1 for e in use if e["start"] < split_at <= e["end"])
    reason = None
    if items == 0:
        reason = "no channel has data in the selected period"
    elif test == 0:
        reason = "no annotated event falls in the evaluation period - choose a longer or later window"
    return {"train_events": train, "test_events": test, "events_spanning_split": spanning,
            "trainable": reason is None, "blocked_reason": reason}


def prepare(mission: str, folder: Path, channels: list[str] | None, start: str | None, end: str | None,
            rule: str = "30s") -> dict:
    n = int(mission.split("m")[-1])
    folder = Path(folder)
    for f in ("labels.csv", "anomaly_types.csv"):
        if not (folder / f).exists():
            raise ValueError(f"{folder} is not an extracted ESA-Mission{n} folder ({f} missing)")
    if not (folder / "channels").is_dir():
        raise ValueError(f"{folder}/channels missing")
    labels = pd.read_csv(folder / "labels.csv")
    types = pd.read_csv(folder / "anomaly_types.csv")
    need = {"ID", "Channel", "StartTime", "EndTime"} - set(labels.columns)
    if need:
        raise ValueError(f"labels.csv lacks {sorted(need)} (has {list(labels.columns)})")
    if not {"ID", "Category"} <= set(types.columns):
        raise ValueError(f"anomaly_types.csv lacks ID/Category (has {list(types.columns)})")
    labels["StartTime"] = pd.to_datetime(labels.StartTime, utc=True).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels.EndTime, utc=True).dt.tz_localize(None)
    meta = pd.read_csv(folder / "channels.csv") if (folder / "channels.csv").exists() else pd.DataFrame()
    available = sorted(p.stem for p in (folder / "channels").glob("*.zip"))
    if not channels:      # documented default: the 6 most-annotated channels
        channels = labels.Channel.value_counts().index.intersection(available)[:6].tolist() or available[:6]
    unknown = [c for c in channels if c not in available]
    if unknown:
        raise ValueError(f"unknown channels {unknown} (e.g. {available[:5]})")
    t0 = pd.Timestamp(start) if start else None
    t1 = pd.Timestamp(end) if end else None
    if t0 is None and t1 is None:
        # documented default: one year. Mission 1: around the official 2007-01-01 test boundary;
        # otherwise the latest year of the first selected channel.
        if n in OFFICIAL_TEST_SPLIT:
            t0, t1 = pd.Timestamp("2006-07-01"), pd.Timestamp("2007-07-01")
        else:
            t0, t1 = _densest_year(labels, types, channels, read_channel(folder, channels[0]).index)
        start, end = str(t0.date()), str(t1.date())
    w = store.Writer(mission)
    sums, gaps_total, ranges = {}, 0, []
    ucol = _col(meta, "Physical Unit", "Unit", "PhysicalUnit") if len(meta) else None
    scol = _col(meta, "Subsystem") if len(meta) else None
    chans_meta = []
    tmerge = types.set_index("ID")
    for ch in channels:
        p = folder / "channels" / f"{ch}.zip"
        sums[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        s = read_channel(folder, ch)
        if t0 is not None:
            s = s[s.index >= t0]
        if t1 is not None:
            s = s[s.index < t1]
        if len(s) < 10:
            raise ValueError(f"{ch}: only {len(s)} samples in the selected range")
        raw_gaps = int((s.index.to_series().diff() > MAX_GAP).sum())
        grid = pd.date_range(s.index[0].floor(rule), s.index[-1].ceil(rule), freq=rule)
        lim = max(1, int(MAX_GAP / pd.Timedelta(rule)))
        r = s.reindex(s.index.union(grid)).ffill(limit=None)
        # zero-order hold, but never across raw gaps longer than MAX_GAP
        last_obs = pd.Series(s.index, index=s.index).reindex(r.index).ffill()
        stale = (r.index.to_series() - last_obs) > MAX_GAP
        r[stale.to_numpy()] = np.nan
        r = r.reindex(grid)
        ep = (r.index.astype("int64") // 10**9).to_numpy(np.float64)
        lab = np.zeros(len(r), np.int8)
        mine = labels[labels.Channel == ch]
        for _, row in mine.iterrows():
            cat = str(tmerge.loc[row.ID, "Category"]) if row.ID in tmerge.index else "Anomaly"
            code = 1 if cat == "Anomaly" else 2 if cat == "Rare Event" else 3
            m = (r.index >= row.StartTime) & (r.index <= row.EndTime)
            lab[m] = np.maximum(lab[m], code)
            if row.EndTime >= r.index[0] and row.StartTime <= r.index[-1]:
                extra = {k: (None if pd.isna(tmerge.loc[row.ID, k]) else str(tmerge.loc[row.ID, k]))
                         for k in ("Class", "Subclass", "Dimensionality", "Locality") if row.ID in tmerge.index and k in tmerge.columns}
                w.events.append({"id": str(row.ID), "channel": ch, "start": row.StartTime.timestamp(),
                                 "end": row.EndTime.timestamp(), "category": cat, **extra})
        unit = None
        sub = None
        if ucol:
            mm = meta[meta[_col(meta, "Channel")] == ch] if _col(meta, "Channel") else pd.DataFrame()
            if len(mm):
                unit = None if pd.isna(mm.iloc[0][ucol]) else str(mm.iloc[0][ucol])
                sub = None if not scol or pd.isna(mm.iloc[0][scol]) else str(mm.iloc[0][scol])
        gaps_total += raw_gaps
        ranges.append((r.index[0], r.index[-1]))
        w.add({"id": ch, "label": ch + (f" · {sub}" if sub else ""), "channel": ch, "subsystem": sub,
               "raw_samples": int(len(s)), "raw_gaps_over_10min": raw_gaps, "units": {"value": unit},
               "events": int(((mine.EndTime >= r.index[0]) & (mine.StartTime <= r.index[-1])).sum())},
              ep, {"value": r.to_numpy(np.float64)}, label=lab)
        chans_meta.append({"id": ch, "name": ch, "unit": unit, "subsystem": sub})
    lo, hi = min(a for a, _ in ranges), max(b for _, b in ranges)
    off = OFFICIAL_TEST_SPLIT.get(n)
    if off and lo < pd.Timestamp(off) < hi:
        split_at, protocol = pd.Timestamp(off), f"official ESA-ADB Mission{n} test boundary ({off})"
    else:
        split_at = lo + (hi - lo) * 0.7
        protocol = "chronological 70/30 split of the installed subset (local, not the official benchmark split)"
    import json as _json
    sel = folder / "selection.json"
    origin = _json.loads(sel.read_text()) if sel.exists() else {}
    ready = readiness(w.events, split_at.timestamp(), len(w.items))
    return w.publish({
        "source": {"name": f"ESA Anomaly Dataset - ESA-Mission{n}", "doi": "10.5281/zenodo.15237121", "version": "v2",
                   "record": "https://zenodo.org/records/15237121", "archive": origin.get("file"),
                   "archive_md5": origin.get("md5"), "install_mode": origin.get("mode", "upload"),
                   "files": sums, "license": "CC BY 3.0 IGO",
                   "attribution": "ESA ESOC, KP Labs, Airbus DS - ESA Anomalies Dataset (ESA-AD)."},
        "training_readiness": ready,
        "scope": f"subset: channels {', '.join(channels)}; {lo.date()} to {hi.date()}; resampled {rule} ZOH (holds never bridge >10 min)",
        "time_basis": "UTC (prepared on a 30 s grid)", "time_is_epoch": True,
        "date_range": [str(lo), str(hi)], "split_at": split_at.timestamp(), "split_at_utc": str(split_at),
        "split_protocol": protocol, "selection": {"channels": channels, "start": start, "end": end, "resample": rule},
        "counts": {"channels": len(channels), "events": len(w.events), "raw_gaps_over_10min": gaps_total,
                   "available_channels": len(available), "labels_rows": int(len(labels))},
        "channels": chans_meta})
