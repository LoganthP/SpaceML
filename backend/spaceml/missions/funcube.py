"""FUNcube-1 (AO-73): a fixed snapshot exported from the FUNcube Data Warehouse (CSV).

The warehouse export's exact column set was not reachable from the build environment, so the importer is
schema-checked rather than column-hard-coded: one timestamp column (name containing time/date, parsed as UTC)
and numeric engineering columns. Units are taken from "Name (unit)" / "Name [unit]" headers when present.
Every import records the column list, row counts, duplicates and gaps in the manifest for review.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import store

UNIT = re.compile(r"^(.*?)\s*[\(\[]([^\)\]]+)[\)\]]\s*$")


def prepare(src: Path) -> dict:
    df = pd.read_csv(src)
    tcol = next((c for c in df.columns if re.search(r"time|date", c, re.I)
                 and pd.to_datetime(df[c], utc=True, errors="coerce").notna().mean() > 0.95), None)
    if tcol is None:
        raise ValueError(f"no timestamp column found in {list(df.columns)[:12]}")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    df = df.assign(_t=ts).dropna(subset=["_t"]).sort_values("_t")
    dup = int(df.duplicated(subset=["_t"]).sum())
    df = df.drop_duplicates(subset=["_t"], keep="last")
    num = [c for c in df.columns if c not in (tcol, "_t") and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    if len(df) < 100 or not num:
        raise ValueError(f"{len(df)} rows and {len(num)} numeric fields - not enough for a replay")
    t = (df._t.astype("int64") // 10**9).to_numpy(np.float64)
    w = store.Writer("funcube")
    chans = []
    for c in num:
        m = UNIT.match(c)
        name, unit = (m.group(1).strip(), m.group(2).strip()) if m else (c, None)
        w.add({"id": re.sub(r"[^A-Za-z0-9_.-]", "_", name), "label": name, "field": c, "units": {"value": unit}},
              t, {"value": pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)})
        chans.append({"id": re.sub(r"[^A-Za-z0-9_.-]", "_", name), "name": name, "unit": unit})
    return w.publish({
        "source": {"name": "FUNcube Data Warehouse export - FUNcube-1 (AO-73, NORAD 39444)",
                   "files": {Path(src).name: {"sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(), "bytes": Path(src).stat().st_size}},
                   "license": "FUNcube Data Warehouse terms - verify before redistribution",
                   "attribution": "AMSAT-UK FUNcube team and the FUNcube Data Warehouse contributors."},
        "scope": f"snapshot: {len(df)} records, {len(num)} fields", "time_is_epoch": True,
        "time_basis": f"'{tcol}' column of the warehouse export (UTC)",
        "date_range": [str(df._t.iloc[0]), str(df._t.iloc[-1])],
        "counts": {"records": int(len(df)), "fields": len(num), "unparsable_timestamps": bad_ts, "duplicates": dup},
        "columns": list(map(str, df.columns.drop("_t"))), "channels": chans})
