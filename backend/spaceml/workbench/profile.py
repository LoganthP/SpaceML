"""What is this dataset? Classifies an upload so the Workbench can explain it and pick the right analysis.

Uses file names first (NASA product ids encode mission, camera and sol), then content
(bands, georeferencing, column names). Every conclusion lists the evidence behind it.
"""
from __future__ import annotations

import re
from collections import Counter

# Mars 2020 (Perseverance) product ids, e.g. NLF_1699_0817769320_891ECM_N0790102NCAM00709_01_195J
M2020_RE = re.compile(r"^(?P<cam>[A-Z]{2})(?P<x>[A-Z0-9_])_(?P<sol>\d{4})_(?P<sclk>\d{10})")
# Curiosity (MSL) product ids, e.g. NLB_486272784EDR_F0481570NCAM00415M_
MSL_RE = re.compile(r"^(?P<cam>[A-Z]{2})(?P<x>[A-Z_])_(?P<sclk>\d{9})(?P<prod>[A-Z]{3})")

M2020_CAMERAS = {"NL": "Navcam left", "NR": "Navcam right", "FL": "Front Hazcam left", "FR": "Front Hazcam right",
                 "RL": "Rear Hazcam left", "RR": "Rear Hazcam right", "ZL": "Mastcam-Z left", "ZR": "Mastcam-Z right",
                 "SI": "SuperCam RMI", "SC": "SHERLOC context", "SW": "SHERLOC WATSON", "CC": "CacheCam",
                 "LR": "Lander vision", "ED": "EDL camera", "HN": "Ingenuity nav", "HS": "Ingenuity colour"}
MSL_CAMERAS = {"NL": "Navcam left", "NR": "Navcam right", "FL": "Front Hazcam left", "FR": "Front Hazcam right",
               "RL": "Rear Hazcam left", "RR": "Rear Hazcam right", "ML": "Mastcam left", "MR": "Mastcam right",
               "MH": "MAHLI", "CR": "ChemCam RMI", "MD": "MARDI"}
TERRAIN_CAMS = {"NL", "NR", "FL", "FR", "RL", "RR"}

SPACE_WEATHER_WORDS = {"bz", "by", "bx", "dst", "kp", "ae", "f107", "f10.7", "vsw", "speed", "density", "np",
                       "pdyn", "flow_speed", "proton_density", "imf", "symh", "sym-h", "ap"}
TELEMETRY_WORDS = {"telemetry", "channel", "voltage", "current", "temp", "temperature", "pressure", "battery",
                   "bus", "sensor", "rpm", "gyro", "wheel", "power", "attitude"}


def parse_name(name: str) -> dict | None:
    base = name.rsplit("/", 1)[-1].upper()
    m = M2020_RE.match(base)
    if m:
        cam = m.group("cam")
        return {"mission": "Mars 2020 (Perseverance)", "camera_code": cam,
                "camera": M2020_CAMERAS.get(cam, f"camera {cam}"), "sol": int(m.group("sol")),
                "sclk": int(m.group("sclk")), "eye": "L" if cam.endswith("L") else "R" if cam.endswith("R") else None}
    m = MSL_RE.match(base)
    if m and m.group("cam") in MSL_CAMERAS:
        cam = m.group("cam")
        try:
            from ..mars.traverse import parse_product_id
            pid = parse_product_id(base) or {}
        except Exception:  # noqa: BLE001
            pid = {}
        return {"mission": "Mars Science Laboratory (Curiosity)", "camera_code": cam, "camera": MSL_CAMERAS[cam],
                "sol": pid.get("sol_estimate"), "sclk": int(m.group("sclk")), "site": pid.get("site"),
                "drive": pid.get("drive"), "eye": "L" if cam.endswith("L") else "R" if cam.endswith("R") else None}
    return None


def _images(meta: dict) -> dict:
    imgs = meta.get("images", [])
    parsed = [(e, parse_name(e["file"])) for e in imgs]
    known = [p for _, p in parsed if p]
    reasons, facts = [], {"images": len(imgs)}
    geo = sum(bool((e.get("geo") or {}).get("georeferenced") or (e.get("geo") or {}).get("crs")) for e in imgs)
    bands = Counter(e.get("bands") for e in imgs)
    sizes = Counter((e.get("width"), e.get("height")) for e in imgs)
    masks = sum("mask" in e for e in imgs)
    if known and len(known) >= max(1, len(imgs) // 2):
        missions = Counter(p["mission"] for p in known)
        cams = Counter(p["camera"] for p in known)
        sols = sorted({p["sol"] for p in known if p.get("sol") is not None})
        mission = missions.most_common(1)[0][0]
        terrain_view = sum(p["camera_code"] in TERRAIN_CAMS for p in known)
        stereo = _stereo_pairs([e["file"] for e, p in parsed if p])
        facts.update(mission=mission, cameras=dict(cams), sols=sols[:20], stereo_pairs=stereo,
                     named=len(known))
        reasons.append(f"{len(known)} of {len(imgs)} file names are NASA {mission} product ids")
        reasons.append("cameras: " + ", ".join(f"{c} ×{n}" for c, n in cams.most_common(4)))
        if sols:
            reasons.append(f"sol {sols[0]}" + (f"–{sols[-1]}" if len(sols) > 1 else ""))
        if stereo:
            reasons.append(f"{stereo} left/right stereo pair(s)")
        if masks:
            reasons.append(f"{masks} image(s) come with a label mask - terrain accuracy can be measured")
        label = f"Mars rover surface imagery · {mission.split(' (')[-1].rstrip(')')}"
        if terrain_view:
            label += " · " + cams.most_common(1)[0][0].split(" ")[0]
        return {"category": "mars_rover_images", "label": label, "confidence": 0.95, "reasons": reasons,
                "facts": facts, "recommended": "image.insights",
                "why": "Classifies every image (scene type, terrain texture and - once the Mars model is "
                       "installed - soil/bedrock/sand/rock per pixel) and summarises the set."}
    if geo:
        reasons.append(f"{geo} of {len(imgs)} images are georeferenced GeoTIFFs")
        multi = sum(n for b, n in bands.items() if (b or 0) >= 4)
        if multi:
            reasons.append(f"{multi} image(s) have 4+ spectral bands (water/NDWI mode possible)")
        pair = len(imgs) >= 2
        if pair:
            reasons.append("two or more scenes: before/after change detection applies")
        return {"category": "orbital_imagery", "label": "Orbital / aerial imagery (georeferenced)",
                "confidence": 0.85, "reasons": reasons, "facts": {**facts, "georeferenced": geo},
                "recommended": "image.change" if pair else "image.insights",
                "why": "Maps where the surface changed between the dates, with areas in km²." if pair else
                       "Summarises brightness, texture and content of each scene."}
    names = " ".join(e["file"].lower() for e in imgs)
    before_after = len(imgs) >= 2 and (any(w in names for w in ("before", "after", "pre", "post"))
                                       or re.search(r"(19|20)\d\d", names) is not None) and len(sizes) == 1
    mono = bands.get(1, 0)
    reasons.append(f"{len(imgs)} image(s), {len(sizes)} distinct size(s), "
                   f"{'grayscale' if mono == len(imgs) else 'colour/multi-band'}")
    if before_after:
        reasons.append("file names look like dates or before/after and all images have the same size")
        return {"category": "image_pair", "label": "Before/after image series", "confidence": 0.6,
                "reasons": reasons, "facts": facts, "recommended": "image.change",
                "why": "The images look like the same place at different times."}
    return {"category": "generic_images", "label": "Image collection", "confidence": 0.5, "reasons": reasons,
            "facts": facts, "recommended": "image.insights",
            "why": "Groups the images by content and flags exposure, blur and duplicates; terrain classes for "
                   "rover-like images when the Mars model is installed."}


def _stereo_pairs(files: list[str]) -> int:
    keys: dict[tuple, set] = {}
    for f in files:
        p = parse_name(f)
        if p and p.get("eye") and p.get("sclk"):
            keys.setdefault((p["camera_code"][0], p["sclk"] // 10), set()).add(p["eye"])  # same instant
    return sum(1 for eyes in keys.values() if {"L", "R"} <= eyes)


def _tables(meta: dict) -> dict:
    tables = meta.get("tables", [])
    cols = [c for t in tables for c in t.get("columns", [])]
    names = {re.sub(r"[^a-z0-9.\-]", "", c["name"].lower()) for c in cols}
    numeric = [c for c in cols if c["type"] == "numeric"]
    times = [c["name"] for c in cols if c["type"] == "datetime"]
    labels = [t["suggested"]["label_col"] for t in tables if (t.get("suggested") or {}).get("label_col")] \
        or [c["name"] for c in cols if c["type"] == "binary" and "label" in c["name"].lower()]
    rows = sum(t.get("rows", 0) for t in tables)
    reasons = [f"{len(tables)} table(s), {rows:,} rows, {len(numeric)} numeric column(s)"]
    if times:
        reasons.append(f"time column: {times[0]}")
    if labels:
        reasons.append(f"0/1 column {labels[0]!r} can serve as ground truth (precision/recall)")
    sw = names & SPACE_WEATHER_WORDS
    tl = {n for n in names if any(w in n for w in TELEMETRY_WORDS)}
    facts = {"tables": len(tables), "rows": rows, "numeric_columns": len(numeric), "time_columns": times,
             "label_columns": labels}
    if len(sw) >= 2:
        reasons.append("solar-wind / geomagnetic columns: " + ", ".join(sorted(sw)[:6]))
        cat, label = "space_weather", "Space-weather time series"
    elif tl or labels:
        if tl:
            reasons.append("sensor-like columns: " + ", ".join(sorted(tl)[:6]))
        cat, label = "spacecraft_telemetry", "Spacecraft / sensor telemetry"
    else:
        cat, label = "sensor_timeseries", "Numeric time series"
    return {"category": cat, "label": label, "confidence": 0.75 if cat != "sensor_timeseries" else 0.55,
            "reasons": reasons, "facts": facts, "recommended": "timeseries.anomaly",
            "why": "Flags unusual intervals with a robust detector and a z-score baseline"
                   + (" and scores both against your labels." if labels else ".")}


def classify(meta: dict) -> dict:
    kind = meta.get("kind")
    if kind == "images":
        return _images(meta)
    if kind == "timeseries":
        return _tables(meta)
    if kind == "samples":
        sm = meta.get("samples", {})
        flare = sm.get("flare")
        return {"category": "flare_windows" if flare else "labeled_windows",
                "label": "Solar active-region windows labeled with flare class" if flare else "Labeled multivariate windows",
                "confidence": 0.95 if flare else 0.8,
                "reasons": [f"{sm.get('n', 0):,} records with a label and a 'values' object of {len(sm.get('params', []))} "
                            f"parameters × {sm.get('n_steps')} steps"] + (["SHARP magnetic-field parameter names and "
                                                                           "GOES flare classes (Q/B/C/M/X)"] if flare else []),
                "facts": {"classes": sm.get("class_counts")}, "recommended": "samples.classify",
                "why": "Learns which windows precede a flare and checks it against a single-parameter threshold."
                       if flare else "Learns to predict the label of each window and checks it against a simple threshold."}
    return {"category": "light_curve", "label": "Stellar light curve(s)", "confidence": 0.9,
            "reasons": ["time + flux columns or a Kepler/K2/TESS FITS file"], "facts": {},
            "recommended": "lightcurve.transit", "why": "Searches for periodic transits and scores the candidate."}
