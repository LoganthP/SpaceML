"""Analysis registry: each analysis declares what it accepts, which settings users can change,
what it produces and how it can be evaluated. The API validates requests against these
declarations and the frontend renders the settings form and picks the viewer from them.

Adding a module = add one entry here plus a ``run(dataset_id, params, run_dir)`` function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .store import WorkbenchError, get


def P(name, label, type, default=None, **kw) -> dict:  # noqa: N802 - tiny schema helper
    return {"name": name, "label": label, "type": type, "default": default, **kw}


ANALYSES: dict[str, dict[str, Any]] = {
    "timeseries.anomaly": {
        "title": "Time-series anomaly flags",
        "description": "Flags unusual intervals in telemetry, sensor logs or space-weather indices. Compares a "
                       "z-score baseline with a rolling robust detector; scores both against labels if you have them.",
        "input": "timeseries", "output": "timeline", "requires_model": None,
        "evaluation": "Event-level precision / recall / F1 when a 0/1 label column or labeled intervals are given",
        "params": [
            P("table", "Table", "table", required=True),
            P("time_col", "Time / step column", "column", None, filter=["datetime", "numeric"], optional=True,
              help="Leave empty to use the row number as the step"),
            P("sensor_cols", "Sensor columns", "columns", [], filter=["numeric"], min=1, max=12, required=True),
            P("units", "Units", "units", {}, optional=True,
              help="Optional unit per sensor column (e.g. V, °C, nT, km/s) - shown on charts and in the report"),
            P("label_col", "Label column (0/1)", "column", None, filter=["binary"], optional=True,
              help="Optional ground truth: 1 = anomalous row"),
            P("labeled_intervals", "Labeled intervals", "text", "", optional=True,
              help="Optional, e.g. 1200-1350, 4000-4100 (steps, or values of the time column)"),
            P("train_fraction", "Reference fraction", "number", 0.3, min=0.05, max=0.9, step=0.05,
              help="Leading part of the series treated as nominal for the z-score statistics"),
            P("z_threshold", "z-score threshold (σ)", "number", 4.0, min=1.5, max=20, step=0.5),
            P("robust_threshold", "Robust threshold (MAD σ)", "number", 6.0, min=2, max=30, step=0.5),
            P("window", "Rolling window (steps)", "integer", 120, min=10, max=20000),
            P("smoothing", "EWMA smoothing span", "integer", 1, min=1, max=500, help="1 = no smoothing"),
            P("merge_gap", "Merge flags closer than (steps)", "integer", 10, min=0, max=10000),
            P("min_length", "Minimum interval length (steps)", "integer", 1, min=1, max=10000),
        ],
    },
    "image.insights": {
        "title": "Classify & summarise images",
        "description": "Classifies every image (scene type, ground texture, exposure, sharpness), groups similar "
                       "images, finds near-duplicates and stereo pairs, and writes plain-language findings. With "
                       "the Mars model installed it also labels soil, bedrock, sand and big rock in each image.",
        "input": "images", "min_images": 1, "output": "insights", "requires_model": "mars (optional)",
        "evaluation": "Descriptive - scene types and texture are heuristics; terrain classes come from the trained model",
        "params": [
            P("use_model", "Use the Mars terrain model when installed", "select", "yes", options=[
                {"value": "yes", "label": "Yes - add soil / bedrock / sand / big rock per image"},
                {"value": "no", "label": "No - image statistics only (faster)"}]),
        ],
    },
    "image.change": {
        "title": "Before / after change detection",
        "description": "Compares two aligned images of the same area (orbital, aerial or rover) and maps where the "
                       "surface changed, with area estimates. Water mode measures flood / river extent from "
                       "multispectral bands.",
        "input": "images", "min_images": 2, "output": "change", "requires_model": None,
        "evaluation": "Changed area and fraction; regions ranked by size for review",
        "params": [
            P("before", "Before image", "image", required=True),
            P("after", "After image", "image", required=True),
            P("method", "Method", "select", "difference", options=[
                {"value": "difference", "label": "Generic difference (any imagery)"},
                {"value": "water", "label": "Surface water (NDWI, needs green + NIR bands)"}]),
            P("threshold_mode", "Threshold", "select", "otsu", options=[
                {"value": "otsu", "label": "Automatic (Otsu)"}, {"value": "manual", "label": "Manual"}],
              show_if={"method": "difference"}),
            P("manual_threshold", "Manual threshold", "number", 1.0, min=0.01, max=50, step=0.05,
              show_if={"threshold_mode": "manual", "method": "difference"}),
            P("smoothing_sigma", "Noise smoothing (px)", "number", 1.5, min=0, max=10, step=0.5,
              show_if={"method": "difference"}),
            P("green_band", "Green band #", "integer", 3, min=1, max=20, show_if={"method": "water"},
              help="Sentinel-2: 3 · Landsat 8/9 & HLS L30: 3 · 4-band PlanetScope: 2"),
            P("nir_band", "NIR band #", "integer", 8, min=1, max=20, show_if={"method": "water"},
              help="Sentinel-2: 8 · Landsat 8/9 & HLS L30: 5 · 4-band PlanetScope: 4"),
            P("water_threshold", "NDWI water threshold", "number", 0.0, min=-0.5, max=0.8, step=0.05,
              show_if={"method": "water"}),
            P("min_region_pixels", "Ignore regions smaller than (px)", "integer", 25, min=1, max=100000),
            P("pixel_size_m", "Pixel size (m)", "number", None, min=0.01, max=100000, optional=True,
              help="Only needed when the images are not projected GeoTIFFs"),
        ],
    },
    "image.terrain": {
        "title": "Terrain segmentation (Mars)",
        "description": "Labels each pixel as soil, bedrock, sand or big rock with the trained DeepLabV3+ model. "
                       "Works best on Curiosity / Perseverance NAVCAM-like grayscale images.",
        "input": "images", "min_images": 1, "output": "segmentation", "requires_model": "mars",
        "evaluation": "Per-image IoU vs. an all-soil baseline when an *_mask/*_label image with AI4Mars class ids is included",
        "params": [P("image", "Image", "image", required=True)],
    },
    "samples.classify": {
        "title": "Predict the label of each window (e.g. solar flares)",
        "description": "Summarises every labeled window (last value, mean, spread, trend per parameter), trains a "
                       "random forest and compares it with the classic single-parameter threshold. Reports TSS and "
                       "HSS - the skill scores used in flare forecasting - and sends misses and false alarms to "
                       "Review Studio.",
        "input": "samples", "output": "classification", "requires_model": None,
        "evaluation": "TSS, HSS, recall, false-alarm rate and ROC AUC on held-out windows (a different file when "
                      "you upload several partitions)",
        "params": [
            P("target", "What to predict", "select", "auto", options=[
                {"value": "auto", "label": "Automatic (≥M flares if there are enough, else ≥C; otherwise the rarest class)"},
                {"value": "ge_M", "label": "Flares of class M or X (≥M)"},
                {"value": "ge_C", "label": "Flares of class C, M or X (≥C)"},
                {"value": "minority", "label": "The rarest class vs the rest"}]),
            P("folds", "Cross-validation folds (one file)", "integer", 5, min=2, max=10),
            P("threshold_mode", "Alert threshold", "select", "auto", options=[
                {"value": "auto", "label": "Automatic - best TSS on the training data"},
                {"value": "manual", "label": "Fixed probability"}]),
            P("threshold", "Alert probability", "number", 0.5, min=0.05, max=0.95, step=0.05,
              show_if={"threshold_mode": "manual"}),
            P("trees", "Trees in the forest", "integer", 300, min=50, max=1000),
        ],
    },
    "lightcurve.transit": {
        "title": "Transit search + candidate score",
        "description": "Box least squares period search, phase folding, odd/even and secondary-eclipse checks, "
                       "the exoplanet CNN score and a 3D view of the system. Opens in the Exoplanet Finder.",
        "input": "lightcurve", "output": "lightcurve", "requires_model": "exoplanet (optional)",
        "evaluation": "Candidate score + diagnostics (no ground truth)", "params": [],
    },
}


def catalog() -> list[dict]:
    return [{"id": k, **v} for k, v in ANALYSES.items()]


def compatible(meta: dict) -> list[str]:
    """Compatible analyses, the one recommended for this dataset first."""
    out = _compatible(meta)
    try:
        from .profile import classify
        rec = classify(meta).get("recommended")
    except Exception:  # noqa: BLE001
        rec = None
    return sorted(out, key=lambda a: a != rec)


def _compatible(meta: dict) -> list[str]:
    out = []
    for k, a in ANALYSES.items():
        if a["input"] != meta.get("kind"):
            continue
        if meta.get("kind") == "images" and len(meta.get("images", [])) < a.get("min_images", 1):
            continue
        out.append(k)
    return out


def _shown(spec: dict, vals: dict) -> bool:
    return all(vals.get(k) == v for k, v in (spec.get("show_if") or {}).items())


def validate(analysis: str, dataset_id: str, params: dict) -> tuple[dict, dict]:
    """Return (meta, clean params). Raises WorkbenchError with a readable message."""
    if analysis not in ANALYSES or analysis == "lightcurve.transit":
        raise WorkbenchError(f"Unknown workbench analysis {analysis!r}")
    a = ANALYSES[analysis]
    meta = get(dataset_id)
    if analysis not in _compatible(meta):
        raise WorkbenchError(f"{a['title']} needs a {a['input']} dataset"
                             + (f" with at least {a['min_images']} images" if a.get("min_images", 1) > 1 else ""))
    clean: dict = {}
    table = None
    for spec in a["params"]:
        name, t = spec["name"], spec["type"]
        v = params.get(name, spec["default"])
        if v in ("", None) and t not in ("columns", "text"):
            v = spec["default"]
        if t == "number" and v is not None:
            v = _num(v, spec, float)
        elif t == "integer" and v is not None:
            v = int(_num(v, spec, float))
        elif t == "select":
            allowed = [o["value"] for o in spec["options"]]
            if v not in allowed:
                raise WorkbenchError(f"{spec['label']}: choose one of {', '.join(allowed)}")
        elif t == "table":
            ids = [x["id"] for x in meta.get("tables", [])]
            v = v or (ids[0] if ids else None)
            if v not in ids:
                raise WorkbenchError("Choose a table from this dataset")
            table = next(x for x in meta["tables"] if x["id"] == v)
        elif t in ("column", "columns"):
            cols = {c["name"]: c for c in (table or {}).get("columns", [])}
            vals = v if t == "columns" else ([v] if v else [])
            if isinstance(vals, str):
                vals = [vals]
            for c in vals:
                if c not in cols:
                    raise WorkbenchError(f"{spec['label']}: column {c!r} not found")
                if spec.get("filter") and cols[c]["type"] not in spec["filter"]:
                    raise WorkbenchError(f"{spec['label']}: {c!r} is {cols[c]['type']}, "
                                         f"expected {' or '.join(spec['filter'])}")
            if t == "columns":
                vals = list(dict.fromkeys(vals))
                if len(vals) < spec.get("min", 0) or len(vals) > spec.get("max", 999):
                    raise WorkbenchError(f"{spec['label']}: select {spec.get('min', 0)}-{spec.get('max')} columns")
                v = vals
            else:
                v = v or None
        elif t == "image":
            ids = [x["id"] for x in meta.get("images", [])]
            if v not in ids:
                raise WorkbenchError(f"{spec['label']}: choose an image from this dataset")
        elif t == "text":
            v = str(v or "")[:5000]
        elif t == "units":
            if not isinstance(v, dict):
                v = {}
            v = {str(k)[:120]: str(u).strip()[:24] for k, u in v.items() if str(u).strip()}
        if spec.get("required") and v in (None, "", []) and _shown(spec, clean | params):
            raise WorkbenchError(f"{spec['label']} is required")
        clean[name] = v
    if analysis == "timeseries.anomaly":
        used = [clean.get("time_col"), clean.get("label_col")]
        if any(c in clean["sensor_cols"] for c in used if c):
            raise WorkbenchError("The time and label columns cannot also be sensor columns")
    return meta, clean


def _num(v, spec, cast):
    try:
        x = cast(v)
    except (TypeError, ValueError) as e:
        raise WorkbenchError(f"{spec['label']} must be a number") from e
    if "min" in spec and x < spec["min"] or "max" in spec and x > spec["max"]:
        raise WorkbenchError(f"{spec['label']} must be between {spec.get('min')} and {spec.get('max')}")
    return x


def runner(analysis: str) -> Callable[[str, dict, Path, Path | None], dict]:
    if analysis == "timeseries.anomaly":
        from . import timeseries
        return lambda ds, p, rd, vd: timeseries.run(ds, p)
    if analysis == "image.change":
        from . import change
        return lambda ds, p, rd, vd: change.run(ds, p, rd)
    if analysis == "image.terrain":
        return _terrain
    if analysis == "samples.classify":
        from . import mvts
        return lambda ds, p, rd, vd: mvts.run(ds, p)
    if analysis == "image.insights":
        from . import imageinsights
        return lambda ds, p, rd, vd: imageinsights.run(ds, {**p, "use_model": p.get("use_model", "yes") == "yes"},
                                                       rd, vd)
    raise WorkbenchError(f"Unknown analysis {analysis}")


def _terrain(dataset_id: str, p: dict, run_dir: Path, version_dir: Path | None) -> dict:
    import numpy as np
    from PIL import Image

    from ..mars import infer
    from .store import dataset_dir, load_image, to_display

    if version_dir is None:
        raise FileNotFoundError("No Mars model installed. Train one (Data & Training page or "
                                "notebooks/mars_training.ipynb), then try again.")
    e, arr, _ = load_image(dataset_id, p["image"])
    gray = run_dir / "source.png"
    to_display(arr, [0] if arr.shape[0] < 3 else None).convert("L").save(gray)
    label = None
    if e.get("mask"):
        m = np.array(Image.open(dataset_dir(dataset_id) / e["mask"]))
        m = m[..., 0] if m.ndim == 3 else m
        if set(np.unique(m)).issubset({0, 1, 2, 3, 255}):
            label = dataset_dir(dataset_id) / e["mask"]
    res = infer.run(gray, version_dir, run_dir, label)
    res["image"] = {"id": e["id"], "file": e["file"]}
    return res
