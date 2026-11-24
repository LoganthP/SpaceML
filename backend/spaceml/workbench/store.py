"""Workbench datasets: one upload flow for any supported observation file.

An upload (single file or archive) is sniffed and stored as one of:

* ``lightcurve``  - photometry with time + flux/mag columns, or FITS light curves.
                    Delegated to ``spaceml.exoplanet.uploads`` (transit search, CNN score, 3D view).
* ``timeseries``  - any other table: telemetry, sensor logs, space-weather indices (CSV, TSV,
                    TXT, XLSX, JSON, Parquet, NPY). Stored column-wise in a compressed .npz.
* ``images``      - PNG / JPEG / TIFF / GeoTIFF rasters (single images for terrain segmentation,
                    pairs for change detection). Files named ``*_mask`` / ``*_label`` are kept as
                    ground truth for the image with the same stem.

storage/workbench/<id>/meta.json describes the dataset (profile, suggestions, file hashes);
the content hash makes every run report traceable to the exact bytes analysed.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..common.utils import now_iso, read_json, write_json
from ..config import STORAGE_DIR
from ..exoplanet import ingest

WB_DIR = STORAGE_DIR / "workbench"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TABLE_EXT = set(ingest.TABLE_EXT) | {".npy"}
MAX_ROWS = 2_000_000
MAX_COLS = 300
MAX_IMAGES = 200
MASK_WORDS = ("_mask", "_label", "-mask", "-label", "_gt", "_truth")
TIME_WORDS = ["timestamp", "time", "datetime", "date", "utc", "epoch", "t", "step", "index", "sol", "jd", "mjd"]
LABEL_WORDS = ["label", "labels", "anomaly", "is_anomaly", "anomalous", "attack", "fault", "event", "flag", "target", "y"]

Image.MAX_IMAGE_PIXELS = 400_000_000


class WorkbenchError(ValueError):
    """User-facing problem with an uploaded dataset."""


# ------------------------------------------------------------------ helpers
def _id_ok(dataset_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{12}", dataset_id or ""):
        raise WorkbenchError("Invalid dataset id")
    return dataset_id


def dataset_dir(dataset_id: str) -> Path:
    return WB_DIR / _id_ok(dataset_id)


def _sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _is_lightcurve_table(p: Path) -> bool:
    """True when a table has a recognisable time column and a flux/mag column."""
    try:
        df = ingest._read_table(p)
    except Exception:  # noqa: BLE001
        return False
    cols = {ingest._norm(c): c for c in df.columns}
    has_time = ingest._pick(cols, ingest.TIME_NAMES) is not None
    flux_names = [n for n in ingest.FLUX_NAMES + ingest.MAG_NAMES  # unambiguous photometry names only
                  if "flux" in n or "mag" in n or n in ("fcor", "kepmag", "tmag")]
    has_flux = ingest._pick(cols, flux_names) is not None
    headerless = all(re.fullmatch(r"(col)?\d+", str(c)) for c in df.columns)
    return has_time and has_flux and not headerless


def _gather(files: list[Path]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"images": [], "tables": [], "fits": []}
    for f in files:
        ext = f.suffix.lower()
        if ext in IMAGE_EXT:
            out["images"].append(f)
        elif ext in ingest.FITS_EXT:
            out["fits"].append(f)
        elif ext in TABLE_EXT:
            out["tables"].append(f)
    return out


def sniff(files: list[Path]) -> str:
    g = _gather(files)
    if g["images"] and len(g["images"]) >= len(g["tables"]) + len(g["fits"]):
        return "images"
    if g["fits"]:
        return "lightcurve"
    if g["tables"]:
        lc = sum(_is_lightcurve_table(t) for t in g["tables"][:20])
        return "lightcurve" if lc and lc >= len(g["tables"][:20]) / 2 else "timeseries"
    raise WorkbenchError("No supported files found. Upload CSV/TSV/TXT/XLSX/JSON/Parquet/NPY tables, "
                         "FITS light curves, or PNG/JPEG/TIFF images (optionally inside a .zip).")


# ------------------------------------------------------------------ public API
def create(raw_path: Path, filename: str, kind: str = "auto") -> dict:
    """Store an upload; returns dataset meta (or the exoplanet dataset for light curves)."""
    WB_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wb_", dir=WB_DIR))
    try:
        src = work / filename
        shutil.copy(raw_path, src)
        try:
            files = ingest.expand(src, work / "x")
        except ingest.IngestError as e:
            raise WorkbenchError(str(e)) from e
        files = [f for f in files if not f.name.startswith((".", "__MACOSX"))]
        from . import mvts
        labeled = mvts.detect(files) if kind in ("auto", "samples") else []
        detected = "samples" if labeled else (sniff(files) if kind == "auto" else kind)
        if detected == "samples" and not labeled:
            raise WorkbenchError("No labeled samples found: each JSON record needs a label and a 'values' object")
        if detected not in ("lightcurve", "timeseries", "images", "samples"):
            raise WorkbenchError(f"Unknown dataset kind {kind!r}")
        if detected == "lightcurve":
            from ..exoplanet import uploads
            try:
                meta = uploads.create(raw_path, filename)
            except ingest.IngestError as e:
                raise WorkbenchError(str(e)) from e
            return {"kind": "lightcurve", "id": meta["id"], "name": meta.get("name", filename),
                    "exoplanet_dataset": meta}
        ds_id = uuid.uuid4().hex[:12]
        out = WB_DIR / ds_id
        out.mkdir()
        try:
            if detected == "samples":
                meta = mvts.create(labeled, out)
                meta["sha256"] = _sha256(labeled)
            elif detected == "timeseries":
                meta = _create_timeseries(files, out)
            else:
                meta = _create_images(files, out)
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            raise
        meta.update({"id": ds_id, "kind": detected, "name": filename, "created_at": now_iso(),
                     "source_files": [f.name for f in files][:50], "n_source_files": len(files)})
        write_json(out / "meta.json", meta)
        return meta
    finally:
        shutil.rmtree(work, ignore_errors=True)


def list_all() -> list[dict]:
    out = []
    if WB_DIR.exists():
        for d in WB_DIR.iterdir():
            m = read_json(d / "meta.json")
            if m:
                out.append({k: m.get(k) for k in ("id", "kind", "name", "created_at", "sha256", "summary")})
    try:
        from ..exoplanet import uploads
        for m in uploads.list_all():
            out.append({"id": m["id"], "kind": "lightcurve", "name": m.get("name"), "created_at": m.get("created_at"),
                        "sha256": m.get("sha256"),
                        "summary": f"{m.get('n_targets', len(m.get('targets', [])))} light-curve target(s)"})
    except Exception:  # noqa: BLE001
        pass
    return sorted(out, key=lambda m: m.get("created_at") or "", reverse=True)


def get(dataset_id: str) -> dict:
    m = read_json(dataset_dir(dataset_id) / "meta.json")
    if not m:
        raise FileNotFoundError("Dataset not found")
    return m


def delete(dataset_id: str) -> None:
    shutil.rmtree(dataset_dir(dataset_id), ignore_errors=True)


# ------------------------------------------------------------------ time series
def _read_any_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".npy":
        a = np.load(p, allow_pickle=False)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2:
            raise WorkbenchError(f"{p.name}: expected a 1-D or 2-D array, got shape {a.shape}")
        return pd.DataFrame(a, columns=[f"col_{i}" for i in range(a.shape[1])])
    df = ingest._read_table(p)
    if all(re.fullmatch(r"\d+", str(c)) for c in df.columns):  # headerless -> readable names
        df.columns = [f"col{c}" for c in df.columns]
    return df


def _profile(name: str, s: pd.Series) -> dict:
    info: dict = {"name": name, "missing": int(s.isna().sum())}
    if pd.api.types.is_datetime64_any_dtype(s):
        info.update(type="datetime", min=str(s.min()), max=str(s.max()))
        return info
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.9:
        uniq = num.dropna().unique()
        is_bin = len(uniq) <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0})
        info.update(type="binary" if is_bin else "numeric",
                    min=_f(num.min()), max=_f(num.max()), mean=_f(num.mean()), std=_f(num.std()),
                    monotonic=bool(num.dropna().is_monotonic_increasing))
        return info
    if s.dtype == object:
        sample = s.dropna().astype(str).head(200)
        parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
        if len(sample) and parsed.notna().mean() >= 0.9:
            info.update(type="datetime", min=str(sample.iloc[0]), max=str(s.dropna().astype(str).iloc[-1]))
            return info
    info.update(type="text", examples=s.dropna().astype(str).head(3).tolist())
    return info


def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _suggest(profile: list[dict]) -> dict:
    by = {_norm(c["name"]): c for c in profile}
    time_col = next((c["name"] for c in profile if c["type"] == "datetime"), None)
    if time_col is None:
        for w in TIME_WORDS:
            c = by.get(w)
            if c and c["type"] == "numeric" and c.get("monotonic"):
                time_col = c["name"]
                break
    label_col = None
    for w in LABEL_WORDS:
        c = by.get(w)
        if c and c["type"] == "binary":
            label_col = c["name"]
            break
    sensors = [c["name"] for c in profile if c["type"] == "numeric" and c["name"] not in (time_col, label_col)
               and (c.get("std") or 0) > 0]
    return {"time_col": time_col, "label_col": label_col, "sensor_cols": sensors[:6]}


def _create_timeseries(files: list[Path], out: Path) -> dict:
    tables = [f for f in files if f.suffix.lower() in TABLE_EXT]
    if not tables:
        raise WorkbenchError("No table files found for a time-series dataset")
    entries, kept = [], []
    for i, p in enumerate(sorted(tables)[:50]):
        try:
            df = _read_any_table(p)
        except WorkbenchError:
            raise
        except Exception as e:  # noqa: BLE001
            raise WorkbenchError(f"Could not read {p.name}: {e}") from e
        if df.empty or len(df) < 20:
            raise WorkbenchError(f"{p.name} has only {len(df)} rows - at least 20 are needed")
        if len(df) > MAX_ROWS:
            raise WorkbenchError(f"{p.name} has {len(df):,} rows; the limit is {MAX_ROWS:,}. Split or resample it.")
        df = df.iloc[:, :MAX_COLS]
        df.columns = _dedupe([str(c).strip() or f"col_{j}" for j, c in enumerate(df.columns)])
        profile = [_profile(c, df[c]) for c in df.columns]
        arrays: dict[str, np.ndarray] = {}
        for c, pr in zip(df.columns, profile):
            if pr["type"] in ("numeric", "binary"):
                arrays[f"n::{c}"] = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
            elif pr["type"] == "datetime":
                ts = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
                arrays[f"d::{c}"] = (ts.astype("int64") / 1e9).where(ts.notna()).to_numpy(np.float64)
        if not any(k.startswith("n::") for k in arrays):
            raise WorkbenchError(f"{p.name} has no numeric columns to analyse")
        tid = f"t{i}"
        np.savez_compressed(out / f"{tid}.npz", **arrays)
        head = df.head(8).astype(object).where(df.head(8).notna(), None)
        entries.append({"id": tid, "file": p.name, "rows": int(len(df)), "columns": profile,
                        "head": [[_cell(v) for v in row] for row in head.itertuples(index=False)],
                        "suggested": _suggest(profile)})
        kept.append(p)
    return {"tables": entries, "sha256": _sha256(kept),
            "summary": f"{len(entries)} table(s), {sum(e['rows'] for e in entries):,} rows"}


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return str(v)[:60] if v is not None else None


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_table(dataset_id: str, table_id: str) -> tuple[dict, dict[str, np.ndarray]]:
    meta = get(dataset_id)
    entry = next((t for t in meta.get("tables", []) if t["id"] == table_id), None)
    if entry is None:
        raise WorkbenchError("Table not found in this dataset")
    z = np.load(dataset_dir(dataset_id) / f"{table_id}.npz")
    return entry, {k: z[k] for k in z.files}


# ------------------------------------------------------------------ images
def _read_raster(p: Path) -> tuple[np.ndarray, dict]:
    """Return (bands, H, W) float32 array and georeferencing info (if any)."""
    geo: dict = {"georeferenced": False}
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(p) as src:
                a = src.read(out_dtype="float32", masked=True)
                arr = a.filled(np.nan)
                if src.crs is not None and not src.transform.is_identity:
                    res = (abs(src.transform.a), abs(src.transform.e))
                    geo = {"georeferenced": True, "crs": src.crs.to_string(), "res": res,
                           "projected": bool(src.crs.is_projected),
                           "bounds": list(src.bounds), "transform": list(src.transform)[:6]}
                return arr, geo
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            raise WorkbenchError(f"Could not read {p.name}: {e}") from e
    try:
        with Image.open(p) as im:
            im.load()
            if im.mode in ("I;16", "I;16B", "I", "F"):
                arr = np.asarray(im, dtype=np.float32)[None]
            else:
                im = im.convert("RGB") if im.mode not in ("L", "RGB") else im
                a = np.asarray(im, dtype=np.float32)
                arr = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    except Exception as e:  # noqa: BLE001
        raise WorkbenchError(f"Could not read image {p.name}: {e}") from e
    if arr.shape[0] == 3 and np.allclose(arr[0], arr[1]) and np.allclose(arr[1], arr[2]):
        arr = arr[:1]  # grayscale saved as RGB
    return arr, geo


def to_display(arr: np.ndarray, bands: list[int] | None = None) -> Image.Image:
    """Percentile-stretched 8-bit preview (RGB from the first three bands, or grayscale)."""
    if bands is None:
        bands = [0, 1, 2] if arr.shape[0] >= 3 else [0]
    sel = arr[bands]
    out = np.zeros(sel.shape, np.uint8)
    for i, b in enumerate(sel):
        v = b[np.isfinite(b)]
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
        out[i] = np.clip((np.nan_to_num(b, nan=lo) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out[0]) if len(bands) == 1 else Image.fromarray(out.transpose(1, 2, 0))


def _create_images(files: list[Path], out: Path) -> dict:
    imgs = sorted(f for f in files if f.suffix.lower() in IMAGE_EXT)
    if not imgs:
        raise WorkbenchError("No images found")
    if len(imgs) > MAX_IMAGES:
        raise WorkbenchError(f"{len(imgs)} images found; the limit is {MAX_IMAGES} per dataset")
    masks = {f.stem.lower(): f for f in imgs if any(w in f.stem.lower() for w in MASK_WORDS)}
    entries, kept = [], []
    for i, p in enumerate(f for f in imgs if f.stem.lower() not in masks):
        arr, geo = _read_raster(p)
        iid = f"i{i}"
        ext = ".tif" if p.suffix.lower() in (".tif", ".tiff") else p.suffix.lower()
        shutil.copy(p, out / f"{iid}{ext}")
        thumb = to_display(arr)
        thumb.thumbnail((320, 320))
        thumb.save(out / f"{iid}_thumb.png")
        mask_for = next((m for k, m in masks.items() if k.startswith(p.stem.lower()) and k != p.stem.lower()), None)
        rec = {"id": iid, "file": p.name, "path": f"{iid}{ext}", "width": int(arr.shape[2]),
               "height": int(arr.shape[1]), "bands": int(arr.shape[0]), "geo": geo}
        if mask_for is not None:
            shutil.copy(mask_for, out / f"{iid}_mask.png")
            rec["mask"] = f"{iid}_mask.png"
        entries.append(rec)
        kept.append(p)
    if not entries:
        raise WorkbenchError("Only mask/label images were found - include the images themselves")
    return {"images": entries, "sha256": _sha256(kept),
            "summary": f"{len(entries)} image(s)" + (f", {sum('mask' in e for e in entries)} with masks"
                                                     if any("mask" in e for e in entries) else "")}


def image_entry(dataset_id: str, image_id: str) -> tuple[dict, Path]:
    meta = get(dataset_id)
    e = next((x for x in meta.get("images", []) if x["id"] == image_id), None)
    if e is None:
        raise WorkbenchError("Image not found in this dataset")
    return e, dataset_dir(dataset_id) / e["path"]


def load_image(dataset_id: str, image_id: str) -> tuple[dict, np.ndarray, dict]:
    e, p = image_entry(dataset_id, image_id)
    arr, geo = _read_raster(p)
    return e, arr, geo
