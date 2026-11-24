"""Before/after change detection for two images of the same area.

Methods
* ``difference`` - any imagery (grayscale, RGB, multispectral). Each band of the "after"
  image is matched to the "before" image's mean/std (removes global illumination change),
  the per-pixel change magnitude is the RMS band difference, lightly smoothed, and pixels
  above an Otsu (automatic) or manual threshold are changed. Small specks are removed.
* ``water`` - multispectral imagery with a green and a near-infrared band (Landsat,
  Sentinel-2, NASA HLS ...). NDWI = (G - NIR) / (G + NIR); water where NDWI > threshold.
  Reports water gained, water lost and persistent water.

Georeferenced GeoTIFFs give the area in km^2 from the pixel size (projected CRS). For a
geographic CRS or plain images the user supplies the pixel size in metres, otherwise the area
is reported in pixels only. If the "after" GeoTIFF is on a different grid it is reprojected
onto the "before" grid; plain images of different size are resized (flagged in the result).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .store import WorkbenchError, image_entry, load_image, to_display

MAX_SIDE = 4096


def otsu(values: np.ndarray, bins: int = 256) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between[:-1]))])


def _align(before_id, after_id, dataset_id):
    eb, b, gb = load_image(dataset_id, before_id)
    ea, a, ga = load_image(dataset_id, after_id)
    notes = []
    if b.shape[1:] != a.shape[1:]:
        if gb.get("georeferenced") and ga.get("georeferenced"):
            import rasterio
            from rasterio.warp import Resampling, reproject

            _, pb = image_entry(dataset_id, before_id)
            _, pa = image_entry(dataset_id, after_id)
            with rasterio.open(pb) as sb, rasterio.open(pa) as sa:
                dst = np.full((a.shape[0],) + b.shape[1:], np.nan, np.float32)
                for i in range(a.shape[0]):
                    reproject(a[i], dst[i], src_transform=sa.transform, src_crs=sa.crs,
                              dst_transform=sb.transform, dst_crs=sb.crs, resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
            a = dst
            notes.append("After image reprojected onto the before image's grid")
        else:
            a = np.stack([np.asarray(Image.fromarray(band).resize((b.shape[2], b.shape[1]), Image.BILINEAR))
                          for band in a])
            notes.append(f"Images differ in size ({ea['width']}x{ea['height']} vs {eb['width']}x{eb['height']}); "
                         "after image resized - make sure both cover the same area")
    if a.shape[0] != b.shape[0]:
        k = min(a.shape[0], b.shape[0])
        a, b = a[:k], b[:k]
        notes.append(f"Band counts differ; using the first {k} band(s)")
    return eb, ea, b, a, gb, notes


def _area_per_pixel(geo: dict, pixel_size_m: float | None) -> tuple[float | None, str]:
    if pixel_size_m:
        return pixel_size_m ** 2, f"pixel size {pixel_size_m:g} m (user)"
    if geo.get("georeferenced") and geo.get("projected"):
        rx, ry = geo["res"]
        return rx * ry, f"pixel size {rx:g} x {ry:g} m from {geo['crs']}"
    if geo.get("georeferenced") and geo.get("res") and geo.get("bounds"):
        # geographic CRS (degrees): metres per degree at the scene's centre latitude
        lat = (geo["bounds"][1] + geo["bounds"][3]) / 2
        mx = geo["res"][0] * 111_320 * float(np.cos(np.radians(lat)))
        my = geo["res"][1] * 110_574
        return mx * my, f"≈{mx:.0f} x {my:.0f} m pixels (lat/lon grid at {lat:.1f}°)"
    return None, "pixels (set a pixel size to get km²)"


def _regions(mask: np.ndarray, px_area: float | None, min_pixels: int, top: int = 8) -> tuple[np.ndarray, list[dict]]:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, []
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_pixels) + 1
    cleaned = np.isin(lab, keep)
    objs = ndimage.find_objects(lab)
    regs = []
    for i in keep[np.argsort(-sizes[keep - 1])][:top]:
        sl = objs[i - 1]
        px = int(sizes[i - 1])
        regs.append({"pixels": px, "area_km2": px * px_area / 1e6 if px_area else None,
                     "bbox": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)]})
    return cleaned, regs


def _footprint(geo: dict) -> list[float] | None:
    """[west, south, east, north] in degrees, for placing the scene on the 3D globe."""
    if not geo.get("georeferenced") or not geo.get("bounds") or not geo.get("crs"):
        return None
    try:
        from rasterio.warp import transform_bounds
        w, s, e, n = transform_bounds(geo["crs"], "EPSG:4326", *geo["bounds"], densify_pts=21)
        return [round(float(v), 5) for v in (w, s, e, n)]
    except Exception:  # noqa: BLE001
        return None


def run(dataset_id: str, p: dict, run_dir: Path) -> dict:
    if p["before"] == p["after"]:
        raise WorkbenchError("Pick two different images")
    eb, ea, b, a, geo, notes = _align(p["before"], p["after"], dataset_id)
    if max(b.shape[1:]) > MAX_SIDE:
        f = MAX_SIDE / max(b.shape[1:])
        size = (int(b.shape[2] * f), int(b.shape[1] * f))
        b = np.stack([np.asarray(Image.fromarray(x).resize(size, Image.BILINEAR)) for x in b])
        a = np.stack([np.asarray(Image.fromarray(x).resize(size, Image.BILINEAR)) for x in a])
        notes.append(f"Downsampled to {size[0]}x{size[1]} for analysis")
        if geo.get("res"):
            geo = {**geo, "res": (geo["res"][0] / f, geo["res"][1] / f)}
    valid = np.all(np.isfinite(b), 0) & np.all(np.isfinite(a), 0)
    if valid.mean() < 0.05:
        raise WorkbenchError("The two images barely overlap (less than 5% valid pixels)")
    px_area, area_note = _area_per_pixel(geo, p.get("pixel_size_m") or None)
    method = p["method"]
    min_px = int(p["min_region_pixels"])
    hist = None

    if method == "water":
        g, nir = int(p["green_band"]) - 1, int(p["nir_band"]) - 1
        if max(g, nir) >= b.shape[0]:
            raise WorkbenchError(f"Water mode needs bands {g + 1} and {nir + 1}; the images have {b.shape[0]} band(s). "
                                 "Use the difference method for grayscale/RGB images.")

        def ndwi(x):
            with np.errstate(invalid="ignore", divide="ignore"):
                return (x[g] - x[nir]) / (x[g] + x[nir])
        nb, na = ndwi(b), ndwi(a)
        thr = float(p["water_threshold"])
        wb, wa = (nb > thr) & valid, (na > thr) & valid
        gain, _ = _regions(wa & ~wb, px_area, min_px)
        loss, _ = _regions(wb & ~wa, px_area, min_px)
        changed = gain | loss
        _, regions = _regions(changed, px_area, min_px)
        classes = np.zeros(b.shape[1:], np.uint8)
        classes[wb & wa] = 3
        classes[loss] = 2
        classes[gain] = 1
        legend = {"1": "water gained", "2": "water lost", "3": "persistent water"}
        counts = {"water_before": int(wb.sum()), "water_after": int(wa.sum()),
                  "gained": int(gain.sum()), "lost": int(loss.sum())}
        threshold = thr
    else:
        bb, aa = b.copy(), a.copy()
        for i in range(bb.shape[0]):  # radiometric normalisation of "after" to "before"
            vb, va = bb[i][valid], aa[i][valid]
            sb, sa = vb.std() or 1.0, va.std() or 1.0
            aa[i] = (aa[i] - va.mean()) / sa * sb + vb.mean()
            bb[i] = bb[i] / sb
            aa[i] = aa[i] / sb
        mag = np.sqrt(np.nanmean((aa - bb) ** 2, axis=0))
        mag = ndimage.gaussian_filter(np.nan_to_num(mag), sigma=float(p["smoothing_sigma"]))
        mag[~valid] = np.nan
        threshold = otsu(mag) if p["threshold_mode"] == "otsu" else float(p["manual_threshold"])
        raw = (mag > threshold) & valid
        changed, regions = _regions(ndimage.binary_opening(raw, iterations=1), px_area, min_px)
        classes = changed.astype(np.uint8)
        legend = {"1": "changed"}
        counts = {"changed": int(changed.sum())}
        h, e = np.histogram(mag[np.isfinite(mag)], bins=60)
        hist = {"counts": h.tolist(), "edges": [round(float(x), 4) for x in e], "threshold": threshold}

    to_display(b, [0, 1, 2] if b.shape[0] >= 3 and method != "water" else None).save(run_dir / "before.png")
    to_display(a, [0, 1, 2] if a.shape[0] >= 3 and method != "water" else None).save(run_dir / "after.png")
    Image.fromarray(classes).save(run_dir / "change.png")

    n_valid = int(valid.sum())
    n_changed = int((classes == 1).sum() + (classes == 2).sum())
    return {
        "dataset_id": dataset_id, "method": method, "settings": p,
        "before": {"id": eb["id"], "file": eb["file"]}, "after": {"id": ea["id"], "file": ea["file"]},
        "width": int(b.shape[2]), "height": int(b.shape[1]), "bands": int(b.shape[0]),
        "threshold": threshold, "legend": legend, "counts": counts,
        "valid_fraction": n_valid / valid.size,
        "changed_fraction": n_changed / max(n_valid, 1),
        "changed_area_km2": n_changed * px_area / 1e6 if px_area else None,
        "pixel_area_m2": px_area, "area_basis": area_note,
        "regions": regions, "histogram": hist, "notes": notes,
        "georeferenced": bool(geo.get("georeferenced")), "crs": geo.get("crs"),
        "footprint_lonlat": _footprint(geo),
        "files": {"before": "before.png", "after": "after.png", "change": "change.png"},
    }
