"""Ready-to-upload sample datasets for the Workbench, built from real NASA data.

    python -m spaceml.workbench.samples telemetry          # CSVs from the SMAP/MSL test files (offline)
    python -m spaceml.workbench.samples imagery            # before/after GeoTIFFs from NASA GIBS (internet)
    python -m spaceml.workbench.samples all

Files are written to ``samples/`` in the project root. Upload them on the Workbench page.

* telemetry: one CSV per channel with ``step``, ``value`` (telemetry, column 0 of the NASA
  test file), the first three command channels, and ``anomaly`` (1 inside the expert-labeled
  intervals of labeled_anomalies.csv). This lets you test the upload flow end to end and get
  event-level precision/recall against NASA's labels.
* imagery: MODIS Terra corrected-reflectance snapshots of the same area on two dates, from the
  NASA Global Imagery Browse Services (GIBS) WMS, saved as EPSG:4326 GeoTIFFs so change
  detection can estimate the area in km^2. Inspect them for clouds before interpreting.
"""
from __future__ import annotations

import argparse
import urllib.parse

import numpy as np
import pandas as pd

from ..common.utils import download
from ..config import REPO_ROOT

OUT = REPO_ROOT / "samples"
GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

EVENTS = {
    # Lake Mead (Nevada/Arizona): reservoir shrinkage during the 2000-2022 drought.
    "lake-mead": {"bbox": (35.95, -114.85, 36.55, -114.0), "dates": ("2000-07-08", "2022-07-12"),
                  "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
                  "about": "Lake Mead, USA - reservoir extent 2000 vs 2022 (MODIS Terra, 250 m)"},
    # Manchar Lake / Indus valley (Pakistan): 2022 monsoon flood vs. the year before.
    "indus-flood": {"bbox": (26.0, 67.3, 27.6, 68.6), "dates": ("2021-08-28", "2022-08-28"),
                    "layer": "MODIS_Terra_CorrectedReflectance_Bands721",
                    "about": "Indus valley, Pakistan - 2022 flood vs 2021 (MODIS bands 7-2-1: water appears dark blue)"},
}


def telemetry(channels: list[str]) -> None:
    from ..telemetry import data

    chans = data.load_channels()
    OUT.mkdir(exist_ok=True)
    for c in channels:
        if c not in chans:
            print(f"  unknown channel {c}, skipping")
            continue
        _, test = data.load_arrays(c)
        label = np.zeros(len(test), int)
        for a, b in chans[c].sequences:
            label[a:b + 1] = 1
        df = pd.DataFrame({"step": np.arange(len(test)), "value": test[:, 0]})
        for k in range(1, min(4, test.shape[1])):
            df[f"command_{k}"] = test[:, k].astype(int)
        df["anomaly"] = label
        p = OUT / f"{chans[c].spacecraft.lower()}_{c}_test_with_labels.csv"
        df.to_csv(p, index=False, float_format="%.6g")
        print(f"  wrote {p.name}: {len(df):,} rows, {len(chans[c].sequences)} labeled anomaly interval(s)")


def imagery(event: str, size: int = 1024) -> None:
    import rasterio
    from PIL import Image
    from rasterio.transform import from_bounds

    ev = EVENTS[event]
    lat0, lon0, lat1, lon1 = ev["bbox"]
    w = size
    h = max(64, int(size * (lat1 - lat0) / (lon1 - lon0)))
    folder = OUT / event
    folder.mkdir(parents=True, exist_ok=True)
    for date in ev["dates"]:
        q = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": ev["layer"], "STYLES": "",
             "CRS": "EPSG:4326", "BBOX": f"{lat0},{lon0},{lat1},{lon1}", "WIDTH": w, "HEIGHT": h,
             "FORMAT": "image/png", "TIME": date}
        tif = folder / f"{event}_{date}.tif"
        if tif.exists():
            continue
        png = folder / f"{event}_{date}.png"
        download(f"{GIBS}?{urllib.parse.urlencode(q)}", png, timeout=40)
        with png.open("rb") as stream:
            head = stream.read(8)
        if not head.startswith(b"\x89PNG"):
            raise ValueError(f"GIBS returned a non-PNG response for {date}")
        with Image.open(png) as image:
            arr = np.asarray(image.convert("RGB")).transpose(2, 0, 1)
        tmp = tif.with_suffix(".tmp")
        with rasterio.open(tmp, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs="EPSG:4326",
                           transform=from_bounds(lon0, lat0, lon1, lat1, w, h), compress="deflate") as dst:
            dst.write(arr)
        tmp.replace(tif)
        print(f"  wrote {tif}")
    (folder / "ABOUT.txt").write_text(f"{ev['about']}\nSource: NASA GIBS WMS, layer {ev['layer']}, dates {ev['dates']}\n"
                                      f"Bounding box (lat/lon): {ev['bbox']}\nUpload both .tif files together (zip the folder).\n")
    import zipfile
    for date in ev["dates"]:
        (folder / f"{event}_{date}.png").unlink(missing_ok=True)
    path = OUT / f".{event}.tmp"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*(f"{event}_{date}.tif" for date in ev["dates"]), "ABOUT.txt"]:
            archive.write(folder / name, name)
    path.replace(OUT / f"{event}.zip")
    print(f"  saved {OUT / (event + '.zip')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Workbench sample datasets from real NASA data")
    p.add_argument("what", choices=["telemetry", "imagery", "all"])
    p.add_argument("--channels", nargs="+", default=["P-1", "T-8", "C-1", "M-6"])
    p.add_argument("--event", choices=list(EVENTS), nargs="+", default=list(EVENTS))
    p.add_argument("--out", choices=["repo", "data"], default="repo",
                   help="repo = samples/ (for uploading by hand), data = the app's data volume (Data & Training)")
    a = p.parse_args()
    global OUT
    if a.out == "data":
        from ..config import DATA_DIR
        OUT = DATA_DIR / "workbench_samples"
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    if a.what in ("telemetry", "all"):
        print("Telemetry samples (from data/telemetry):")
        telemetry(a.channels)
    if a.what in ("imagery", "all"):
        for e in a.event:
            print(f"Imagery sample {e} (downloading from NASA GIBS):")
            try:
                imagery(e)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{e}: {type(ex).__name__}: {ex}")
                print(f"  failed: {ex}. Check your internet connection; you can also export two dates from "
                      "https://worldview.earthdata.nasa.gov (Take snapshot → GeoTIFF) and upload them.")
        if errors:
            from ..common.progress import fail
            fail("NASA GIBS scene preparation failed. Retry or upload matching GeoTIFFs in Workbench. " + "; ".join(errors))
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
