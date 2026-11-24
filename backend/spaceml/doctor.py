"""Check that everything needed to run Space ML Mission Control is in place.

    python -m spaceml.doctor            # prints a checklist with the fix for every problem
"""
from __future__ import annotations

import importlib
import platform
import sys
import urllib.request

from .config import DATA_DIR, MODELS_DIR, STORAGE_DIR, ensure_dirs

OK, WARN, BAD = "OK  ", "WARN", "FAIL"
SITES = {
    "NASA Exoplanet Archive (Kepler catalog)": "https://exoplanetarchive.ipac.caltech.edu/",
    "MAST (Kepler light curves)": "https://mast.stsci.edu/",
    "Hugging Face (AI4Mars)": "https://huggingface.co/",
    "GitHub (telemetry mirror)": "https://raw.githubusercontent.com/",
}


def line(status: str, what: str, fix: str = "") -> None:
    print(f"[{status}] {what}" + (f"\n        -> {fix}" if fix else ""))


def reachable(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "spaceml-doctor"})
        urllib.request.urlopen(req, timeout=8)
        return True
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code < 500
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ensure_dirs()
    print(f"Space ML Mission Control doctor - Python {platform.python_version()} on {platform.system()}\n")
    if sys.version_info < (3, 10):
        line(BAD, "Python 3.10+ required", "install Python 3.11 or 3.12")
    for mod in ("fastapi", "torch", "numpy", "pandas", "sklearn", "lightkurve", "astropy",
                "segmentation_models_pytorch", "datasets", "PIL", "psutil", "openpyxl", "scipy", "rasterio", "pyarrow"):
        try:
            m = importlib.import_module(mod)
            line(OK, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"{mod} not importable ({e})", "pip install -r backend/requirements.txt")
    print(f"\nFolders: data={DATA_DIR}  models={MODELS_DIR}  storage={STORAGE_DIR}\n")

    from .telemetry import data as tdata
    if tdata.is_downloaded():
        line(OK, "Telemetry data downloaded")
    else:
        line(BAD, "Telemetry data missing", "python -m spaceml.telemetry.data download")
    tel = sorted((MODELS_DIR / "telemetry").glob("*/*.pt"))
    line(OK if tel else WARN, f"Telemetry models: {len(tel)} channel checkpoints",
         "" if tel else "python -m spaceml.telemetry.train --channels T-8 C-1 M-6")

    from .exoplanet import data as edata
    st = edata.status()
    if not st["catalog"]["present"]:
        line(BAD, "Kepler TCE catalog missing", "python -m spaceml.exoplanet.data tce")
    else:
        line(OK if st["catalog"]["has_sky"] else WARN, "Kepler TCE catalog present"
             + ("" if st["catalog"]["has_sky"] else " (old format: no sky positions for the 3D sky view)"),
             "" if st["catalog"]["has_sky"] else "python -m spaceml.exoplanet.data tce   (re-download)")
        pc, lab = st.get("planets_cached", 0), st.get("labeled_cached", 0)
        enough = pc >= 15 and lab - pc >= 15
        line(OK if enough else WARN, f"Kepler light curves cached: {st['cached_stars']} stars, {lab} labeled TCEs "
             f"({pc} planet candidates)", "" if enough else "python -m spaceml.exoplanet.data lightcurves --n-stars 300")
    exo = sorted((MODELS_DIR / "exoplanet").glob("*/cnn.pt"))
    line(OK if exo else WARN, f"Exoplanet models: {len(exo)}", "" if exo else "python -m spaceml.exoplanet.train")

    from .mars import data as mdata
    n = len(mdata.list_samples())
    line(OK if n else WARN, f"Mars curated samples: {n}", "" if n else "python -m spaceml.mars.data samples --n 36")
    mars = sorted((MODELS_DIR / "mars").glob("*/model.pt"))
    line(OK if mars else WARN, f"Mars models: {len(mars)}",
         "" if mars else "GPU: notebooks/mars_training.ipynb  |  CPU demo: python -m spaceml.mars.train --quick")

    print("\nNetwork:")
    for name, url in SITES.items():
        r = reachable(url)
        line(OK if r else WARN, name + (" reachable" if r else " NOT reachable"),
             "" if r else "check firewall/proxy/VPN; downloads from this site will fail")


if __name__ == "__main__":
    main()
