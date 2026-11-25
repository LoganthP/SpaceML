"""Static description of every supported telemetry source (what it is, where it comes from, what it allows).

Nothing here touches the network. Whether a source is *installed* is decided by store.py (prepared data on
disk); whether a model exists by models.py. The catalog only says what the app can do with a source once
its data is present, and where the data comes from.
"""
from __future__ import annotations

MISSIONS: dict[str, dict] = {
    "opssat": {
        "name": "ESA OPS-SAT", "agency": "ESA", "family": "segments",
        "summary": "Labeled telemetry segments from ESA's OPS-SAT CubeSat (OPSSAT-AD benchmark).",
        "item_label": "segment", "time_basis": "UTC timestamps of each sample (1 s or 5 s sampling)",
        "train": {"kind": "random_forest", "label": "Random forest on the published segment features",
                  "quick_note": "RandomForest (scikit-learn) on the 18 published features, official train/test split; seconds on a CPU"},
        "evaluate": "whole held-out segments (official test split)",
        "source": {"title": "OPSSAT-AD - anomaly detection dataset for satellite telemetry",
                   "doi": "10.5281/zenodo.12588359 (v1); newer: 10.5281/zenodo.15108715 (v2)",
                   "url": "https://zenodo.org/records/15108715",
                   "authors": "Ruszczak, Kotowski, Evans, Nalepa (2024)", "license": "CC BY 4.0",
                   "redistribution": "yes (attribution required)",
                   "reference": "https://github.com/kplabs-pl/OPS-SAT-AD"},
        "prepare": "python -m spaceml.missions.prepare opssat --from <folder with dataset.csv and segments.csv>",
    },
    "hubble": {
        "name": "Hubble Space Telescope", "agency": "NASA/ESA", "family": "pointing", "telescope": "hst",
        "summary": "Pointing and jitter from HST observation logs (_jit 3-second averages + _jif summaries).",
        "item_label": "exposure", "time_basis": "seconds since exposure start (jit 'Seconds', 3 s averages)",
        "train": {"kind": "isolation_forest_pointing", "label": "Isolation Forest on jitter windows (split by exposure)",
                  "quick_note": "separate Hubble detector; exposures split before windowing; review candidates only (no fault labels)"},
        "evaluate": "alert counts and reviewed outcomes (no verified fault labels)",
        "source": {"title": "HST observation logs (*_jit.fits, *_jif.fits) from MAST", "doi": None,
                   "url": "https://hst-docs.stsci.edu/hstdhb/5-observation-logs/5-1-observation-log-files",
                   "authors": "STScI / MAST", "license": "public MAST data", "redistribution": "yes (public archive data; cite MAST)",
                   "reference": "https://www.stsci.edu/hst/instrumentation/focus-and-pointing/pointing/jitter-file-format-definition"},
        "prepare": "python -m spaceml.missions.prepare hubble --from <folder of *_jit.fits / *_jif.fits>  (or --download <dataset ids>)",
    },
    "webb": {
        "name": "James Webb Space Telescope", "agency": "NASA/ESA/CSA", "family": "pointing", "telescope": "jwst",
        "summary": "Fine Guidance Sensor POINTING tables (FG / TRACK cal products): jitter and two-axis offsets per visit.",
        "item_label": "visit", "time_basis": "UTC from POINTING Time units (MJD days, or relative seconds/milliseconds anchored to the file stamp)",
        "train": {"kind": "isolation_forest_pointing", "label": "Isolation Forest on jitter windows (split by visit)",
                  "quick_note": "separate Webb detector; visits split before windowing; review candidates only (no fault labels)"},
        "evaluate": "alert counts and reviewed outcomes (no verified fault labels)",
        "source": {"title": "JWST FGS guider products (*_gs-fg_*_cal.fits, *_gs-track_*_cal.fits) from MAST", "doi": None,
                   "url": "https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-pointing-performance/jwst-fine-guide-stability",
                   "authors": "STScI / MAST", "license": "public MAST data", "redistribution": "yes (public archive data; cite MAST)",
                   "reference": "https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-pointing-performance/jwst-fine-guide-stability"},
        "prepare": "python -m spaceml.missions.prepare webb --from <folder of *_gs-fg_*_cal.fits / *_gs-track_*_cal.fits>",
    },
}

ORDER = ["opssat", "hubble", "webb"]


def get(mission: str) -> dict:
    if mission not in MISSIONS:
        raise KeyError(f"unknown mission '{mission}' (known: {', '.join(ORDER)})")
    return MISSIONS[mission]


def trainable(mission: str) -> bool:
    return bool(get(mission).get("train"))
