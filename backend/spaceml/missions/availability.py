"""Requested missions whose real datasets have not been published in this release.

These are explicit catalog states, never synthetic telemetry or successful imports.
"""
ARCHIVED_SOURCES = {
    "esa-m1": ("ESA Mission 1", "ESA", "https://zenodo.org/communities/esa-ad/"),
    "esa-m2": ("ESA Mission 2", "ESA", "https://zenodo.org/communities/esa-ad/"),
    "esa-m3": ("ESA Mission 3", "ESA", "https://zenodo.org/communities/esa-ad/"),
    "funcube": ("FUNcube-1 (AO-73)", "AMSAT", "https://warehouse.funcube.org.uk/"),
    "satnogs": ("SatNOGS", "Libre Space Foundation", "https://db.satnogs.org/"),
}

# These adapters remain on disk, but are intentionally absent from the product.
SOURCES = {}


def catalog_entries():
    entries = []
    for mid, (name, agency, url) in SOURCES.items():
        entries.append({"id": "mission-" + mid, "module": "missions", "mission": mid, "title": name,
                        "dataset": {"ready": False, "trainable": False, "summary": "Data missing",
                                    "blocked_reason": "No validated real dataset is published for this mission."},
                        "model": None, "train": [], "download": None, "active_job": None, "last_failure": None,
                        "notes": "Source adapter retained; import and validation are required before publication.",
                        "status": {"data": "missing", "data_label": "Data missing", "model": "not_applicable",
                                   "readiness": "missing", "readiness_label": "Data missing",
                                   "page": "/telemetry?mission=" + mid,
                                   "next": {"kind": "none", "label": "Data missing — import and validate real source data"}}})
    return entries


def mission_status(mid):
    name, agency, url = SOURCES[mid]
    return {"id": mid, "name": name, "agency": agency, "family": "pointing", "summary": "Data missing — no validated dataset published.",
            "item_label": "channel", "time_basis": "Not yet validated", "trainable": False,
            "train_label": None, "detect": None, "evaluate": None,
            "source": {"title": name, "url": url, "doi": None, "authors": agency,
                       "license": "Verify upstream terms before import", "redistribution": "not assessed", "reference": url},
            "dataset": {"installed": False, "state": "Data missing", "scope": None, "counts": {},
                        "date_range": [], "items": 0, "events": 0, "bytes": 0, "source_version": None,
                        "next_action": "Import and validate real data before enabling this mission."},
            "model": {"ready": False, "version": None, "run_kind": None, "headline": None},
            "repair": None, "install": None, "versions": [], "auto_install": None}
