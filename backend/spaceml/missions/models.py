"""Model versions per mission: models/missions/<mission>/<version>/ (manifest.json, metrics.json, model.joblib,
predictions.json). Training writes to a hidden staging folder and renames it only on success, so a failed or
cancelled job never replaces the last working model. Active = newest full run, else newest quick run."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..common.utils import read_json
from ..config import MODELS_DIR

ROOT = MODELS_DIR / "missions"


def versions(mission: str) -> list[dict]:
    d = ROOT / mission
    out = []
    for v in sorted(d.iterdir()) if d.exists() else []:
        if v.is_dir() and not v.name.startswith(".") and (v / "manifest.json").exists():
            man = read_json(v / "manifest.json")
            met = read_json(v / "metrics.json") if (v / "metrics.json").exists() else None
            out.append({"version": v.name, "path": str(v), "manifest": man, "metrics": met,
                        "created": v.stat().st_mtime, "run_kind": man.get("run_kind", "quick")})
    return out


def active(mission: str) -> dict | None:
    """The model for the data version that is installed now (newest full run, else newest quick run).
    When no model was trained on this data version, the newest model is returned with stale=True."""
    vs = versions(mission)
    if not vs:
        return None
    from . import store
    fp = (store.manifest(mission) or {}).get("fingerprint")
    same = [v for v in vs if fp and v["manifest"].get("dataset_fingerprint") in (fp, None)]
    pick = max(same or vs, key=lambda v: (v["run_kind"] == "full", v["created"], v["version"]))
    # stale = trained on a different data version (a model without a recorded fingerprint is never called stale)
    return {**pick, "stale": bool(fp) and not same}


def version_dir(mission: str, version: str) -> Path:
    p = ROOT / mission / version
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(f"{mission}: model version {version} not found")
    return p


class Staging:
    """with Staging(mission, quick) as d: ...write into d...  -> published as <timestamp>-<kind> on success."""

    def __init__(self, mission: str, quick: bool):
        self.mission, self.kind = mission, "quick" if quick else "full"
        self.version = time.strftime("v%Y%m%d-%H%M%S") + f"-{self.kind}"
        self.dir = ROOT / mission / f".staging-{self.version}"

    def __enter__(self) -> Path:
        # a cancelled/killed job cannot clean up after itself; heavy jobs run one at a time, so any older
        # staging folder of this mission is abandoned work
        for old in (ROOT / self.mission).glob(".staging-*") if (ROOT / self.mission).exists() else []:
            shutil.rmtree(old, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def __exit__(self, et, ev, tb):
        if et is None:
            self.final = ROOT / self.mission / self.version
            self.dir.replace(self.final)
        else:
            shutil.rmtree(self.dir, ignore_errors=True)
        return False
