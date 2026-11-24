"""Discovers versioned checkpoints under models/<module>/<version>/ and their metrics."""
from __future__ import annotations

import json
from pathlib import Path

from spaceml.common.utils import read_json
from spaceml.config import MODELS_DIR

from .db import row_to_dict, tx

MODULES = ("telemetry", "exoplanet", "mars", "spaceweather")


_sig: tuple | None = None
_checked = 0.0


def _signature() -> tuple:
    files = []
    for module in MODULES:
        root = MODELS_DIR / module
        if root.exists():
            for f in [p for p in list(root.glob("*/manifest.json")) + list(root.glob("*/metrics.json"))
                      if not p.parent.name.startswith(".")]:
                try:
                    files.append((str(f), f.stat().st_mtime))
                except OSError:
                    pass
    return tuple(sorted(files))


def refresh() -> None:
    """Rescan at most every 3 s, and only if a manifest/metrics file changed.

    New checkpoints therefore show up without a restart or a manual rescan.
    """
    import time
    global _checked
    if time.time() - _checked < 3:
        return
    _checked = time.time()
    if _signature() != _sig:
        scan()


def scan() -> None:
    """Sync the model_versions and evaluations tables with what is on disk."""
    global _sig
    _sig = _signature()
    with tx() as c:
        c.execute("DELETE FROM model_versions")
        c.execute("DELETE FROM evaluations")
        for module in MODULES:
            root = MODELS_DIR / module
            if not root.exists():
                continue
            for vdir in sorted(root.iterdir()):
                if vdir.name.startswith("."):   # staging folders of a training job in progress
                    continue
                manifest = read_json(vdir / "manifest.json")
                if not vdir.is_dir() or not manifest:
                    continue
                c.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?)",
                          (module, vdir.name, manifest.get("model_name"), manifest.get("created_at"),
                           str(vdir), json.dumps(manifest)))
                metrics = read_json(vdir / "metrics.json")
                if metrics:
                    c.execute("INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?,?,?)",
                              (module, vdir.name, metrics.get("created_at"), metrics.get("model_name"),
                               metrics.get("baseline_name"), json.dumps(metrics)))


def versions(module: str) -> list[dict]:
    refresh()
    with tx() as c:
        rows = c.execute("SELECT * FROM model_versions WHERE module=? ORDER BY created_at DESC",
                         (module,)).fetchall()
    return [row_to_dict(r) for r in rows]


def active(module: str) -> dict | None:
    """Latest version; a version with metrics wins over a newer, unevaluated one."""
    vs = versions(module)
    if not vs:
        return None
    evaluated = {e["version"] for e in evaluations(module)}
    if module == "telemetry":
        # One checkpoint per channel: a partial retrain (2 channels) must never replace a version
        # that covers 80 channels. Most channels wins, then evaluated, then newest.
        def cover(v: dict) -> int:
            return len(list(Path(v["path"]).glob("*.pt")))
        return max(vs, key=lambda v: (cover(v), v["version"] in evaluated, v.get("created_at") or ""))
    # a full run beats a quick pipeline check; then evaluated beats unevaluated; then newest
    def kind_full(v: dict) -> bool:
        m = read_json(Path(v["path"]) / "manifest.json", {}) or {}
        return m.get("run_kind") != "quick" and not str(v["version"]).endswith("-quick")
    return max(vs, key=lambda v: (kind_full(v), v["version"] in evaluated, v.get("created_at") or ""))


def telemetry_dir_for(chan_id: str) -> Path | None:
    """The active telemetry version if it has this channel, else the newest version that does."""
    act = active("telemetry")
    if act and (Path(act["path"]) / f"{chan_id}.pt").exists():
        return Path(act["path"])
    for v in versions("telemetry"):
        if (Path(v["path"]) / f"{chan_id}.pt").exists():
            return Path(v["path"])
    return None


def active_dir(module: str) -> Path:
    v = active(module)
    if not v:
        raise FileNotFoundError(f"No trained {module} model yet. Train one from Data & Training (or this page). "
                                f"Run: setup:{module}")
    return Path(v["path"])


def version_dir(module: str, version: str) -> Path:
    """Directory of one specific model version (for pinned / repeated runs)."""
    for v in versions(module):
        if v["version"] == version:
            return Path(v["path"])
    raise FileNotFoundError(f"Model version {module}/{version} is no longer installed - cannot repeat with the same "
                            "model. Repeat with the current model instead.")


def evaluations(module: str | None = None) -> list[dict]:
    refresh()
    q = "SELECT * FROM evaluations"
    args: tuple = ()
    if module:
        q += " WHERE module=?"
        args = (module,)
    with tx() as c:
        rows = c.execute(q + " ORDER BY created_at DESC", args).fetchall()
    return [row_to_dict(r) for r in rows]
