"""Automatic mission data install at startup (SMC_AUTO_INSTALL=on, the default; off disables it).

When the machine is online, every mission whose data is missing and whose source is small and public (see
spaceml.missions.install.auto_source) gets an "Install data" job queued in the background. Jobs run one at a
time through the normal queue, so the dashboard stays usable and progress shows in the job tray. Failed
attempts are retried at most every 6 hours. Large sources (ESA-AD, several GB) are never started automatically.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request

from spaceml.common.utils import read_json, write_json
from spaceml.config import STORAGE_DIR

log = logging.getLogger("autoinstall")
STATE = STORAGE_DIR / "auto_install.json"
RETRY_S = 6 * 3600
PROBE = ("https://mast.stsci.edu/", "https://zenodo.org/")


def enabled() -> bool:
    return os.environ.get("SMC_AUTO_INSTALL", "on").strip().lower() not in ("off", "0", "false", "no")


def online(timeout: float = 5.0) -> bool:
    for url in PROBE:
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True                          # the server answered: we are online
        except Exception:  # noqa: BLE001
            continue
    return False


def plan() -> list[str]:
    """Missions that would be installed automatically right now (ignoring connectivity)."""
    from spaceml.missions import catalog, store
    from spaceml.missions.install import auto_source
    state = read_json(STATE, {}) if STATE.exists() else {}
    now = time.time()
    try:                                     # missions the mounted release bundle provides are installed by it
        from spaceml import release
        bundled = {d["id"][len("mission-"):] for d in (release.load_manifest() or {}).get("datasets", [])
                   if d["id"].startswith("mission-")}
    except Exception:  # noqa: BLE001
        bundled = set()
    out = []
    for mid in catalog.MISSIONS:
        src = auto_source(mid)
        if not src or not src.get("automatic") or store.installed(mid) or mid in bundled:
            continue
        if now - float(state.get(mid, 0)) < RETRY_S:
            continue
        out.append(mid)
    return out


def run() -> list[str]:
    if not enabled():
        return []
    todo = plan()
    if not todo:
        return []
    if not online():
        log.info("auto-install skipped: no internet connection")
        return []
    from . import jobs
    state = read_json(STATE, {}) if STATE.exists() else {}
    started = []
    for mid in todo:
        try:
            jobs.submit(f"missions.{mid}.install", {})
            state[mid] = time.time()
            started.append(mid)
        except Exception:  # noqa: BLE001 - never block startup
            log.warning("auto-install of %s could not be queued", mid, exc_info=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    write_json(STATE, state)
    if started:
        log.info("auto-install queued: %s", ", ".join(started))
    return started
