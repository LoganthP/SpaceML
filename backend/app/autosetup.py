"""Automatic setup: download every dataset and train every model, with no terminal commands.

On each start of the API (and whenever someone presses "Run setup" in the dashboard) a single
background thread walks the steps below in order. Each step first checks whether its result is
already on disk and is skipped if so, so a restart only does the work that is still missing.
Steps run one at a time (they share the CPU), and a step that fails does not stop the others:
its reason is kept and shown in the dashboard with a Retry button.

Profiles (env SPACEML_AUTOSETUP):
    full   (default) data + every model, including the CPU Mars model (the longest step)
    light  data + the quick models (space weather, telemetry); no exoplanet/Mars training
    off    nothing runs automatically; the buttons still work
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

from spaceml.common.utils import now_iso, read_json, write_json
from spaceml.config import STORAGE_DIR

from . import jobs, registry

log = logging.getLogger("autosetup")
STATE = STORAGE_DIR / "autosetup.json"
MAX_AUTO_ATTEMPTS = 2           # a step that failed this often waits for a manual Retry
_lock = threading.Lock()
_thread: threading.Thread | None = None


@dataclass
class Step:
    id: str
    module: str
    title: str
    detail: str
    task: str
    needed: Callable[[], bool]
    params: Callable[[], dict] = dict
    profiles: tuple[str, ...] = ("full", "light")
    minutes: str = ""
    lane: str = "net"                  # "net" (downloads) and "cpu" (training) run side by side
    after: tuple[str, ...] = ()        # steps that must have finished first (in this run)


# ------------------------------------------------------------------ what is missing
def _sw_needed() -> bool:
    from spaceml.spaceweather import data as swd
    return registry.active("spaceweather") is None or 2024 not in swd.cached_years()


def _tele_data_needed() -> bool:
    from spaceml.telemetry import data
    return not data.is_downloaded()


def _tele_models_needed() -> bool:
    from pathlib import Path

    from spaceml.telemetry import data
    if not data.is_downloaded():
        return False
    act = registry.active("telemetry")
    have = {p.stem for p in Path(act["path"]).glob("*.pt")} if act else set()
    return len(set(data.load_channels()) - have) > 0


def _tele_params() -> dict:
    act = registry.active("telemetry")
    return {"mode": "fast", **({"version": act["version"]} if act else {})}


def _exo_needed() -> bool:
    return registry.active("exoplanet") is None


def _exo_data_needed() -> bool:
    if registry.active("exoplanet") is not None:
        return False
    from spaceml.exoplanet import data
    want = int(os.environ.get("SPACEML_SETUP_STARS", "300"))
    try:
        return not data.TCE_PATH.exists() or len({k for k, _ in data.cached_view_ids()}) < 0.8 * want
    except Exception:  # noqa: BLE001
        return True


def _exo_data_params() -> dict:
    return {"n_stars": int(os.environ.get("SPACEML_SETUP_STARS", "300")), "workers": 6}


def _exo_params() -> dict:
    return {"epochs": 60}


def _mars_samples_needed() -> bool:
    from spaceml.mars import data
    return not data.list_samples()


def _mars_model_needed() -> bool:
    return registry.active("mars") is None


def _mars_task() -> str:
    return "mars.install_weights" if os.environ.get("SPACEML_MARS_WEIGHTS_URL") else "mars.train_quick"


def _traverse_needed() -> bool:
    from spaceml.mars import traverse
    return not traverse.WAYPOINTS_PATH.exists()


STEPS: list[Step] = [
    # ---- network lane: downloads, one after another
    Step("telemetry_data", "telemetry", "Telemetry: full SMAP/MSL dataset",
         "81 channels with 105 expert-labeled anomalies (NASA S3 or GitHub mirror)", "telemetry.download",
         _tele_data_needed, minutes="1-3"),
    Step("mars_samples", "mars", "Mars: expert-labeled AI4Mars samples",
         "36 Curiosity NAVCAM images with expert terrain labels (Hugging Face)", "mars.samples",
         _mars_samples_needed, lambda: {"n": 36}, minutes="1-3"),
    Step("traverse", "mars", "Rover Traverse: Curiosity waypoints",
         "NASA/JPL mission-map layer, ~0.7 MB", "mars.traverse", _traverse_needed, minutes="<1"),
    Step("spaceweather", "spaceweather", "Space weather: OMNI2 data → Dst forecaster → May 2024 replay",
         "NASA SPDF, ~30 MB; trains in about a minute", "spaceweather.setup", _sw_needed, minutes="2-5"),
    Step("exoplanet_data", "exoplanet", "Exoplanets: Kepler catalog + light curves",
         "NASA Exoplanet Archive + MAST, 6 parallel downloads; resumable", "exoplanet.data", _exo_data_needed,
         _exo_data_params, profiles=("full",), minutes="15-40"),
    # ---- CPU lane: training, runs while the downloads above continue
    Step("telemetry_models", "telemetry", "Telemetry: train the remaining channel models + evaluate",
         "Fast mode, one process per CPU core; already trained channels are kept", "telemetry.train",
         _tele_models_needed, _tele_params, minutes="10-30", lane="cpu", after=("telemetry_data",)),
    Step("mars_model", "mars", "Mars: train the terrain segmentation model (CPU)",
         "DeepLabV3+ (ResNet-34) on streamed AI4Mars images; a GPU model can replace it later",
         "mars.train_quick", _mars_model_needed, lambda: {}, profiles=("full",), minutes="10-25", lane="cpu",
         after=("mars_samples", "traverse")),
    Step("exoplanet", "exoplanet", "Exoplanets: train the transit CNN + baseline",
         "Two-branch 1D CNN on the cached light curves", "exoplanet.train", _exo_needed, _exo_params,
         profiles=("full",), minutes="2-5", lane="cpu", after=("exoplanet_data",)),
]


def profile() -> str:
    p = os.environ.get("SPACEML_AUTOSETUP", "off").strip().lower()   # opt-in: nothing runs by itself
    return p if p in ("full", "light", "off") else "off"


# ------------------------------------------------------------------ state
def _state() -> dict:
    return read_json(STATE, {}) or {}


def _save(st: dict) -> None:
    write_json(STATE, st)


def _set_step(sid: str, **kw) -> None:
    with _lock:
        st = _state()
        st.setdefault("steps", {}).setdefault(sid, {}).update(kw)
        _save(st)


def status() -> dict:
    st = _state()
    prof = profile()
    out = []
    for s in STEPS:
        rec = (st.get("steps") or {}).get(s.id, {})
        try:
            needed = s.needed()
        except Exception as e:  # noqa: BLE001
            needed, rec = True, {**rec, "check_error": str(e)}
        state = rec.get("state") or ("done" if not needed else "pending")
        if not needed and state not in ("running",):
            state = "done"
        job = jobs.get(rec["job_id"]) if rec.get("job_id") else None
        if state == "running" and (not job or job["status"] not in ("queued", "running")) and not _running():
            state = "interrupted"
        out.append({"id": s.id, "module": s.module, "title": s.title, "detail": s.detail,
                    "task": _task_for(s), "minutes": s.minutes,
                    "included": prof in s.profiles, "state": state, "attempts": rec.get("attempts", 0),
                    "error": rec.get("error"), "job_id": rec.get("job_id"),
                    "progress": (job or {}).get("progress"), "message": (job or {}).get("message"),
                    "eta_seconds": _eta(job) if state == "running" else None,
                    "finished_at": rec.get("finished_at")})
    done = sum(x["state"] in ("done",) for x in out if x["included"])
    total = sum(x["included"] for x in out)
    return {"profile": prof, "running": _running(), "started_at": st.get("started_at"),
            "finished_at": st.get("finished_at"), "steps": out, "done": done, "total": total,
            "complete": done == total}


def _eta(job: dict | None) -> int | None:
    """Seconds left, extrapolated from progress so far (None until there is enough signal)."""
    if not job or not job.get("started_at") or not job.get("progress"):
        return None
    from datetime import datetime, timezone
    try:
        start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
    except ValueError:
        return None
    el = (datetime.now(timezone.utc) - start).total_seconds()
    p = float(job["progress"])
    if p < 0.03 or el < 20:
        return None
    return max(0, int(el / p * (1 - p)))


def _task_for(s: Step) -> str:
    return _mars_task() if s.id == "mars_model" else s.task


def _running() -> bool:
    return _thread is not None and _thread.is_alive()


# ------------------------------------------------------------------ runner
def start(reason: str = "manual", only: list[str] | None = None, force: bool = False) -> dict:
    """Start the setup thread (no-op if it is already running)."""
    global _thread
    if reason == "startup" and profile() == "off":
        return status()
    with _lock:
        if _running():
            return {"already_running": True}
        st = _state()
        if force or only:
            for sid in (only or [s.id for s in STEPS]):
                rec = (st.get("steps") or {}).get(sid)
                if rec:
                    rec["attempts"] = 0
                    rec.pop("error", None)
                    if rec.get("state") in ("failed", "source_unavailable", "interrupted", "cancelled"):
                        rec["state"] = "pending"
        st.update(started_at=now_iso(), finished_at=None, reason=reason)
        _save(st)
        _thread = threading.Thread(target=_run, args=(only, reason != "startup"), daemon=True, name="autosetup")
        _thread.start()
    return status()


def _run(only: list[str] | None, manual: bool) -> None:
    """Two lanes in parallel: downloads ("net") and training ("cpu"). A step waits for the steps
    in its `after` list; a failed prerequisite does not block forever (the step then reports why)."""
    # automatic start: the configured profile; a press of "Run setup" in the dashboard: everything
    prof = "full" if manual else profile()
    finished = {s.id: threading.Event() for s in STEPS}
    plan = [s for s in STEPS if (not only or s.id in only) and (only or prof in s.profiles)]
    for s in STEPS:
        if s not in plan:
            finished[s.id].set()

    def lane(name: str) -> None:
        for s in [x for x in plan if x.lane == name]:
            try:
                for dep in s.after:
                    while not finished[dep].wait(5):
                        _set_step(s.id, state="waiting", error=None)
                _maybe_run(s, manual)
            except Exception as e:  # noqa: BLE001
                log.exception("setup step %s crashed", s.id)
                _set_step(s.id, state="failed", error=str(e), finished_at=now_iso())
            finally:
                finished[s.id].set()

    threads = [threading.Thread(target=lane, args=(n,), daemon=True, name=f"autosetup-{n}") for n in ("net", "cpu")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with _lock:
        st = _state()
        st["finished_at"] = now_iso()
        _save(st)


def _maybe_run(s: Step, manual: bool) -> None:
    try:
        if not s.needed():
            _set_step(s.id, state="done")
            return
    except Exception as e:  # noqa: BLE001
        log.warning("setup check %s failed: %s", s.id, e)
    rec = (_state().get("steps") or {}).get(s.id, {})
    if not manual and rec.get("attempts", 0) >= MAX_AUTO_ATTEMPTS:
        return                             # keeps failing: wait for a manual Retry
    _run_step(s, rec.get("attempts", 0) + 1)


def _run_step(s: Step, attempt: int) -> None:
    task = _task_for(s)
    # wait (up to 6 h) while someone runs another job of the same module from the dashboard
    t0 = time.time()
    while jobs.busy(s.module) and time.time() - t0 < 6 * 3600:
        _set_step(s.id, state="waiting", error=None)
        time.sleep(5)
    params = s.params() if s.id != "mars_model" or task == "mars.train_quick" else {}
    try:
        job = jobs.submit(task, {**params, "origin": "auto-setup"})
    except Exception as e:  # noqa: BLE001
        _set_step(s.id, state="failed", error=str(e), attempts=attempt, finished_at=now_iso())
        return
    _set_step(s.id, state="running", job_id=job["id"], error=None, attempts=attempt, started_at=now_iso())
    while True:
        time.sleep(3)
        j = jobs.get(job["id"]) or {}
        if j.get("status") not in ("queued", "running"):
            break
    ok = j.get("status") == "succeeded"
    try:
        still = s.needed()
    except Exception:  # noqa: BLE001
        still = not ok
    if ok and still and s.id == "telemetry_models":
        still = False   # some channels can legitimately fail; the rest are trained
    state = "done" if ok and not still else (j.get("status") if j.get("status") in
                                               ("source_unavailable", "cancelled") else "failed")
    err = None if state == "done" else (j.get("error") or ("Finished, but the result is still missing"
                                                           if ok else "Failed"))
    _set_step(s.id, state=state, error=err, finished_at=now_iso())
    log.info("setup step %s -> %s %s", s.id, state, err or "")
