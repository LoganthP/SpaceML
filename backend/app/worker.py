"""Background inference worker.

The API inserts a `queued` run and hands it to this thread pool. The worker loads
the active model for the module, runs inference, writes the result to
storage/runs/<id>/result.json and updates the run row. The frontend polls the run.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from spaceml.common.utils import now_iso, write_json
from spaceml.config import RUNS_DIR

from . import registry
from .db import row_to_dict, tx

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")

# module -> callable(input_ref, version_dir, run_dir) -> (result, summary)
MODEL_OPTIONAL = {"exoplanet", "missions"}
HANDLERS: dict[str, Callable[[dict, Any, Any], tuple[dict, dict]]] = {}


def register(module: str):
    def deco(fn):
        HANDLERS[module] = fn
        return fn
    return deco


def submit(module: str, input_ref: dict, input_label: str, version: str | None = None,
           repeat_of: str | None = None) -> dict:
    """Queue a run. `version` pins a model version (used by Repeat run); default = active model."""
    if module not in HANDLERS:
        raise ValueError(f"Unknown module {module}")
    if module == "missions":
        from spaceml.missions import models as mmodels
        if version:
            mmodels.version_dir(input_ref["mission"], version)
        act = mmodels.active(input_ref["mission"])
        shown = version or (act["version"] if act else None)
    elif version and module in registry.MODULES:
        registry.version_dir(module, version)       # raises if it is gone
        shown = version
    else:
        active = registry.active(module) if module in registry.MODULES else None
        shown = active["version"] if active else None
    run_id = uuid.uuid4().hex[:12]
    with tx() as c:
        c.execute("INSERT INTO runs (id, module, input_ref, input_label, model_version, status, created_at, "
                  "origin, repeat_of, pinned_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (run_id, module, json.dumps(input_ref), input_label, shown, "queued", now_iso(),
                   "computed", repeat_of, version))
    _pool.submit(_execute, run_id, module, input_ref, version)
    return get(run_id)


def _version_dir(module: str, input_ref: dict, version: str | None = None):
    if module == "missions":
        from pathlib import Path

        from spaceml.missions import models as mmodels
        if version:
            return mmodels.version_dir(input_ref["mission"], version)
        act = mmodels.active(input_ref["mission"])
        return Path(act["path"]) if act else None
    if version and module in registry.MODULES:
        return registry.version_dir(module, version)
    if module == "telemetry" and input_ref.get("chan_id"):
        d = registry.telemetry_dir_for(str(input_ref["chan_id"]))
        if d is not None:
            return d
    if module == "workbench":
        from spaceml.workbench.analyses import ANALYSES
        need = (ANALYSES.get(input_ref.get("analysis"), {}) or {}).get("requires_model")
        if need == "mars (optional)":
            need = "mars"
            try:
                return registry.active_dir(need)
            except FileNotFoundError:
                return None
        if need not in registry.MODULES:
            return None
        try:
            return registry.active_dir(need)
        except FileNotFoundError:
            return None
    try:
        return registry.active_dir(module)
    except FileNotFoundError:
        if module not in MODEL_OPTIONAL:
            raise
        return None  # exoplanet views/3D/search still work without a trained CNN


def _execute(run_id: str, module: str, input_ref: dict, version: str | None = None) -> None:
    t0 = time.time()
    with tx() as c:
        c.execute("UPDATE runs SET status='running', started_at=? WHERE id=?", (now_iso(), run_id))
    run_dir = RUNS_DIR / run_id
    stage = RUNS_DIR / f".{run_id}.partial"          # outputs are published only when complete
    import shutil
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        version_dir = _version_dir(module, input_ref, version)
        result, summary = HANDLERS[module](input_ref, version_dir, stage)
        vname = version_dir.name if version_dir else None
        result = {"run_id": run_id, "module": module, "model_version": vname, **result}
        write_json(stage / "result.json", result)
        shutil.rmtree(run_dir, ignore_errors=True)
        stage.replace(run_dir)
        with tx() as c:
            c.execute("UPDATE runs SET status='succeeded', finished_at=?, duration_ms=?, summary=?, "
                      "result_path=?, model_version=? WHERE id=?",
                      (now_iso(), int((time.time() - t0) * 1000), json.dumps(summary),
                       str(run_dir / "result.json"), vname, run_id))
    except Exception as e:  # noqa: BLE001 - surface every failure to the user
        traceback.print_exc()
        shutil.rmtree(stage, ignore_errors=True)
        status, msg = classify_error(e)
        with tx() as c:
            c.execute("UPDATE runs SET status=?, finished_at=?, duration_ms=?, error=? WHERE id=?",
                      (status, now_iso(), int((time.time() - t0) * 1000), msg, run_id))


def classify_error(e: BaseException) -> tuple[str, str]:
    """Map an exception to a run status the UI can tell apart from a clean result.

    source_unavailable  - an upstream archive could not be reached
    insufficient_data   - the input does not contain enough valid data
    failed              - anything else (a bug or bad input)
    """
    from spaceml.common.fetch import SourceUnavailable
    msg = str(e) if isinstance(e, (FileNotFoundError, ValueError, RuntimeError, LookupError)) else f"{type(e).__name__}: {e}"
    if isinstance(e, SourceUnavailable) or any(k in msg for k in ("urlopen", "Connection refused", "timed out",
                                                                   "Name or service not known")):
        if not isinstance(e, SourceUnavailable):
            msg += " - the NASA archive could not be reached; check your internet connection and try again"
        return "source_unavailable", msg
    if msg.startswith("insufficient_data") or type(e).__name__ == "InsufficientData":
        return "insufficient_data", msg.removeprefix("insufficient_data: ")
    return "failed", msg


def get(run_id: str) -> dict | None:
    with tx() as c:
        return row_to_dict(c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())


def list_runs(module: str | None = None, limit: int = 100) -> list[dict]:
    q, args = "SELECT * FROM runs", []
    if module:
        q += " WHERE module=?"
        args.append(module)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with tx() as c:
        return [row_to_dict(r) for r in c.execute(q, args).fetchall()]


def delete(run_id: str) -> None:
    import shutil
    with tx() as c:
        c.execute("DELETE FROM runs WHERE id=?", (run_id,))
        c.execute("DELETE FROM reviews WHERE run_id=?", (run_id,))
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)
