"""Starter pack: a small, versioned bundle of real data + reference runs for the first launch.

    python -m app.starter status                    # what the pack contains / what is installed
    python -m app.starter build                     # build starter/ from what is on this machine
    python -m app.starter build --with-network      # first download + train what is missing (needs internet)
    python -m app.starter replay --event 2024-05    # compute a replay run with the installed forecaster

On startup the API installs the pack (idempotent): files are copied into data/ and models/
only when missing and only after their sha256 matches pack.json; reference runs are added
to the run history with origin = "bundled" so the UI can label them apart from runs computed
on this machine.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from spaceml.common.fetch import sha256_file
from spaceml.common.utils import now_iso, read_json, write_json
from spaceml.config import DATA_DIR, MODELS_DIR, REPO_ROOT, RUNS_DIR, STORAGE_DIR

import os

STARTER_DIR = Path(os.environ.get("SPACEML_STARTER_DIR", REPO_ROOT / "starter"))
PACK_FILE = "pack.json"
MARKER = STORAGE_DIR / "starter_installed.json"
TELEMETRY_CHANNELS = ["A-5", "A-6", "A-8", "A-9", "C-1", "C-2", "D-12", "M-6", "P-1", "T-8", "T-9"]
REFERENCE_TELEMETRY = ["P-1", "T-8", "C-1"]
REPLAY_EVENT = "2024-05"


# ------------------------------------------------------------------ install
def pack() -> dict | None:
    return read_json(STARTER_DIR / PACK_FILE)


def install(verify: bool = True) -> dict:
    """Copy missing pack files into place and register bundled runs. Safe to call on every start."""
    p = pack()
    if not p:
        return {"installed": False, "reason": "no starter pack in this build"}
    marker = read_json(MARKER, {}) or {}
    if marker.get("pack_version") == p["pack_version"] and marker.get("complete"):
        _register_runs(p)     # cheap: only inserts missing rows (e.g. after the DB was reset)
        return {"installed": True, "pack_version": p["pack_version"], "copied": 0, "skipped": True}
    roots = {"data": DATA_DIR, "models": MODELS_DIR}
    copied, errors = 0, []
    for item in p["contents"]:
        for f in item["files"]:
            src = STARTER_DIR / f["path"]
            top, rel = f["path"].split("/", 1)
            if top not in roots:
                continue
            if f.get("archive"):            # a .tar.gz unpacked into data/ or models/
                if f.get("skip_if_exists") and (roots[top] / f["skip_if_exists"]).exists():
                    continue                # already prepared (possibly a newer source version) - never mix
                if not src.exists():
                    errors.append(f"missing in pack: {f['path']}")
                    continue
                if verify and sha256_file(src) != f["sha256"]:
                    errors.append(f"checksum mismatch: {f['path']}")
                    continue
                try:
                    copied += _extract(src, roots[top])
                except Exception as e:  # noqa: BLE001
                    errors.append(f"could not unpack {f['path']}: {e}")
                continue
            dest = roots[top] / rel
            if dest.exists():
                continue
            if not src.exists():
                errors.append(f"missing in pack: {f['path']}")
                continue
            if verify and sha256_file(src) != f["sha256"]:
                errors.append(f"checksum mismatch: {f['path']}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".installing")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
            copied += 1
    n_runs = _register_runs(p)
    write_json(MARKER, {"pack_version": p["pack_version"], "installed_at": now_iso(), "copied": copied,
                        "runs": n_runs, "errors": errors, "complete": not errors})
    return {"installed": True, "pack_version": p["pack_version"], "copied": copied, "runs": n_runs, "errors": errors}


def _extract(archive: Path, root: Path) -> int:
    """Unpack only members that are missing; refuse absolute paths and '..' (no path traversal)."""
    import tarfile
    n = 0
    root = root.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            target = (root / m.name).resolve()
            if root not in target.parents or Path(m.name).is_absolute():
                raise ValueError(f"unsafe path in archive: {m.name}")
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".installing")
            with tf.extractfile(m) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.replace(target)
            n += 1
    return n


def _register_runs(p: dict) -> int:
    from .db import tx
    added = 0
    for rid in p.get("reference_runs", []):
        src = STARTER_DIR / "runs" / rid
        row = read_json(src / "run.json")
        if not row:
            continue
        with tx() as c:
            if c.execute("SELECT 1 FROM runs WHERE id=?", (rid,)).fetchone():
                continue
        dest = RUNS_DIR / rid
        if not dest.exists():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns("run.json"))
        with tx() as c:
            c.execute("INSERT OR IGNORE INTO runs (id, module, input_ref, input_label, model_version, status, error, "
                      "summary, result_path, created_at, started_at, finished_at, duration_ms, origin) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, row["module"], json.dumps(row["input_ref"]), row["input_label"], row.get("model_version"),
                       "succeeded", None, json.dumps(row.get("summary") or {}), str(dest / "result.json"),
                       row["created_at"], row.get("started_at"), row.get("finished_at"), row.get("duration_ms"),
                       "bundled"))
        added += 1
    return added


def status() -> dict:
    p = pack()
    marker = read_json(MARKER, {}) or {}
    if not p:
        return {"present": False, "installed": marker}
    from spaceml.spaceweather import data as swd
    have_sw_model = (MODELS_DIR / "spaceweather").exists() and any(
        (d / "manifest.json").exists() for d in (MODELS_DIR / "spaceweather").iterdir() if not d.name.startswith("."))
    missing = []
    if not have_sw_model or 2024 not in swd.cached_years():
        missing.append({"module": "spaceweather", "what": "OMNI2 data / Dst forecaster",
                        "how": "comes with the release bundle, or downloads when you press Train forecaster on Data & Training"})
    return {"present": True, "pack_version": p["pack_version"], "created_at": p["created_at"],
            "contents": [{k: v for k, v in it.items() if k != "files"} | {"files": len(it["files"]),
                          "bytes": sum(f["bytes"] for f in it["files"])} for it in p["contents"]],
            "reference_runs": p.get("reference_runs", []), "installed": marker, "not_in_pack": missing}


# ------------------------------------------------------------------ build
def _add(contents: list, item: dict, files: list[tuple[Path, str]]) -> None:
    """files: (source path on this machine, path inside the pack)."""
    recs = []
    for src, rel in files:
        if not src.exists():
            continue
        dest = STARTER_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dest)
        recs.append({"path": rel, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)})
    if recs:
        contents.append({**item, "files": recs})


def _compute_run(module: str, inp: dict, label: str, version_dir: Path, origin: str = "bundled") -> dict:
    """Run a handler in-process (same code path as the dashboard) and return the run row."""
    import time

    from . import handlers  # noqa: F401
    from .worker import HANDLERS
    rid = ("ref-" if origin == "bundled" else "") + uuid.uuid4().hex[:10]
    run_dir = RUNS_DIR / f".{rid}.partial"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    t0, started = time.time(), now_iso()
    result, summary = HANDLERS[module](inp, version_dir, run_dir)
    result = {"run_id": rid, "module": module, "model_version": version_dir.name, **result}
    write_json(run_dir / "result.json", result)
    final = RUNS_DIR / rid
    run_dir.replace(final)
    return {"id": rid, "module": module, "input_ref": inp, "input_label": label, "model_version": version_dir.name,
            "summary": summary, "created_at": started, "started_at": started, "finished_at": now_iso(),
            "duration_ms": int((time.time() - t0) * 1000), "dir": final}


def build(with_network: bool = False) -> dict:
    from spaceml.spaceweather import data as swd
    from spaceml.telemetry import data as tdata

    from . import registry
    from .db import init_db
    init_db()
    registry.scan()
    STARTER_DIR.mkdir(parents=True, exist_ok=True)
    contents: list[dict] = []
    refs: list[str] = []

    # 1. telemetry: the channels with shipped models (real NASA arrays + expert labels)
    files = [(tdata.LABELS_PATH, "data/telemetry/labeled_anomalies.csv")]
    for ch in TELEMETRY_CHANNELS:
        files += [(tdata.TRAIN_DIR / f"{ch}.npy", f"data/telemetry/train/{ch}.npy"),
                  (tdata.TEST_DIR / f"{ch}.npy", f"data/telemetry/test/{ch}.npy")]
    _add(contents, {"id": "telemetry", "title": "SMAP & Curiosity telemetry - 11 channels with trained models",
                    "source": "NASA SMAP/MSL telemetry, expert-labeled anomalies (Hundman et al., KDD 2018)",
                    "source_url": "https://github.com/khundman/telemanom", "data_date": "SMAP/MSL mission data, labels 2018",
                    "license": "NASA public data (telemanom release, Apache-2.0 code)"}, files)

    tv = registry.active("telemetry")
    if tv:
        vdir = Path(tv["path"])
        _add(contents, {"id": "telemetry-model", "title": f"Telemetry LSTM models ({vdir.name}) + held-out metrics",
                        "source": "trained by this project on the NASA telemetry above",
                        "data_date": (read_json(vdir / "manifest.json", {}) or {}).get("created_at", ""),
                        "license": "project output"},
             [(f, f"models/telemetry/{vdir.name}/{f.name}") for f in sorted(vdir.iterdir())
              if f.is_file() and not f.name.startswith(".")])

    # 2. space weather: download + train when asked and possible
    if with_network:
        need = [y for y in swd.DEFAULT_YEARS if y not in swd.cached_years()]
        if need:
            swd.download_years(need)
    sw_model = registry.active("spaceweather")
    if with_network and not sw_model and all(y in swd.cached_years() for y in range(2010, 2018)):
        from spaceml.spaceweather import forecast
        forecast.train(list(range(2010, 2017)), [2017], forecast.DEFAULT_HORIZONS)
        registry.scan()
        sw_model = registry.active("spaceweather")
    if sw_model and 2024 in swd.cached_years():
        vdir = Path(sw_model["path"])
        files = [(swd.cache_path(2024), "data/spaceweather/omni2_2024.npz"),
                 (swd.SW_DIR / "omni2_2024.source.json", "data/spaceweather/omni2_2024.source.json")]
        files += [(f, f"models/spaceweather/{vdir.name}/{f.name}") for f in sorted(vdir.iterdir())
                  if f.is_file() and not f.name.startswith(".")]
        prov = read_json(swd.SW_DIR / "omni2_2024.source.json", {}) or {}
        _add(contents, {"id": "spaceweather", "title": "OMNI2 2024 hourly + Dst forecaster",
                        "source": "NASA/GSFC SPDF OMNIWeb OMNI2", "source_url": swd.BASE_URL.format(year=2024),
                        "data_date": f"retrieved {prov.get('retrieved_at', 'unknown')}, Dst {swd.dst_status(2024)}",
                        "license": "NASA public data - cite King & Papitashvili (2005)"}, files)

    # 3. reference runs (computed now, labelled 'bundled' after install)
    from . import worker  # noqa: F401
    shutil.rmtree(STARTER_DIR / "runs", ignore_errors=True)
    tv = registry.active("telemetry")
    if tv:
        for ch in REFERENCE_TELEMETRY:
            if tdata.has_channel(ch) and (Path(tv["path"]) / f"{ch}.pt").exists():
                sc = tdata.load_channels()[ch].spacecraft
                row = _compute_run("telemetry", {"chan_id": ch}, f"{ch} · {sc}", Path(tv["path"]))
                refs.append(_pack_run(row))
    if sw_model and 2024 in swd.cached_years():
        ev = next(e for e in swd.EVENTS if e["id"] == REPLAY_EVENT)
        row = _compute_run("spaceweather", {"event_id": REPLAY_EVENT}, f"Replay · {ev['name']}", Path(sw_model["path"]))
        refs.append(_pack_run(row))

    p = {"pack_version": now_iso()[:10].replace("-", "") + "-" + uuid.uuid4().hex[:4], "created_at": now_iso(),
         "created_with": {"app": "3.0.0", "python": sys.version.split()[0], "with_network": with_network},
         "contents": contents, "reference_runs": refs}
    write_json(STARTER_DIR / PACK_FILE, p)
    total = sum(f["bytes"] for it in contents for f in it["files"])
    print(f"Starter pack {p['pack_version']}: {len(contents)} data item(s), {len(refs)} reference run(s), "
          f"{total / 1e6:.1f} MB in {STARTER_DIR}")
    if not any(it["id"] == "spaceweather" for it in contents):
        print("  Space weather is NOT in the pack (no OMNI2 2024 data / forecaster on this machine). "
              "Run with --with-network on a machine with internet to include it.")
    return p


def _pack_run(row: dict) -> str:
    dest = STARTER_DIR / "runs" / row["id"]
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(row["dir"], dest)
    write_json(dest / "run.json", {k: v for k, v in row.items() if k != "dir"})
    shutil.rmtree(row["dir"], ignore_errors=True)   # the server registers it on install
    return row["id"]


def compute_replay(event_id: str) -> str:
    """Compute a replay run (origin 'computed') with the active forecaster and register it."""
    from spaceml.spaceweather import data as swd

    from . import registry
    from .db import init_db, tx
    init_db()
    registry.scan()
    ev = next((e for e in swd.EVENTS if e["id"] == event_id), None)
    if not ev:
        raise SystemExit(f"Unknown event {event_id}")
    vdir = registry.active_dir("spaceweather")
    print(f"::step::Replaying {ev['name']} with model {vdir.name}", flush=True)
    row = _compute_run("spaceweather", {"event_id": event_id}, f"Replay · {ev['name']}", vdir, origin="computed")
    with tx() as c:
        c.execute("INSERT INTO runs (id, module, input_ref, input_label, model_version, status, summary, result_path, "
                  "created_at, started_at, finished_at, duration_ms, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (row["id"], "spaceweather", json.dumps(row["input_ref"]), row["input_label"], row["model_version"],
                   "succeeded", json.dumps(row["summary"]), str(row["dir"] / "result.json"), row["created_at"],
                   row["started_at"], row["finished_at"], row["duration_ms"], "computed"))
    print(f"Replay run {row['id']}: {row['summary']['headline']}")
    return row["id"]


def main() -> None:
    p = argparse.ArgumentParser(description="Starter pack")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--with-network", action="store_true", help="download/train what is missing first")
    sub.add_parser("status")
    sub.add_parser("install")
    r = sub.add_parser("replay")
    r.add_argument("--event", default=REPLAY_EVENT)
    a = p.parse_args()
    if a.cmd == "build":
        build(a.with_network)
    elif a.cmd == "install":
        from .db import init_db
        init_db()
        print(install())
    elif a.cmd == "replay":
        compute_replay(a.event)
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
