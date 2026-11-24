"""Background jobs: data downloads, training and evaluation launched from the dashboard.

Every job runs one or more whitelisted `python -m spaceml...` commands as a child
process (the exact same commands you can type in a terminal), streams the output to
storage/jobs/<id>.log and keeps status/progress in SQLite. Heavy jobs share a bounded queue, while independent mission datasets use separate slots. Model versions are rescanned when a job finishes.
"""
from __future__ import annotations

import json
import re
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from spaceml.common.utils import now_iso
from spaceml.config import STORAGE_DIR

from . import registry
from .manifest import APP_VERSION
from spaceml import config as _mcfg
from spaceml.missions import catalog as _mcat
from .db import row_to_dict, tx

BACKEND_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = STORAGE_DIR / "jobs"
_procs: dict[str, subprocess.Popen] = {}
_cancelled: set[str] = set()
_lock = threading.Lock()


@dataclass
class Task:
    module: str
    title: str
    build: Callable[[dict], list[list[str]]]   # params -> list of argv (after `python -m`)
    params: dict                               # name -> (type, default, min, max)
    heavy: bool = False                        # CPU-heavy training: one at a time, lower priority


HEAVY_SLOTS = max(1, int(os.environ.get("SPACEML_HEAVY_JOBS", "1")))
_heavy = threading.Semaphore(HEAVY_SLOTS)
_TB_EXC = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit))(: .*)?$")


def _yearspec(v, default: str) -> str:
    """Year list such as '2010-2016' or '2015 2017' - digits, dashes, commas and spaces only."""
    v = re.sub(r"[^0-9,\- ]", "", str(v or default))[:60].strip()
    return v or default


def _new_version(tag: str) -> str:
    from spaceml.common.utils import new_version
    return f"{new_version()}-{tag}"


def _url(v) -> str:
    v = str(v or os.environ.get("SPACEML_MARS_WEIGHTS_URL", "")).strip()
    if not re.fullmatch(r"https://[\w.\-]+(:\d+)?/[\w./\-%~+=?&]*", v):
        raise ValueError("Give an https:// link to a .zip that contains model.pt and manifest.json")
    return v


_SAFE = re.compile(r"^[A-Za-z0-9_.:\-]{1,120}$")


def _install_cmd(mission: str, p: dict) -> list[str]:
    """Install = download from the official source, or import files the user uploaded (validated paths only)."""
    cmd = ["spaceml.missions.install", mission]
    if p.get("source"):
        src = (_mcfg.DATA_DIR / "missions-source" / mission / str(p["source"])).resolve()
        if not str(src).startswith(str((_mcfg.DATA_DIR / "missions-source" / mission).resolve())) or not src.is_dir():
            raise ValueError("unknown upload folder")
        cmd += ["--source", str(src)]
    else:
        cmd.append("--download")
    files = p.get("files") or []
    if isinstance(files, str):
        files = files.replace(",", " ").split()
    for f in files[:200]:
        if not _SAFE.match(f):
            raise ValueError(f"invalid file name {f!r}")
    if files:
        cmd += ["--files", *files]
    return cmd


def _int(p, k, default, lo, hi):
    try:
        v = int(p.get(k, default))
    except (TypeError, ValueError):
        raise ValueError(f"'{k}' must be a whole number")
    if not lo <= v <= hi:
        raise ValueError(f"'{k}' must be between {lo} and {hi}")
    return v


def _channels(p) -> list[str]:
    import re
    raw = p.get("channels") or []
    if isinstance(raw, str):
        raw = [c for c in re.split(r"[\s,]+", raw) if c]
    for c in raw:
        if not re.fullmatch(r"[A-Z]-\d{1,2}", c):
            raise ValueError(f"Invalid channel id {c!r}")
    return list(raw)


def _tele_version(p) -> str:
    import re
    v = p.get("version") or ((registry.active("telemetry") or {}).get("version")) or "v1"
    if not re.fullmatch(r"[\w.\-]{1,40}", v):
        raise ValueError("Invalid version name")
    return v


def _tele_train(p):
    ver = _tele_version(p)
    chans = _channels(p)
    epochs = _int(p, "epochs", 35, 3, 200)
    full = p.get("mode") == "full"
    train = ["spaceml.telemetry.train", "--version", ver, "--epochs", str(epochs)] if full else \
        ["spaceml.telemetry.train", "--version", ver, "--fast"]
    train += ["--channels", *chans] if chans else []
    return [train, ["spaceml.telemetry.evaluate", "--version", ver]]


def _spaceweather_train(p):
    from spaceml.spaceweather import data
    from spaceml.spaceweather.forecast import _years

    train = _yearspec(p.get("train_years"), "2010-2016")
    test = _yearspec(p.get("test_years"), "2017")
    missing = sorted((set(_years(train)) | set(_years(test))) - set(data.cached_years()))
    if missing:
        raise ValueError(f"OMNI2 years {missing} are missing. Use Download on the OMNI2 card or install a release bundle.")
    commands = [["spaceml.spaceweather.forecast", "--train-years", train, "--test-years", test]]
    if 2024 in data.cached_years():
        commands.append(["app.starter", "replay", "--event", "2024-05"])
    return commands


def _mars_train(p):
    quick = p.get("mode", "quick") != "full"
    from spaceml.mars import local

    counts = local.validate().get("splits", {})
    if counts.get("train", {}).get("usable", 0) < 16:
        raise ValueError("Mars needs at least 16 usable training pairs. Install the AI4Mars dataset or repair it from Data & Training.")
    cmd = ["spaceml.mars.train"] + (["--quick"] if quick else [])
    if p.get("epochs"):
        cmd += ["--epochs", str(_int(p, "epochs", 3, 1, 100))]
    if p.get("max_train"):
        cmd += ["--max-train", str(_int(p, "max_train", 300, 20, 100000))]
    commands = [cmd]
    if any(count.get("usable", 0) for split, count in counts.items() if split.startswith("test")):
        commands.append(["spaceml.mars.evaluate", "--latest"] + (["--limit", "120"] if quick else []))
    return commands


def _exo_train(p):
    from spaceml.exoplanet import data

    try:
        rows = data.cached_training_rows()
    except FileNotFoundError:
        raise ValueError("Install the Kepler catalog and folded light curves from Data & Training first.") from None
    if min(int(rows.planet.sum()), int((rows.planet == 0).sum())) < data.MIN_PER_CLASS:
        raise ValueError(f"Kepler needs at least {data.MIN_PER_CLASS} folded signals per class. Download more light curves first.")
    quick = p.get("mode", "full") == "quick"
    cmd = ["spaceml.exoplanet.train", "--epochs", str(_int(p, "epochs", 40 if quick else 60, 3, 300))]
    return [cmd + (["--max-stars", str(_int(p, "max_stars", 80, 20, 5000))] if quick else [])]


TASKS: dict[str, Task] = {
    # ---------------- telemetry
    "telemetry.download": Task("telemetry", "Download SMAP/MSL telemetry",
                               lambda p: [["spaceml.telemetry.data", "download"]], {}),
    "telemetry.train": Task("telemetry", "Train + evaluate telemetry LSTMs", _tele_train,
                            {"channels": "list", "epochs": 35, "version": "str", "mode": "fast"}, heavy=True),
    "telemetry.retrain": Task("telemetry", "Retrain all channels from scratch (new version)",
                              lambda p: (lambda v: [["spaceml.telemetry.train", "--version", v, "--fast"],
                                                    ["spaceml.telemetry.evaluate", "--version", v]])(_new_version("retrain")),
                              {}, heavy=True),
    "telemetry.evaluate": Task("telemetry", "Evaluate telemetry models",
                               lambda p: [["spaceml.telemetry.evaluate", "--version", _tele_version(p)]], {}, heavy=True),
    # ---------------- exoplanet
    "exoplanet.catalog": Task("exoplanet", "Download Kepler DR24 catalog",
                              lambda p: [["spaceml.exoplanet.data", "tce"]], {}),
    "exoplanet.lightcurves": Task("exoplanet", "Download & fold Kepler light curves",
                                  lambda p: [["spaceml.exoplanet.data", "lightcurves",
                                              "--n-stars", str(_int(p, "n_stars", 300, 10, 5000)),
                                              "--workers", str(_int(p, "workers", 6, 1, 12)),
                                              "--max-quarters", str(_int(p, "max_quarters", 8, 0, 17))]],
                                  {"n_stars": 300, "workers": 6, "max_quarters": 8}),
    "exoplanet.data": Task("exoplanet", "Download the Kepler catalog + light curves",
                           lambda p: [["spaceml.exoplanet.data", "tce", "--if-missing"],
                                      ["spaceml.exoplanet.data", "lightcurves",
                                       "--n-stars", str(_int(p, "n_stars", 300, 10, 5000)),
                                       "--workers", str(_int(p, "workers", 6, 1, 12))]],
                           {"n_stars": 300, "workers": 6}),
    "exoplanet.train": Task("exoplanet", "Train exoplanet CNN + baseline", _exo_train,
                            {"mode": "full", "epochs": 60, "max_stars": 80}, heavy=True),
    "exoplanet.setup": Task("exoplanet", "Full exoplanet setup (catalog → light curves → train)",
                            lambda p: [["spaceml.exoplanet.data", "tce", "--if-missing"],
                                       ["spaceml.exoplanet.data", "lightcurves",
                                        "--n-stars", str(_int(p, "n_stars", 300, 10, 5000)),
                                        "--workers", str(_int(p, "workers", 4, 1, 12))],
                                       ["spaceml.exoplanet.train", "--epochs", str(_int(p, "epochs", 60, 3, 300))]],
                            {"n_stars": 300, "workers": 4, "epochs": 60}, heavy=True),
    # ---------------- mars
    "mars.dataset": Task("mars", "Download + validate AI4Mars image/mask pairs",
                         lambda p: [["spaceml.mars.local", "build", "--train", str(p.get("train") or "3000"),
                                     "--test", "all"],
                                    ["spaceml.mars.data", "samples", "--n", "36"]],
                         {"train": "3000"}),
    "mars.samples": Task("mars", "Prepare curated AI4Mars samples",
                         lambda p: [["spaceml.mars.data", "samples", "--n", str(_int(p, "n", 36, 4, 200))]], {"n": 36}),
    "mars.train": Task("mars", "Train + evaluate the Mars terrain model", _mars_train,
                       {"mode": "quick", "epochs": 0, "max_train": 0}, heavy=True),
    "mars.train_quick": Task("mars", "Quick Mars training run (pipeline check)",
                             lambda p: _mars_train({**p, "mode": "quick"}), {}, heavy=True),
    "mars.resume": Task("mars", "Resume the interrupted Mars training run",
                        lambda p: [["spaceml.mars.train", "--resume"], ["spaceml.mars.evaluate", "--latest"]], {},
                        heavy=True),
    "mars.evaluate": Task("mars", "Evaluate Mars model",
                          lambda p: [["spaceml.mars.evaluate", "--latest",
                                      "--limit", str(_int(p, "limit", 0, 0, 5000))]], {"limit": 0}, heavy=True),
    "mars.install_weights": Task("mars", "Install a trained Mars model from a URL",
                                 lambda p: [["spaceml.mars.install", "--url", _url(p.get("url"))]], {"url": "str"}),
    "mars.traverse": Task("mars", "Download Curiosity traverse waypoints (NASA/JPL)",
                          lambda p: [["spaceml.mars.traverse", "download"]], {}),
    # ---------------- space weather
    "spaceweather.download": Task("spaceweather", "Download OMNIWeb solar-wind data",
                                  lambda p: [["spaceml.spaceweather.data", "download", "--years",
                                              *[str(int(y)) for y in (p.get("years") or [2003, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2024])
                                                if 1963 <= int(y) <= 2030][:40]]], {"years": "list"}),
    "spaceweather.train": Task("spaceweather", "Train Dst forecasters (vs persistence)",
                               _spaceweather_train,
                               {"train_years": "2010-2016", "test_years": "2017"}, heavy=True),
    "spaceweather.setup": Task("spaceweather", "Space weather setup: download OMNI2 → train → replay May 2024",
                               lambda p: [["spaceml.spaceweather.data", "download", "--years",
                                           "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2024"],
                                          ["spaceml.spaceweather.forecast", "--train-years", "2010-2016",
                                           "--test-years", "2017", "--if-missing"],
                                          ["app.starter", "replay", "--event", "2024-05"]], {}, heavy=True),
    "spaceweather.replay": Task("spaceweather", "Replay a storm with the installed forecaster",
                                lambda p: [["app.starter", "replay", "--event",
                                            re.sub(r"[^\w\-]", "", str(p.get("event_id") or "2024-05"))[:16]]],
                                {"event_id": "2024-05"}),
    # ---------------- multi-mission telemetry (one task per mission so status/retry stay per mission)
    **{f"missions.{m}.train": Task("missions", f"Train {_mcat.MISSIONS[m]['name']} detector",
                                   (lambda m: lambda p: [["spaceml.missions.train", m]
                                                         + ([] if p.get("mode") == "full" else ["--quick"])])(m),
                                   {"mode": "quick"}, heavy=True)
       for m in _mcat.ORDER if _mcat.MISSIONS[m].get("train")},
    "missions.opssat.download": Task("missions", "Download OPSSAT-AD v2 from Zenodo and prepare it",
                                     lambda p: [["spaceml.missions.install", "opssat", "--download"]], {}),
    **{f"missions.{m}.install": Task("missions", f"Install {_mcat.MISSIONS[m]['name']} data",
                                     (lambda m: lambda p: [_install_cmd(m, p)])(m), {})
       for m in _mcat.ORDER},
    # ---------------- workbench
    "workbench.samples": Task("workbench", "Download before/after satellite scenes (NASA GIBS)",
                              lambda p: [["spaceml.workbench.samples", "imagery", "--out", "data"]], {}),
    # ---------------- release bundle
    "release.install": Task("release", "Install the release bundle (verify checksums, copy datasets)",
                            lambda p: [["spaceml.release", "install"]], {}),
    "release.build": Task("release", "Package this installation's datasets as a release bundle",
                          lambda p: [["spaceml.release", "build", "--from-data"]], {}),
}
HEAVY_TASKS = [k for k, t in TASKS.items() if t.heavy]

def init() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with tx() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, task TEXT NOT NULL, module TEXT NOT NULL, title TEXT NOT NULL,
            params TEXT, status TEXT NOT NULL, progress REAL, step TEXT, message TEXT, error TEXT,
            command TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, returncode INTEGER)""")
        cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
        for col in ("last_progress_at", "waiting_for", "app_version", "dismissed"):
            if col not in cols:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
        c.execute("UPDATE jobs SET status='interrupted', error='Interrupted: the server stopped while this job ran. "
                  "Finished parts are kept (resumable steps continue where they stopped); press Retry.' "
                  "WHERE status IN ('queued','running')")


def _update(job_id: str, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with tx() as c:
        c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), job_id))


def get(job_id: str) -> dict | None:
    with tx() as c:
        return row_to_dict(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def list_jobs(limit: int = 30) -> list[dict]:
    """Recent jobs. Jobs of tasks that no longer exist (missions removed in 3.1.8) stay in the database but are
    not listed - they cannot be retried and would only clutter the job tray."""
    with tx() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit * 3,)).fetchall()
    out = []
    for r in rows:
        if r["task"] not in TASKS:
            continue
        if len(out) >= limit:
            break
        d = row_to_dict(r)
        d["params"] = json.loads(d["params"]) if isinstance(d.get("params"), str) else d.get("params")
        out.append(d)
    return out


def log_tail(job_id: str, lines: int = 200) -> list[str]:
    p = JOBS_DIR / f"{job_id}.log"
    if not p.exists():
        return []
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 64_000))
        text = f.read().decode("utf-8", "replace")
    out = [l for l in text.splitlines() if not l.startswith("::progress::")]
    return out[-lines:]


def _scrub(line: str) -> str:
    """Hook for keeping secrets out of job logs (no mission uses credentials at the moment)."""
    return line


def submit(task_name: str, params: dict) -> dict:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name}")
    task = TASKS[task_name]
    params = dict(params or {})
    job_id = uuid.uuid4().hex[:12]
    secret = params.pop("token", None)
    build_params = dict(params)
    if secret:
        sdir = STORAGE_DIR / "secrets"
        sdir.mkdir(parents=True, exist_ok=True)
        tf = sdir / f"{job_id}.token"
        tf.write_text(str(secret))
        os.chmod(tf, 0o600)
        build_params["token_file"] = str(tf)
    try:
        commands = task.build(build_params)  # validates params
    except Exception:
        if secret:
            (STORAGE_DIR / "secrets" / f"{job_id}.token").unlink(missing_ok=True)
        raise
    shown = " && ".join("python -m " + " ".join(cmd) for cmd in commands)
    with tx() as c:
        active = c.execute("SELECT * FROM jobs WHERE module=? AND status IN ('queued','running')",
                           (task.module,)).fetchall()
        for job in active:
            if job["task"] == task_name:
                if secret:      # the running job has its own token; never leave a second copy on disk
                    (STORAGE_DIR / "secrets" / f"{job_id}.token").unlink(missing_ok=True)
                return {**row_to_dict(job), "reused": True}
            if task.module == "missions" and task_name.split(".")[1] != job["task"].split(".")[1]:
                continue
            if "mars.traverse" in (task_name, job["task"]):
                continue  # independent source data; the heavy-job queue still bounds CPU use
            raise RuntimeError(f"'{job['title']}' is still active for this module. Wait for it or cancel it.")
        c.execute("INSERT INTO jobs (id, task, module, title, params, status, progress, command, created_at, app_version) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (job_id, task_name, task.module, task.title, json.dumps(params or {}), "queued", 0.0, shown, now_iso(),
                   APP_VERSION))
    threading.Thread(target=_run, args=(job_id, commands), daemon=True, name=f"job-{job_id}").start()
    return get(job_id)


def dismiss(job_id: str) -> dict | None:
    """Hide a finished job's error box (the job stays in the history)."""
    _update(job_id, dismissed="1")
    return get(job_id)


def _wait_for_slot(job_id: str) -> bool:
    """Heavy jobs run one at a time. While waiting, the job says exactly what it waits for."""
    first = True
    while not _heavy.acquire(blocking=False) if first else not _heavy.acquire(timeout=2):
        first = False
        if job_id in _cancelled:
            return False
        with tx() as c:
            r = c.execute("SELECT j.title FROM jobs j WHERE j.status='running' AND j.task IN (%s) LIMIT 1"
                          % ",".join("?" * len(HEAVY_TASKS)), tuple(HEAVY_TASKS)).fetchone()
        _update(job_id, waiting_for=(r["title"] if r else "another training job"),
                message=f"Queued: waits for '{r['title'] if r else 'another training job'}' to finish (one training job "
                        "at a time keeps the dashboard responsive)")
    return True


def _run(job_id: str, commands: list[list[str]]) -> None:
    job = get(job_id) or {}
    heavy = TASKS.get(job.get("task", ""), Task("", "", lambda p: [], {})).heavy
    if heavy and not _wait_for_slot(job_id):
        _update(job_id, status="cancelled", finished_at=now_iso(), error="Cancelled by user")
        return
    try:
        _run_locked(job_id, commands, heavy)
    finally:
        if heavy:
            _heavy.release()


def _final_error(lines: list[str]) -> str | None:
    """The exception message of the last traceback (not the source line that raised it)."""
    for line in reversed(lines):
        m = _TB_EXC.match(line.strip())
        if m:
            return line.strip()
    return None


def _run_locked(job_id: str, commands: list[list[str]], heavy: bool) -> None:
    log_path = JOBS_DIR / f"{job_id}.log"
    _update(job_id, status="running", started_at=now_iso(), waiting_for=None, last_progress_at=now_iso())
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if heavy:
        from spaceml.common.parallel import train_cpus
        env["SPACEML_TRAIN_CPUS"] = str(train_cpus())
    rc, last_error = 0, None
    recent: list[str] = []
    with open(log_path, "a", encoding="utf-8") as log:
        for n, cmd in enumerate(commands, 1):
            if job_id in _cancelled:
                break
            argv = [sys.executable, "-u", "-m", *cmd]
            log.write(f"$ python -m {' '.join(cmd)}\n")
            log.flush()
            _update(job_id, step=f"Step {n}/{len(commands)}: {cmd[0].split('.')[-1]} {' '.join(cmd[1:2])}".strip(),
                    progress=0.0)
            kwargs = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
            if heavy and os.name != "nt":
                kwargs["preexec_fn"] = lambda: os.nice(10)   # training yields the CPU to the dashboard
            proc = subprocess.Popen(argv, cwd=BACKEND_DIR, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                    bufsize=1, **kwargs)
            with _lock:
                _procs[job_id] = proc
            last_flush = time.time()
            activity = [time.time()]
            stalled: list[str] = []
            watch_stop = threading.Event()

            def watchdog(p=proc, act=activity, st=stalled, stop=watch_stop) -> None:
                limit = STALL_MINUTES * 60
                while not stop.wait(20):
                    if p.poll() is not None:
                        return
                    if time.time() - act[0] > limit:
                        st.append(f"Stalled: no progress for {STALL_MINUTES} minutes, so the job was stopped. "
                                  "Press Retry (finished parts are kept).")
                        _kill_tree(p)
                        return

            threading.Thread(target=watchdog, daemon=True, name=f"watchdog-{job_id}").start()
            for line in proc.stdout:  # type: ignore[union-attr]
                activity[0] = time.time()
                line = _scrub(line.rstrip("\n"))
                if line.startswith("::progress::"):
                    try:
                        _, frac, msg = line.split("::", 3)[1:]
                        done, total = frac.split("/")
                        overall = (n - 1 + float(done) / max(float(total), 1)) / len(commands)
                        _update(job_id, progress=round(overall, 4), message=msg[:300], last_progress_at=now_iso())
                    except ValueError:
                        pass
                    continue
                if line.startswith("::step::"):
                    _update(job_id, message=line[8:][:300], last_progress_at=now_iso())
                elif line.startswith("::error::"):
                    last_error = line[9:][:2000]
                recent.append(line)
                del recent[:-80]
                log.write(line + "\n")
                if time.time() - last_flush > 0.5:
                    log.flush()
                    last_flush = time.time()
            rc = proc.wait()
            watch_stop.set()
            if rc != 0 and not last_error:
                last_error = _final_error(recent)
            if stalled:
                last_error = stalled[0]
            log.flush()
            with _lock:
                _procs.pop(job_id, None)
            if rc != 0:
                break
    try:
        registry.scan()
    except Exception:  # noqa: BLE001
        pass
    if job_id in _cancelled:
        _update(job_id, status="cancelled", finished_at=now_iso(), returncode=rc, error="Cancelled by user")
    elif rc != 0 and last_error and last_error.startswith("source_unavailable"):
        _update(job_id, status="source_unavailable", finished_at=now_iso(), returncode=rc,
                error=last_error.removeprefix("source_unavailable: "))
    elif rc == 0:
        _update(job_id, status="succeeded", finished_at=now_iso(), returncode=0, progress=1.0)
    elif rc < 0 and not last_error:      # killed by a signal: server/container stopped, out of memory, ...
        import signal as _sig
        name = _sig.Signals(-rc).name if -rc in {s.value for s in _sig.Signals} else str(-rc)
        _update(job_id, status="interrupted", finished_at=now_iso(), returncode=rc,
                error=f"Stopped ({name}) - the app restarted or the system ran out of memory. Finished parts are kept; "
                      "press Retry" + (" or Resume" if (job_id and "mars" in (get(job_id) or {}).get("task", "")) else "") + ".")
    else:
        tail = [l for l in log_tail(job_id, 30) if l.strip() and not l.startswith("::")]
        _update(job_id, status="failed", finished_at=now_iso(), returncode=rc,
                error=last_error or (tail[-1] if tail else f"Exited with code {rc}"))


STALL_MINUTES = max(5, int(os.environ.get("SPACEML_JOB_STALL_MINUTES", "30")))


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name != "nt":
            import signal
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass


def retry(job_id: str) -> dict:
    """Submit the same task with the same parameters again (safe: outputs are published atomically)."""
    job = get(job_id)
    if not job:
        raise KeyError(job_id)
    if job["status"] in ("queued", "running"):
        raise RuntimeError("This job is still running")
    params = json.loads(job["params"]) if isinstance(job.get("params"), str) else (job.get("params") or {})
    new = submit(job["task"], params)
    _update(job_id, dismissed="1")        # the new attempt replaces this one's error box
    return new


def bootstrap() -> None:
    """Startup never trains. It installs a mounted release bundle (as a visible job) the first time it sees
    that bundle version; missing small mission datasets are fetched by app.autoinstall when online."""
    import logging
    try:
        from spaceml import release
        m = release.load_manifest()
        st = (release.status() or {}).get("installed") or {}
        if m and not (st.get("bundle_version") == m["bundle_version"] and st.get("complete")):
            submit("release.install", {})
    except Exception:  # noqa: BLE001 - never block startup
        logging.getLogger("release").exception("Release bundle install could not start")
    try:
        from . import autosetup
        if autosetup.profile() != "off":          # opt-in only (SPACEML_AUTOSETUP=full|light)
            autosetup.start(reason="startup")
    except Exception:  # noqa: BLE001
        logging.getLogger("autosetup").exception("Automatic setup could not start")


def busy(module: str) -> bool:
    with tx() as c:
        return c.execute("SELECT 1 FROM jobs WHERE module=? AND status IN ('queued','running')",
                         (module,)).fetchone() is not None


def cancel(job_id: str) -> dict:
    job = get(job_id)
    if not job:
        raise KeyError(job_id)
    if job["status"] not in ("queued", "running"):
        return job
    _cancelled.add(job_id)
    if job["status"] == "queued":
        _update(job_id, status="cancelled", finished_at=now_iso(), error="Cancelled by user")
    with _lock:
        proc = _procs.get(job_id)
    if proc:
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except Exception:  # noqa: BLE001
            proc.kill()
    return get(job_id)
