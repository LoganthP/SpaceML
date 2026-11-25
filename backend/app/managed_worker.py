"""Initialize/recover the one Drive-mounted Vercel worker without erasing user state."""
import json
import os
import threading


def initialize():
    from spaceml.config import REPO_ROOT, STORAGE_DIR
    from spaceml import release
    from spaceml.missions import store
    from .routers.missions import repair_needed
    marker = STORAGE_DIR / "managed-assets.json"
    revision = os.environ["SMC_BUNDLE_SHA256"]
    if marker.is_file() and json.loads(marker.read_text()).get("revision") == revision:
        return
    installed = release.install_if_needed()
    if installed and not installed.get("complete"):
        raise RuntimeError("Bundled release datasets failed verification")
    if not store.installed("webb") or repair_needed("webb"):
        from spaceml.missions.published import install
        install(REPO_ROOT / "release/mission-results")
    marker.write_text(json.dumps({"revision": revision}), encoding="utf-8")


def recover():
    """Resume only tasks interrupted by suspension, preserving the same job IDs.

    Saved allowlisted commands are rebuilt from validated parameters. Never run
    arbitrary command text from a database or restart failures/cancelled work.
    """
    from . import jobs, worker
    from .db import tx
    with tx() as connection:
        pending = [dict(r) for r in connection.execute("SELECT * FROM jobs WHERE status='interrupted' AND error LIKE 'Interrupted: the server stopped%'")]
        runs = [dict(r) for r in connection.execute("SELECT * FROM runs WHERE status IN ('queued','running') OR (status='interrupted' AND error LIKE 'Interrupted: the server stopped before this run%')")]
    for row in pending:
        task = jobs.TASKS.get(row["task"])
        if not task:
            continue
        try:
            commands = task.build(json.loads(row["params"] or "{}"))
        except (ValueError, OSError):
            continue
        jobs._update(row["id"], status="queued", error=None, finished_at=None, message="Resuming after Vercel runtime restart")
        threading.Thread(target=jobs._run, args=(row["id"], commands), daemon=True, name=f"resume-{row['id']}").start()
    for row in runs:
        with tx() as connection:
            connection.execute("UPDATE runs SET status='queued', error=NULL WHERE id=?", (row["id"],))
        threading.Thread(target=worker._execute, args=(row["id"], row["module"], json.loads(row["input_ref"]), row.get("model_version")), daemon=True).start()
