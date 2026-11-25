from tests.test_release_and_jobs import _fresh
from types import SimpleNamespace


def test_managed_resume_keeps_ids_and_rebuilds_only_interrupted_work(tmp_path, monkeypatch):
    _fresh(monkeypatch, tmp_path, "managed-resume")
    from app import db, jobs, worker, managed_worker
    db.init_db()
    jobs.init()
    dispatched, validated = [], []
    class Thread:
        def __init__(self, *, target, args, **kwargs):
            self.target, self.args = target, args
        def start(self):
            dispatched.append((self.target, self.args))
    monkeypatch.setattr(managed_worker, "threading", SimpleNamespace(Thread=Thread))
    def build(params):
        validated.append(params)
        return [["spaceml.telemetry.data", "download"]]
    monkeypatch.setitem(jobs.TASKS, "test.resume", jobs.Task("telemetry", "Test resumable task", build, {}))
    pending, run = {"id": "resume-job"}, {"id": "resume-run"}
    with db.tx() as connection:
        connection.execute("INSERT INTO jobs (id,task,module,title,params,status,created_at) VALUES ('resume-job','test.resume','telemetry','Resume','{\"source\":\"validated\"}','queued','2026-01-01')")
        connection.execute("INSERT INTO runs (id,module,input_ref,input_label,status,created_at) VALUES ('resume-run','telemetry','{\"chan_id\":\"P-1\"}','Recovery','running','2026-01-01')")
        connection.execute("UPDATE jobs SET command='untrusted stored command' WHERE id=?", (pending["id"],))
        connection.execute("INSERT INTO jobs (id,task,module,title,params,status,error,created_at) VALUES ('cancelled','test.resume','telemetry','Cancelled','{}','cancelled','Cancelled by user','2026-01-01')")
    db.init_db()
    jobs.init()
    dispatched.clear()
    managed_worker.recover()
    assert jobs.get(pending["id"])["status"] == "queued"
    assert worker.get(run["id"])["status"] == "queued"
    assert jobs.get("cancelled")["status"] == "cancelled"
    assert len(dispatched) == 2
    assert dispatched[0][1] == (pending["id"], [["spaceml.telemetry.data", "download"]])
    assert dispatched[1][1][0] == run["id"]
    assert validated == [{"source": "validated"}]
