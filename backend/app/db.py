"""SQLite persistence: runs, model versions and evaluation results."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from spaceml.config import DB_PATH, IS_MANAGED_WORKER

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    run_id     TEXT NOT NULL,
    item       TEXT NOT NULL,          -- e.g. flag:120-180, region:3, candidate
    verdict    TEXT NOT NULL,          -- confirmed | dismissed | unsure
    note       TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, item)
);
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    module        TEXT NOT NULL,
    input_ref     TEXT NOT NULL,          -- JSON describing the selected input
    input_label   TEXT NOT NULL,
    model_version TEXT,
    status        TEXT NOT NULL,          -- queued | running | succeeded | failed
    error         TEXT,
    summary       TEXT,                   -- small JSON headline for lists
    result_path   TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    duration_ms   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_module ON runs(module, created_at);

CREATE TABLE IF NOT EXISTS model_versions (
    module      TEXT NOT NULL,
    version     TEXT NOT NULL,
    model_name  TEXT,
    created_at  TEXT,
    path        TEXT NOT NULL,
    manifest    TEXT,
    PRIMARY KEY (module, version)
);

CREATE TABLE IF NOT EXISTS evaluations (
    module        TEXT NOT NULL,
    version       TEXT NOT NULL,
    created_at    TEXT,
    model_name    TEXT,
    baseline_name TEXT,
    metrics       TEXT NOT NULL,          -- full metrics.json payload
    PRIMARY KEY (module, version)
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE" if IS_MANAGED_WORKER else "PRAGMA journal_mode=WAL")
    if IS_MANAGED_WORKER:
        conn.execute("PRAGMA synchronous=FULL")
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


MIGRATIONS = [  # (table, column, definition) - added when missing
    ("runs", "origin", "TEXT NOT NULL DEFAULT 'computed'"),   # computed | bundled
    ("runs", "repeat_of", "TEXT"),
    ("runs", "pinned_version", "TEXT"),
]


def init_db() -> None:
    with tx() as c:
        c.executescript(SCHEMA)
        for table, col, ddl in MIGRATIONS:
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        # Runs that were in flight when the server stopped can never finish.
        c.execute("UPDATE runs SET status='interrupted', error='Interrupted: the server stopped before this run "
                  "finished. Use Repeat run to start it again.' WHERE status IN ('queued','running')")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("input_ref", "summary", "manifest", "metrics"):
        if d.get(k):
            d[k] = json.loads(d[k])
    return d
