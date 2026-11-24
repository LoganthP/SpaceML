"""Tiny progress protocol shared by every CLI script.

Scripts print `::progress::<done>/<total>::<message>` lines. In a terminal they are
shown as normal log lines; the dashboard's job runner parses them into a progress bar.
"""
from __future__ import annotations

import sys
import time

_last = {"t": 0.0}


def report(done: int, total: int, message: str = "", every: float = 0.5) -> None:
    now = time.time()
    if done < total and now - _last["t"] < every:
        return
    _last["t"] = now
    print(f"::progress::{int(done)}/{max(int(total), 1)}::{message}", flush=True)


def step(message: str) -> None:
    """A headline line that the dashboard shows as the job's current step."""
    print(f"::step::{message}", flush=True)


def fail(message: str, code: int = 2) -> None:
    """Exit with a clear, user-facing reason (shown in the dashboard job card)."""
    print(f"::error::{message}", flush=True)
    sys.exit(code)
