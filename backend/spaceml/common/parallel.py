"""Process pool that cannot hang forever.

Why: with the default "fork" start method, a worker forked from a process that has already
imported PyTorch / OpenMP / network libraries can deadlock before doing any work (this is what
froze telemetry training overnight on multi-core machines). Workers here are started with
"spawn" (a fresh interpreter each), and a watchdog stops the pool when no task has finished for
`stall_seconds`; the caller gets the unfinished items back to retry another way.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable, Iterable


def cpu_budget() -> int:
    """CPUs this container may actually use (cgroup quota aware)."""
    n = os.cpu_count() or 2
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:  # cgroup v2: "max 100000" or "400000 100000"
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def train_cpus() -> int:
    """CPUs a training job may use: SPACEML_TRAIN_CPUS, else all but one (the dashboard keeps one)."""
    env = os.environ.get("SPACEML_TRAIN_CPUS", "").strip()
    if env.isdigit() and int(env) > 0:
        return min(int(env), cpu_budget())
    return max(1, cpu_budget() - 1)


def memory_budget_gb() -> float:
    """Memory this container may use, in GB (cgroup limit, else physical RAM)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v != "max" and int(v) < 1 << 50:
                return int(v) / 1e9
        except (OSError, ValueError):
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 4.0


def run(fn: Callable[[Any], Any], items: Iterable[Any], workers: int, *,
        on_result: Callable[[Any, Any], None], on_error: Callable[[Any, BaseException], None],
        initializer: Callable | None = None, initargs: tuple = (), stall_seconds: float = 900) -> list:
    """Run fn(item) in `workers` spawned processes. Returns the items that never finished
    (pool stalled or crashed) so the caller can retry them sequentially."""
    items = list(items)
    if not items:
        return []
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=initializer, initargs=initargs)
    pending = {ex.submit(fn, it): it for it in items}
    try:
        while pending:
            done, _ = wait(pending, timeout=stall_seconds, return_when=FIRST_COMPLETED)
            if not done:
                print(f"::warn::no task finished for {int(stall_seconds / 60)} min - stopping the worker pool "
                      f"and finishing the remaining {len(pending)} item(s) one by one", flush=True)
                break
            for f in done:
                it = pending.pop(f)
                try:
                    on_result(it, f.result())
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ == "BrokenProcessPool":
                        print("::warn::a worker process died (out of memory?) - finishing the rest one by one",
                              flush=True)
                        pending[f] = it
                        raise _Broken from e
                    on_error(it, e)
    except _Broken:
        pass
    finally:
        left = list(pending.values())
        if left:
            for p in list(getattr(ex, "_processes", {}).values()):
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
        ex.shutdown(wait=not left, cancel_futures=True)
    return left


class _Broken(Exception):
    pass
