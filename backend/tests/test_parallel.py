"""The worker pool never hangs: a frozen task is handed back after the stall timeout."""
import time

from spaceml.common import parallel


def _work(x):
    if x == 3:
        time.sleep(3600)          # simulates a deadlocked worker / hung download
    if x == 2:
        raise ValueError("bad item")
    return x * x


def test_pool_returns_results_errors_and_stalled_items():
    got, errs = {}, {}
    t0 = time.time()
    left = parallel.run(_work, [0, 1, 2, 3], 2, on_result=lambda i, r: got.__setitem__(i, r),
                        on_error=lambda i, e: errs.__setitem__(i, str(e)), stall_seconds=8)
    assert got == {0: 0, 1: 1}
    assert "bad item" in errs[2]
    assert left == [3]
    assert time.time() - t0 < 60


def test_budgets_are_sane():
    assert parallel.cpu_budget() >= 1
    assert parallel.memory_budget_gb() > 0
