"""Tiny job used by the test-suite to exercise the job runner (sleeps N seconds, reporting progress)."""
import sys
import time

from .progress import report

if __name__ == "__main__":
    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(int(n * 4)):
        time.sleep(0.25)
        report(i + 1, int(n * 4), "sleeping")
