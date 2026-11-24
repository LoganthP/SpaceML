"""Download and prepare everything in one go (resumable; safe to run again).

    python -m spaceml.quickstart                  # telemetry + Kepler (300 stars) + exoplanet model + Mars samples
    python -m spaceml.quickstart --stars 600 --mars-quick
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(*args: str) -> None:
    print(f"\n=== python -m {' '.join(args)}", flush=True)
    rc = subprocess.call([sys.executable, "-m", *args])
    if rc != 0:
        print(f"\nStep failed: python -m {' '.join(args)}\nFix the problem above (or run "
              "`python -m spaceml.doctor`), then run quickstart again - finished steps are skipped.")
        sys.exit(rc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stars", type=int, default=300, help="Kepler stars to download (more = better model)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-telemetry", action="store_true")
    p.add_argument("--skip-exoplanet", action="store_true")
    p.add_argument("--skip-mars", action="store_true")
    p.add_argument("--mars-quick", action="store_true", help="also train a small CPU Mars model")
    a = p.parse_args()
    from .telemetry import data as tdata

    if not a.skip_telemetry and not tdata.is_downloaded():
        run("spaceml.telemetry.data", "download")
    if not a.skip_exoplanet:
        run("spaceml.exoplanet.data", "tce", "--if-missing")
        run("spaceml.exoplanet.data", "lightcurves", "--n-stars", str(a.stars), "--workers", str(a.workers), "--max-quarters", "8")
        run("spaceml.exoplanet.train")
    if not a.skip_mars:
        from .mars import data as mdata
        if not mdata.list_samples():
            run("spaceml.mars.data", "samples", "--n", "36")
        if a.mars_quick:
            run("spaceml.mars.train", "--quick")
            run("spaceml.mars.evaluate", "--latest", "--limit", "120")
    print("\nAll done. Start the app and open the dashboard.")


if __name__ == "__main__":
    main()
