"""Release preparation: import a source once into the indexed store (data/missions/<mission>/).

Run on a machine that has the source files (or network access for the sources that can be fetched). The
running app never calls this at startup; the Data & Training page offers it as an explicit action.

    python -m spaceml.missions.prepare opssat  --from ./opssat            (or --download v2)
    python -m spaceml.missions.prepare hubble  --from ./hst_logs
    python -m spaceml.missions.prepare webb    --from ./jwst_fgs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..common.progress import fail, step
from . import catalog


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mission", choices=list(catalog.MISSIONS))
    p.add_argument("--from", dest="src", help="local source file or folder")
    p.add_argument("--download", nargs="*", help="fetch from the source (needs network): version / ids")
    a = p.parse_args()
    meta = catalog.get(a.mission)
    t0 = time.time()
    step(f"Preparing {meta['name']}")
    try:
        if a.mission == "opssat":
            from . import opssat
            src = Path(a.src) if a.src else None
            if a.download is not None:
                src = opssat.download(Path(a.src or "opssat-source"), (a.download or ["v2"])[0])
            if not src:
                fail("give --from <folder with dataset.csv and segments.csv> or --download v2")
            man = opssat.prepare(src)
        elif a.mission in ("hubble", "webb"):
            from . import pointing
            if not a.src:
                fail("give --from <folder of FITS files>")
            man = pointing.prepare(a.mission, Path(a.src))
        else:
            fail(f"no importer for {a.mission}")
    except (ValueError, FileNotFoundError, KeyError) as e:
        fail(f"{meta['name']}: {e}")
    print(json.dumps({k: man.get(k) for k in ("mission", "scope", "counts", "date_range", "items", "events",
                                               "prepared_bytes")}, indent=1, default=str))
    print(f"Prepared {meta['name']} in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
