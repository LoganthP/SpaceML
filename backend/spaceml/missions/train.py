"""Quick / full training for a mission (job entry point used by the Train buttons).

    python -m spaceml.missions.train opssat --quick
    python -m spaceml.missions.train hubble --quick
    python -m spaceml.missions.train webb --quick

CPU threads are bounded by SPACEML_TRAIN_CPUS. Artifacts are written to a staging folder and published only
when training and evaluation both succeed; otherwise the previous model stays active.
"""
from __future__ import annotations

import argparse
import os
import time

from ..common.parallel import train_cpus
from ..common.progress import fail, report, step
from . import catalog, models, store


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mission", choices=list(catalog.MISSIONS))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=2137)
    a = p.parse_args()
    meta = catalog.get(a.mission)
    if not meta.get("train"):
        fail(f"{meta['name']} has no training step: its data carries no labels for a model. Use rolling detection "
             "(Score) and review the flagged intervals instead.")
    if not store.installed(a.mission):
        fail(f"{meta['name']} data is not installed. Prepare it once with: {meta['prepare']}")
    cpus = train_cpus()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(cpus)
    man = store.manifest(a.mission) or {}
    step(f"Training {meta['train']['label']} for {meta['name']} ({'quick' if a.quick else 'full'}, {cpus} CPU threads)")
    print(f"dataset: {man.get('scope')} · source {(man.get('source') or {}).get('name')} "
          f"{(man.get('source') or {}).get('version') or ''} · fingerprint {man.get('fingerprint')}", flush=True)
    t0 = time.time()
    try:
        stg = models.Staging(a.mission, a.quick)
        with stg as d:
            kind = meta["train"]["kind"]
            if kind == "random_forest":
                from . import opssat
                m = opssat.train(d, quick=a.quick, seed=a.seed, n_jobs=cpus, progress=report)
            elif kind == "isolation_forest_pointing":
                from . import pointing
                m = pointing.train(a.mission, d, quick=a.quick, seed=a.seed, n_jobs=cpus, progress=report)
            else:
                fail(f"unknown training kind {kind}")
    except (ValueError, FileNotFoundError) as e:
        fail(f"{meta['name']} training failed: {e}")
    h = m.get("headline", {})
    print(f"Saved model version {stg.version} at {stg.final} · {h.get('metric')}: model {h.get('model')} "
          f"vs baseline {h.get('baseline')} · {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
