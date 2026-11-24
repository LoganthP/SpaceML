"""Train one LSTM forecaster per telemetry channel.

    python -m spaceml.telemetry.train                 # all 82 channels, new version
    python -m spaceml.telemetry.train --channels P-1 M-6 --version v-demo
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from ..common.progress import report, step
from ..common.utils import new_version, now_iso, set_seed, write_json
from ..config import MODELS_DIR, SEED
from . import data
from .detection import WINDOW
from .model import LSTMForecaster, make_windows, save_checkpoint

HPARAMS = dict(hidden=80, layers=2, dropout=0.3, window=WINDOW, epochs=35, patience=10,
               min_delta=3e-4, batch_size=64, lr=1e-3, val_frac=0.2)


def train_channel(chan_id: str, out_dir, epochs: int = 35, patience: int = 10, batch_size: int = 64,
                  verbose: bool = True, stride: int = 1) -> dict:
    set_seed(SEED)
    train, _ = data.load_arrays(chan_id)
    X, y = make_windows(train)
    n_val = max(1, int(len(X) * HPARAMS["val_frac"]))  # last 20% of training file, time order kept
    Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    if stride > 1:  # neighbouring windows overlap by 249/250 steps; every n-th one carries nearly the same signal
        Xtr, ytr = Xtr[::stride], ytr[::stride]
        Xva, yva = Xva[::stride], yva[::stride]

    model = LSTMForecaster(train.shape[1], HPARAMS["hidden"], HPARAMS["layers"], HPARAMS["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    loss_fn = nn.MSELoss()
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    bs = batch_size
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(Xva[i:i + 512].contiguous()), yva[i:i + 512]).item()
                                 for i in range(0, len(Xva), 512)]))
        if val < best - HPARAMS["min_delta"]:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    save_checkpoint(model, out_dir / f"{chan_id}.pt", val_mse=best, epochs_run=epoch + 1)
    info = {"chan_id": chan_id, "val_mse": best, "epochs": epoch + 1, "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"  {chan_id:6s} val_mse={best:.5f} epochs={epoch + 1:2d} ({info['seconds']}s)", flush=True)
    return info


FAST = {"epochs": 15, "patience": 4, "batch_size": 256, "stride": 4}
WORKER_GB = 0.7          # resident memory of one training process (torch + one channel)


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _train_one(args: tuple) -> dict:
    c, out, kw = args
    return train_channel(c, out, verbose=True, **kw)


def plan_workers(n_jobs: int, workers: int = 0, threads: int = 0) -> tuple[int, int]:
    """Processes x threads that fit this container's CPUs and memory. Small LSTMs scale better
    across processes than across threads, so use up to one process per CPU (max 8)."""
    from ..common.parallel import memory_budget_gb, train_cpus
    cpus = train_cpus()
    mem_cap = max(1, int((memory_budget_gb() - 1.0) / WORKER_GB))
    w = workers or max(1, min(8, cpus, mem_cap, n_jobs))
    t = threads or max(1, cpus // w)
    return w, t


def main() -> None:
    import os  # noqa: F401

    p = argparse.ArgumentParser(description="Train per-channel telemetry LSTMs")
    p.add_argument("--channels", nargs="*", default=["all"])
    p.add_argument("--version", default=None, help="model version folder (default: timestamp)")
    p.add_argument("--fast", action="store_true",
                   help="~8-10x faster: every 4th training window, batch 256, 15 epochs, patience 4")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="use every n-th training window (default 1, fast 4)")
    p.add_argument("--workers", type=int, default=0, help="channels trained in parallel (0 = auto, up to 4)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads per worker (0 = auto)")
    args = p.parse_args()
    base = FAST if args.fast else {"epochs": HPARAMS["epochs"], "patience": HPARAMS["patience"],
                                   "batch_size": HPARAMS["batch_size"], "stride": 1}
    kw = {"epochs": args.epochs or base["epochs"], "patience": args.patience or base["patience"],
          "batch_size": args.batch_size or base["batch_size"], "stride": args.stride or base["stride"]}

    if not data.is_downloaded():
        from ..common.progress import fail
        fail("Telemetry data is not installed. On Data & Training, press Download on the telemetry card (or add the release bundle).")
    chans = data.load_channels()
    ids = sorted(chans) if args.channels in (["all"], []) else args.channels
    version = args.version or new_version()
    out = MODELS_DIR / "telemetry" / version
    out.mkdir(parents=True, exist_ok=True)
    from ..common.utils import read_json
    created = (read_json(out / "manifest.json") or {}).get("created_at", now_iso())

    def write_manifest():
        write_json(out / "manifest.json", {
            "module": "telemetry", "version": version, "created_at": created,
            "model_name": "LSTM forecaster (2x80, dropout 0.3) + nonparametric dynamic threshold",
            "hparams": {**HPARAMS, **kw, "mode": "fast" if args.fast else "full"},
            "channels": sorted(p.stem for p in out.glob("*.pt")),
            "data": "NASA SMAP & MSL telemetry (Hundman et al. 2018) - train files only",
        })

    write_manifest()
    todo = [c for c in ids if not (out / f"{c}.pt").exists()]
    done = len(ids) - len(todo)
    workers, threads = plan_workers(len(todo), args.workers, args.threads)
    step(f"Training {len(todo)} channel models into {version} ({'fast' if args.fast else 'full'} mode, "
         f"{workers} in parallel x {threads} thread(s); {done} already trained)")
    report(done, len(ids), f"{done}/{len(ids)} channels ready, training the rest…", every=0)
    failed: list[str] = []

    def finished(c: str, _info=None) -> None:
        nonlocal done
        done += 1
        write_manifest()
        report(done, len(ids), f"{c} done ({done}/{len(ids)} channels)", every=0)

    def errored(c: str, e: BaseException) -> None:
        failed.append(c)
        print(f"  {c} failed: {type(e).__name__}: {e}", flush=True)

    left = todo
    if workers > 1 and len(todo) > 1:
        from ..common import parallel
        from spaceml.telemetry.train import _init_worker as init_w, _train_one as one  # importable by name (spawn)
        left = parallel.run(one, [(c, out, kw) for c in todo], workers, initializer=init_w, initargs=(threads,),
                            on_result=lambda it, info: finished(it[0], info), on_error=lambda it, e: errored(it[0], e),
                            stall_seconds=15 * 60)
        left = [it[0] for it in left]
    if left:  # one worker, or the pool stalled/crashed: finish in this process
        torch.set_num_threads(max(1, threads * max(1, workers) // 2) if workers > 1 else threads)
        for c in left:
            if (out / f"{c}.pt").exists():
                finished(c)
                continue
            try:
                train_channel(c, out, **kw)
                finished(c)
            except Exception as e:  # noqa: BLE001
                errored(c, e)
    if failed:
        print(f"{len(failed)} channel(s) could not be trained: {' '.join(failed)}")
    if done == len(failed) and failed:
        from ..common.progress import fail
        fail("No channel could be trained - see the log above.")
    print("Done. Next: python -m spaceml.telemetry.evaluate --version", version)


if __name__ == "__main__":
    main()
