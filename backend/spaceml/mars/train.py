"""Fine-tune DeepLabV3+ on AI4Mars from the LOCAL image/mask pairs (spaceml.mars.local).

    python -m spaceml.mars.train --quick          # pipeline check: 300 pairs, 3 epochs, 256 px
    python -m spaceml.mars.train                  # full run over every local train pair
    python -m spaceml.mars.train --resume         # continue the newest interrupted run
    python -m spaceml.mars.evaluate --latest

No network is used. Before the first epoch a preflight validates sample pairs and runs one
training batch, so a data or model problem fails in seconds, not after an hour. A checkpoint
(model + optimizer + epoch) is written after every epoch; --resume continues from it.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from ..common.progress import fail, report, step
from ..common.utils import new_version, now_iso, set_seed, write_json
from ..config import MODELS_DIR, SEED
from .data import CLASSES, IGNORE
from .metrics import confusion, summarise
from .model import IMAGENET_MEAN, IMAGENET_STD, build_model


def _resize_pair(image: Image.Image, label: np.ndarray, size: int) -> tuple[Image.Image, Image.Image]:
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    # nearest-neighbour for masks: interpolation would invent classes between labels
    lab = Image.fromarray(np.where(label == IGNORE, 255, label).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return img, lab


class PairDataset(Dataset):
    """Items are produced by `get(i) -> (PIL image, PIL label)` at the training size."""

    def __init__(self, get, n: int, augment: bool):
        self.get, self.n, self.augment = get, n, augment

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        image, lab = self.get(i)
        if self.augment:
            if random.random() < 0.5:
                image, lab = image.transpose(Image.FLIP_LEFT_RIGHT), lab.transpose(Image.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        x = (np.asarray(image, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(np.asarray(lab, np.int64))


def _weights(labels) -> np.ndarray:
    counts = np.zeros(4)
    for lab in labels:
        a = np.asarray(lab)
        counts += np.bincount(a[a != 255].ravel(), minlength=4)[:4]
    counts = np.maximum(counts, 1)
    return 1 - counts / counts.sum()


def _cpu_threads() -> int:
    """Leave a core for the dashboard unless SPACEML_TRAIN_CPUS says otherwise."""
    from ..common.parallel import train_cpus
    return train_cpus()


class LocalGetter:
    """Picklable: reads a local pair and resizes it to the training size."""

    def __init__(self, items, size):
        self.items, self.size = items, size

    def __call__(self, i):
        from .local import load_pair
        img, lab = load_pair(self.items[i])
        return _resize_pair(img, lab, self.size)


def _latest_staging():
    d = [x for x in (MODELS_DIR / "mars").glob(".staging-*") if (x / "last.pt").exists()]
    return max(d, key=lambda x: (x / "last.pt").stat().st_mtime) if d else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="pipeline check: 300 pairs, 3 epochs, 256 px")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0, help="limit training pairs (0 = all local pairs)")
    p.add_argument("--version", default=None)
    p.add_argument("--encoder", default=None, help="resnet18 / resnet34 / resnet50 (default resnet34)")
    p.add_argument("--resume", action="store_true", help="continue the newest interrupted run")
    p.add_argument("--no-pretrained", action="store_true", help="same as --pretrained none")
    p.add_argument("--pretrained", choices=["auto", "local", "imagenet", "none"], default="auto",
                   help="encoder init. auto = none for quick runs, local for full runs. local = ImageNet weights only "
                        "if they are already on disk (release bundle / cache), never downloaded. imagenet = download "
                        "if needed (needs internet). none = random init")
    args = p.parse_args()
    from . import local
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(_cpu_threads())

    resume = None
    if args.resume:
        sd = _latest_staging()
        if sd is None:
            fail("Nothing to resume: no interrupted Mars training run was found.")
        resume = torch.load(sd / "last.pt", map_location="cpu", weights_only=False)
        a = resume["args"]
        args.quick, args.epochs, args.img_size, args.batch = a["quick"], a["epochs"], a["img_size"], a["batch"]
        args.max_train, args.encoder, args.version, args.lr = a["max_train"], a["encoder"], a["version"], a["lr"]
        print(f"Resuming {args.version} after epoch {resume['epoch']}/{args.epochs}", flush=True)

    quick = args.quick
    epochs = args.epochs or (3 if quick else 12)
    size = args.img_size or (256 if quick else 384)
    batch = args.batch or (4 if quick else 8)
    encoder = args.encoder or "resnet34"
    max_train = args.max_train or (300 if quick else 0)

    # ---------------- data: local pairs only
    step("Checking the local AI4Mars dataset")
    import os
    print(f"cwd {os.getcwd()} · dataset root {local.ROOT} · manifest {local.INDEX} "
          f"(exists: {local.INDEX.exists()})", flush=True)
    v = local.validate(progress=True)
    for sp_name, c in sorted(v.get("splits", {}).items()):
        print(f"  {sp_name}: {c['pairs']} pairs, {c['usable']} usable, {c['held_out']} held out, problems {c['problems']}",
              flush=True)
    for ex in v.get("problem_examples", []):
        print("  problem:", ex, flush=True)
    idx = local.index()
    if not idx or not v.get("splits", {}).get("train", {}).get("pairs"):
        fail(f"No AI4Mars training pairs at {local.ROOT} (manifest {local.INDEX}). Add them with the release "
             "bundle, or Data & Training → AI4Mars → Download (needs internet once).")
    items = local.usable_pairs("train")
    if len(items) < 16:
        fail(f"Only {len(items)} of {v['splits']['train']['pairs']} train pairs at {local.ROOT} are usable "
             f"(problems: {v['splits']['train']['problems']}; e.g. {'; '.join(v.get('problem_examples', [])[:2])}).")
    test_n = sum(len(local.usable_pairs(t)) for t in local.expert_splits())
    held = v["splits"]["train"]["held_out"]
    print(f"split check (pixel hashes): {len(items)} train images used, {test_n} expert test images kept out of "
          f"training, {held} train duplicates of expert test images held out", flush=True)
    try:
        pf = local.preflight("train", 8)
    except local.SchemaError as e:
        fail(f"AI4Mars local data failed validation: {e}")
    order = np.random.default_rng(SEED).permutation(len(items))
    items = [items[i] for i in order]
    if max_train:
        items = items[:max_train]
    n_val = max(8, len(items) // 10)
    val_items, tr_items = items[:n_val], items[n_val:]
    if len(tr_items) < 8:
        fail(f"Only {len(items)} local training pairs - need at least 16.")
    g_tr, g_va = LocalGetter(tr_items, size), LocalGetter(val_items, size)
    weights = _weights([g_tr(i)[1] for i in range(min(400, len(tr_items)))])
    ds_tr, ds_va = PairDataset(g_tr, len(tr_items), True), PairDataset(g_va, len(val_items), False)
    n_train = len(tr_items)
    source = (f"AI4Mars {idx.get('dataset')} (revision {str(idx.get('revision') or 'n/a')[:12]}), local train pairs: "
              f"{n_train} train / {len(val_items)} validation (seeded split), dataset fingerprint {local.fingerprint()}")
    print(f"preflight: {pf['checked']} of {pf['pairs']} pairs validated", flush=True)
    print(f"train {len(ds_tr)}  val {len(ds_va)}  device {device}  size {size}px  epochs {epochs}  "
          f"threads {torch.get_num_threads()}  mode {'quick (pipeline check)' if quick else 'full'}")
    print("class weights (1 - share):", {c: round(float(w), 3) for c, w in zip(CLASSES, weights)})
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=args.workers, drop_last=True,
                       pin_memory=device == "cuda")
    dl_va = DataLoader(ds_va, batch_size=batch, num_workers=args.workers)

    mode_pt = "none" if args.no_pretrained else args.pretrained
    if mode_pt == "auto":
        mode_pt = "none" if quick else "local"
    if resume:
        mode_pt = "none"                     # weights come from the checkpoint
    from .model import local_encoder_weights
    if mode_pt == "local":
        mode_pt = "imagenet" if local_encoder_weights(encoder) else "none"
        print("encoder init: " + ("ImageNet weights found on disk" if mode_pt == "imagenet" else
              "no ImageNet weights on disk - random init (offline-safe)"), flush=True)
    else:
        print(f"encoder init: {'random (offline-safe pipeline check)' if mode_pt == 'none' else 'ImageNet (download if needed)'}",
              flush=True)
    model = build_model(None if mode_pt == "none" else "imagenet", encoder).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                        ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 3, total_steps=max(1, epochs * len(dl_tr)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    # ---------------- preflight: one real training batch before committing to the run
    if not resume:
        step("Preflight: one training batch")
        x, y = next(iter(dl_tr))
        model.train()
        l0 = loss_fn(model(x.to(device)), y.to(device))
        l0.backward()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(l0):
            fail(f"Preflight failed: the loss is {l0.item()} on the first batch.")
        print(f"preflight batch OK: loss {l0.item():.3f} on {tuple(x.shape)}", flush=True)

    version = args.version or (new_version() + ("-quick" if quick else "-full"))
    final = MODELS_DIR / "mars" / version
    out = MODELS_DIR / "mars" / f".staging-{version}"   # published by rename when complete
    import shutil
    start_ep, best, history, t0 = 0, -1.0, [], time.time()
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        start_ep, best, history = resume["epoch"], resume["best"], resume["history"]
    else:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
    run_args = {"quick": quick, "epochs": epochs, "img_size": size, "batch": batch, "max_train": max_train,
                "encoder": encoder, "version": version, "lr": args.lr}
    total_steps = epochs * len(dl_tr)
    step(f"Training DeepLabV3+ ({'quick pipeline check' if quick else 'full run'})")
    for ep in range(start_ep, epochs):
        model.train()
        for k, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            report(ep * len(dl_tr) + k + 1, total_steps,
                   f"epoch {ep + 1}/{epochs} · batch {k + 1}/{len(dl_tr)} · loss {loss.item():.3f}")
            if k % 100 == 0:
                print(f"  epoch {ep + 1} step {k}/{len(dl_tr)} loss {loss.item():.4f}", flush=True)
        model.eval()
        cm = np.zeros((4, 4), np.int64)
        with torch.no_grad():
            for x, y in dl_va:
                cm += confusion(model(x.to(device)).argmax(1).cpu().numpy(), y.numpy())
        s = summarise(cm)
        history.append({"epoch": ep + 1, "val_miou": s["miou"], "val_iou": s["iou"].tolist()})
        print(f"epoch {ep + 1}: val mIoU {s['miou']:.4f}  IoU { {c: round(float(x), 3) for c, x in zip(CLASSES, s['iou'])} }", flush=True)
        if s["miou"] > best:
            best = s["miou"]
            torch.save({"state_dict": model.state_dict(), "img_size": size, "val_miou": best, "epoch": ep + 1,
                        "encoder": encoder}, out / "model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "history": history, "args": run_args}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")      # resume point
    write_json(out / "manifest.json", {
        "module": "mars", "version": version, "created_at": now_iso(),
        "model_name": f"DeepLabV3+ ({encoder})" + (" - quick run (pipeline check)" if quick else ""),
        "run_kind": "quick" if quick else "full", "encoder": encoder,
        "encoder_init": "imagenet" if mode_pt == "imagenet" else "random",
        "train_test_overlap": 0, "expert_test_images": test_n, "held_out_duplicates": held,
        "dataset_root": str(local.ROOT),
        "img_size": size, "epochs": epochs, "val_miou": best, "class_weights": weights,
        "train_images": n_train, "history": history, "train_seconds": round(time.time() - t0), "data": source})
    (out / "last.pt").unlink(missing_ok=True)
    if final.exists():
        shutil.rmtree(final)
    out.replace(final)
    print(f"Saved {final}. Next: python -m spaceml.mars.evaluate --version {version}")


if __name__ == "__main__":
    main()
