"""Evaluate on the expert-labeled AI4Mars test sets against an all-soil baseline.

    python -m spaceml.mars.evaluate [--version v...] [--splits test_min3 test_min1]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..common.progress import fail, report, step
from ..common.utils import now_iso, read_json, write_json
from ..config import MODELS_DIR
from .data import CLASSES, IGNORE
from .metrics import confusion, summarise
from .model import load, segment


def latest_dir():
    dirs = [d for d in (MODELS_DIR / "mars").glob("*") if (d / "model.pt").exists()]
    if not dirs:
        fail("No Mars model yet. Train one from Data & Training → Mars, or upload a Colab-trained zip.")
    return max(dirs, key=lambda d: (read_json(d / "manifest.json", {}) or {}).get("created_at", ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None)
    p.add_argument("--splits", nargs="+", default=None, help="local test splits (default: every test_* split)")
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N images per split (0 = all)")
    p.add_argument("--latest", action="store_true", help="evaluate the newest checkpoint (default)")
    args = p.parse_args()
    from . import local
    local.validate()
    have = {k: len(local.usable_pairs(k)) for k in local.expert_splits()}
    splits = [s for s in (args.splits or sorted(have)) if have.get(s)]
    if not splits:
        fail("The expert-labeled AI4Mars test pairs are not in this installation (release bundle, or "
             "Data & Training → Mars → Get dataset). Run: task:mars.dataset")

    vdir = MODELS_DIR / "mars" / args.version if args.version else latest_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load(vdir)
    model.to(device)
    rows, struggles = [], []
    for split in splits:
        step(f"Evaluating on {split}")
        items = local.usable_pairs(split)
        cm_m, cm_b = np.zeros((4, 4), np.int64), np.zeros((4, 4), np.int64)
        sand_misses = []
        n = len(items) if not args.limit else min(args.limit, len(items))
        for i in range(n):
            image, label = local.load_pair(items[i])
            pred = segment(model, image, size, device)
            c = confusion(pred, label)
            cm_m += c
            cm_b += confusion(np.zeros_like(label, dtype=np.uint8), label)  # everything is soil
            sand = (label == 2).sum()
            if sand > 2000:
                sand_misses.append((c[2, 2] / sand, i))
            report(i + 1, n, f"{split}: {i + 1}/{n} images")
        m, b = summarise(cm_m), summarise(cm_b)
        metrics = [{"name": "Mean IoU", "model": m["miou"], "baseline": b["miou"]},
                   {"name": "Pixel accuracy", "model": m["pixel_acc"], "baseline": b["pixel_acc"]}]
        metrics += [{"name": f"IoU {c}", "model": float(m["iou"][k]), "baseline": float(b["iou"][k])}
                    for k, c in enumerate(CLASSES)]
        rows.append({"group": f"Expert {split.replace('test_', '')} ({n} images)", "metrics": metrics,
                     "confusion": cm_m.tolist()})
        for recall, idx in sorted(sand_misses)[:4]:
            struggles.append(f"{split} image #{idx}: only {recall:.0%} of sand pixels recognised")
        cmn = cm_m / np.maximum(cm_m.sum(1, keepdims=True), 1)
        struggles.append(f"{split}: {cmn[2, 0]:.0%} of sand predicted as soil; "
                         f"{cmn[3, 1]:.0%} of big rock predicted as bedrock")
        print(f"{split}: mIoU {m['miou']:.4f} (all-soil {b['miou']:.4f}), sand IoU {m['iou'][2]:.4f}")

    manifest = read_json(vdir / "manifest.json", {})
    first = rows[0]["metrics"]
    write_json(vdir / "metrics.json", {
        "module": "mars", "version": vdir.name, "created_at": now_iso(),
        "model_name": manifest.get("model_name", "DeepLabV3+ ResNet-50"),
        "baseline_name": "Everything is soil",
        "eval_split": "AI4Mars expert-labeled Curiosity test sets, local pairs (rover + >30 m range masked)",
        "dataset_fingerprint": local.fingerprint(),
        "headline": {"metric": "Mean IoU", "model": first[0]["model"], "baseline": first[0]["baseline"]},
        "rows": rows, "struggles": struggles, "classes": CLASSES, "ignore_index": IGNORE,
    })
    print(f"Wrote {vdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
