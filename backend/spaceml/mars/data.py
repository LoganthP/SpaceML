"""AI4Mars (NASA JPL) data access.

Primary source: Hugging Face `hassanjbara/AI4MARS` (Curiosity NAVCAM, 18.1k train /
322 expert-labeled test_min3 images). Each row carries the image, the label mask,
the rover mask and the >30 m range mask.

    python -m spaceml.mars.data samples --n 36        # curated expert-labeled samples for the app
    python -m spaceml.mars.data samples --from-nasa /path/to/ai4mars-dataset-merged-0.1

Label values: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 no label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.progress import report
from ..common.utils import read_json, write_json
from ..config import MARS_DIR

HF_DATASET = "hassanjbara/AI4MARS"
CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255
SAMPLES_DIR = MARS_DIR / "samples"
INDEX_PATH = SAMPLES_DIR / "index.json"


def _arr(x) -> np.ndarray | None:
    if x is None:
        return None
    a = np.array(x)
    return a[..., 0] if a.ndim == 3 else a


def _first(row: dict, *keys):
    for k in keys:
        if row.get(k) is not None:
            return row[k]
    return None


def prepare_row(row: dict) -> tuple[Image.Image, np.ndarray, np.ndarray] | None:
    """Image (RGB), label with rover/range regions set to 255, and the combined ignore mask."""
    image = _first(row, "image", "img", "jpg", "png")
    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    lab = _first(row, "label_mask", "label", "labels", "mask", "annotation")
    if lab is None or row.get("has_labels") is False:
        return None          # unlabeled row: callers skip it
    label = _arr(lab).astype(np.int64)
    ignore = np.zeros(label.shape, bool)
    for key in ("rover_mask", "range_mask"):
        m = _arr(row.get(key))
        if m is not None and m.shape == label.shape:
            ignore |= m > 0
    label[ignore] = IGNORE
    label[(label > 3) & (label != IGNORE)] = IGNORE
    return image, label, ignore


def list_samples() -> list[dict]:
    return read_json(INDEX_PATH, []) or []


def sample_name(sample_id: str) -> str:
    return next((s.get("name", "") for s in list_samples() if s["id"] == sample_id), "")


def sample_paths(sample_id: str) -> dict[str, Path]:
    for s in list_samples():
        if s["id"] == sample_id:
            return {k: SAMPLES_DIR / s[k] for k in ("image", "label", "ignore") if s.get(k)}
    raise ValueError(f"Unknown Mars sample {sample_id}")


def _save_sample(idx: int, name: str, image: Image.Image, label: np.ndarray, ignore: np.ndarray, split: str) -> dict:
    sid = f"{split}-{idx:03d}"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("L").save(SAMPLES_DIR / f"{sid}.png", optimize=True)
    Image.fromarray(label.astype(np.uint8)).save(SAMPLES_DIR / f"{sid}_label.png")
    Image.fromarray((ignore * 255).astype(np.uint8)).save(SAMPLES_DIR / f"{sid}_ignore.png")
    labeled = label[label != IGNORE]
    shares = np.bincount(labeled, minlength=4)[:4] / max(len(labeled), 1)
    thumb = image.convert("L").copy()
    thumb.thumbnail((256, 256))
    thumb.save(SAMPLES_DIR / f"{sid}_thumb.jpg", quality=85)
    return {"id": sid, "name": name, "split": split, "image": f"{sid}.png", "label": f"{sid}_label.png",
            "ignore": f"{sid}_ignore.png", "thumb": f"{sid}_thumb.jpg", "width": image.width, "height": image.height,
            "labeled_fraction": float(len(labeled) / label.size),
            "label_shares": {c: float(s) for c, s in zip(CLASSES, shares)}}


def prepare_samples_hf(n: int, split: str = "test_min3") -> None:
    from datasets import load_dataset

    from ..common.progress import fail
    print(f"Streaming {n} expert-labeled images from {HF_DATASET} [{split}] ...")
    try:
        ds = iter(load_dataset(HF_DATASET, split=split, streaming=True))
        first = next(ds)
    except Exception as e:  # noqa: BLE001
        fail(f"source_unavailable: Hugging Face ({HF_DATASET}) could not be reached: {type(e).__name__}: "
             f"{str(e)[:200]}")
    import itertools
    index, rare = [], {2: 0, 3: 0}
    for i, row in enumerate(itertools.chain([first], ds)):
        prepared = prepare_row(row)
        if prepared is None:
            continue
        image, label, ignore = prepared
        if (label != IGNORE).mean() < 0.05:
            continue
        rec = _save_sample(len(index), row.get("file_name") or row.get("name") or f"{split}_{i}",
                           image, label, ignore, split)
        index.append(rec)
        for c in rare:
            rare[c] += int(rec["label_shares"][CLASSES[c]] > 0.02)
        report(len(index), n, f"{len(index)}/{n} samples saved")
        if len(index) >= n:
            break
    write_json(INDEX_PATH, index)
    print(f"Samples ready in {SAMPLES_DIR} (images with sand: {rare[2]}, with big rock: {rare[3]})")


def prepare_samples_local(n: int) -> bool:
    """Pick the review samples from the local expert test pairs (no network). False if none."""
    from . import local
    split = local.test_split()
    if not split:
        return False
    items = local.usable_pairs(split)
    if not items:
        return False
    rng = np.random.default_rng(0)
    pick = [items[i] for i in sorted(rng.choice(len(items), min(n, len(items)), replace=False))]
    index = []
    for k, p in enumerate(pick):
        image, label = local.load_pair(p)
        index.append(_save_sample(k, p["key"], image, label, label == IGNORE, split))
        report(k + 1, len(pick), f"{k + 1}/{len(pick)} samples")
    write_json(INDEX_PATH, index)
    print(f"{len(index)} samples taken from the local {split} pairs")
    return True


def prepare_samples_nasa(root: str, n: int, split: str = "min3") -> None:
    """Use the official data.nasa.gov release folder (ai4mars-dataset-merged-0.1)."""
    root_p = Path(root)
    msl = root_p / "msl" if (root_p / "msl").exists() else root_p
    lbl_dir = msl / "labels" / "test" / f"masked-gold-{split}-100agree"
    files = sorted(lbl_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"No test labels found in {lbl_dir}")
    index = []
    for f in files:
        stem = f.stem.replace("_merged", "")
        img_path = msl / "images" / "edr" / f"{stem}.JPG"
        if not img_path.exists():
            continue
        label = np.array(Image.open(f)).astype(np.int64)
        label = label[..., 0] if label.ndim == 3 else label
        ignore = label == IGNORE  # official test labels already have rover/range masked
        if (label != IGNORE).mean() < 0.05:
            continue
        index.append(_save_sample(len(index), stem, Image.open(img_path), label, ignore, f"test_{split}"))
        if len(index) >= n:
            break
    write_json(INDEX_PATH, index)
    print(f"Saved {len(index)} samples to {SAMPLES_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description="AI4Mars data")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("samples", help="save curated expert-labeled samples for the dashboard")
    s.add_argument("--n", type=int, default=36)
    s.add_argument("--split", default="test_min3")
    s.add_argument("--from-nasa", help="path to the official AI4Mars release instead of Hugging Face")
    args = p.parse_args()
    if args.from_nasa:
        prepare_samples_nasa(args.from_nasa, args.n)
    elif prepare_samples_local(args.n):
        pass
    else:
        prepare_samples_hf(args.n, args.split)


if __name__ == "__main__":
    main()
