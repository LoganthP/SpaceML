from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES, IGNORE
from .metrics import confusion, summarise
from .model import load, segment

MAX_SIDE = 1536


@lru_cache(maxsize=2)
def _model(version_dir: str, mtime: float):
    return load(Path(version_dir))


def run(image_path: Path, version_dir: Path, run_dir: Path,
        label_path: Path | None = None, ignore_path: Path | None = None) -> dict:
    ckpt = version_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError("No Mars checkpoint installed (models/mars/<version>/model.pt). "
                                "Train it in Colab with notebooks/mars_training.ipynb.")
    model, size = _model(str(version_dir), ckpt.stat().st_mtime)
    image = Image.open(image_path)
    image.load()
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    pred = segment(model, image, size)

    ignore = None
    if ignore_path and ignore_path.exists():
        ignore = np.array(Image.open(ignore_path).resize(image.size, Image.NEAREST)) > 0
        pred = pred.copy()
        pred[ignore] = IGNORE  # rover body and terrain beyond 30 m are not classified

    image.convert("L").save(run_dir / "input.png")
    Image.fromarray(pred).save(run_dir / "mask.png")  # raw class ids, coloured in the browser
    valid = pred != IGNORE
    shares = np.bincount(pred[valid].ravel(), minlength=4)[:4] / max(valid.sum(), 1)

    evaluation = None
    if label_path and label_path.exists():
        label = np.array(Image.open(label_path).resize(image.size, Image.NEAREST)).astype(np.int64)
        Image.fromarray(label.astype(np.uint8)).save(run_dir / "label.png")
        m = summarise(confusion(pred, label))
        b = summarise(confusion(np.zeros_like(pred), label))
        evaluation = {
            "labeled_fraction": float((label != IGNORE).mean()),
            "model": {"miou": m["miou"], "pixel_acc": m["pixel_acc"],
                      "iou": {c: (None if not m["present"][k] else float(m["iou"][k])) for k, c in enumerate(CLASSES)}},
            "baseline": {"miou": b["miou"], "pixel_acc": b["pixel_acc"],
                         "iou": {c: (None if not b["present"][k] else float(b["iou"][k])) for k, c in enumerate(CLASSES)}},
        }
    return {
        "width": image.width, "height": image.height,
        "classes": CLASSES,
        "class_shares": {c: float(s) for c, s in zip(CLASSES, shares)},
        "masked_fraction": float((~valid).mean()),
        "files": {"input": "input.png", "mask": "mask.png", **({"label": "label.png"} if evaluation else {})},
        "evaluation": evaluation,
    }
