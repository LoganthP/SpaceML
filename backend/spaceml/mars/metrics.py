from __future__ import annotations

import numpy as np

from .data import IGNORE

N_CLASSES = 4


def confusion(pred: np.ndarray, label: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    ok = label != IGNORE
    t, p = label[ok].astype(np.int64), pred[ok].astype(np.int64)
    return np.bincount(n * t + p, minlength=n * n).reshape(n, n)


def summarise(cm: np.ndarray) -> dict:
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    present = cm.sum(1) > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {
        "iou": iou,
        "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "pixel_acc": float(inter.sum() / max(cm.sum(), 1)),
        "present": present,
    }
