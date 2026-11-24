"""Error smoothing, nonparametric dynamic thresholding, pruning and event scoring.

Follows Hundman et al. (KDD 2018). Nothing in this file looks at the labeled
anomalies except `score_events`, which is only called for reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 250            # steps of history the LSTM sees (paper setting)
SMOOTHING_SPAN = 30     # EWMA span for prediction errors
Z_RANGE = np.arange(2.5, 10.5, 0.5)
PRUNE_P = 0.13          # minimum relative drop between consecutive sequence maxima
ZSCORE_K = 3.0          # baseline: |x - mean_train| > 3 std_train
ERROR_BUFFER = 100      # steps added around flagged errors (telemanom default)


def smooth_errors(y_true: np.ndarray, y_pred: np.ndarray, span: int = SMOOTHING_SPAN) -> np.ndarray:
    return pd.Series(np.abs(y_true - y_pred)).ewm(span=span).mean().to_numpy()


def to_sequences(flags: np.ndarray, offset: int = 0) -> list[list[int]]:
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [[int(s[0]) + offset, int(s[-1]) + offset] for s in splits]


def _buffered(flags: np.ndarray, buffer: int = ERROR_BUFFER) -> np.ndarray:
    """Expand every flagged step by +/- buffer steps (telemanom's error_buffer)."""
    idx = np.flatnonzero(flags)
    out = np.zeros(len(flags), bool)
    if len(idx) == 0:
        return out
    starts = np.clip(idx - buffer, 0, None)
    ends = np.clip(idx + buffer + 1, None, len(flags))
    diff = np.zeros(len(flags) + 1, int)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff)[:-1] > 0


def dynamic_threshold(e: np.ndarray, zs: np.ndarray = Z_RANGE) -> tuple[float, float]:
    """Pick epsilon = mean + z*std that best separates extreme errors (no labels used).

    Score from Hundman et al.: relative drop in mean and std of the errors once the
    flagged ones are removed, penalised by the number of flagged steps and sequences.
    """
    mu, sd = float(e.mean()), float(e.std())
    best_eps, best_z, best_score = mu + zs[-1] * sd, float(zs[-1]), -np.inf
    if sd == 0 or mu == 0:
        return best_eps, best_z
    for z in zs:
        eps = mu + z * sd
        below = e[e < eps]
        flags = _buffered(e >= eps)
        n_anom = int(flags.sum())
        if n_anom == 0 or len(below) == 0:
            continue
        n_seq = len(to_sequences(flags))
        score = ((mu - below.mean()) / mu + (sd - below.std()) / sd) / (n_seq ** 2 + n_anom)
        if score >= best_score and n_seq <= 5 and n_anom < 0.5 * len(e):
            best_eps, best_z, best_score = eps, float(z), score
    return float(best_eps), best_z


def prune(seqs: list[list[int]], e: np.ndarray, floor: float, p: float = PRUNE_P) -> list[list[int]]:
    """Drop sequences whose peak error is not clearly above the next one (paper, Sec. 3.3).

    `floor` is the largest error outside every (buffered) flagged sequence.
    """
    if not seqs:
        return seqs
    maxes = np.array([e[a:b + 1].max() for a, b in seqs])
    order = np.argsort(-maxes)
    sorted_max = np.append(maxes[order], floor)
    keep_upto = -1  # index in `order` of the last sequence followed by a big drop
    for i in range(len(order)):
        drop = (sorted_max[i] - sorted_max[i + 1]) / max(sorted_max[i], 1e-12)
        if drop >= p:
            keep_upto = i
    kept = set(order[: keep_upto + 1].tolist())
    return [s for i, s in enumerate(seqs) if i in kept]


def detect(y_true: np.ndarray, y_pred: np.ndarray, offset: int = WINDOW) -> dict:
    """Full unsupervised detection on one channel's test predictions.

    y_true / y_pred are aligned (prediction i corresponds to test step i + offset).
    Returns sequences in test-step coordinates.
    """
    e = smooth_errors(y_true, y_pred)
    eps, z = dynamic_threshold(e)
    # Paper's guard: ignore channels whose errors are tiny relative to the signal.
    spread = float(np.percentile(y_true, 95) - np.percentile(y_true, 5))
    quiet = e.max() <= 0.05 or (e.std() <= 0.05 * y_true.std() and e.max() <= 0.05 * spread)
    flags = np.zeros(len(e), bool) if quiet else _buffered(e >= eps)
    raw = [s for s in to_sequences(flags) if s[1] > s[0]]
    floor = float(e[~flags].max()) if (~flags).any() else 0.0
    kept = prune(raw, e, floor)
    return {
        "smoothed_error": e,
        "threshold": eps,
        "z": z,
        "sequences": [[a + offset, b + offset] for a, b in kept],
        "pruned": len(raw) - len(kept),
        "quiet_channel": bool(quiet),
    }


def zscore_baseline(train_values: np.ndarray, test_values: np.ndarray, k: float = ZSCORE_K) -> dict:
    mu, sd = float(train_values.mean()), float(train_values.std())
    sd = sd if sd > 1e-8 else 1e-8
    z = np.abs(test_values - mu) / sd
    return {"mean": mu, "std": sd, "k": k, "sequences": to_sequences(z > k), "z": z}


def _hit(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score_events(pred: list[list[int]], true: list[list[int]]) -> dict:
    """Event-level scoring: a labeled anomaly is found if any flagged sequence overlaps it."""
    found = [any(_hit(p, t) for p in pred) for t in true]
    tp = int(sum(found))
    fp = int(sum(not any(_hit(p, t) for t in true) for p in pred))
    return {"tp": tp, "fp": fp, "fn": len(true) - tp, "found": found}


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
