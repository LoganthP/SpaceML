"""Run a trained channel model on the held-out test telemetry."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..common.utils import clean_floats
from . import data
from .detection import WINDOW, detect, score_events, zscore_baseline
from .model import load_checkpoint, predict


@lru_cache(maxsize=16)
def _model(path: str, mtime: float):
    return load_checkpoint(Path(path))[0]


def analyse_channel(chan_id: str, version_dir: Path) -> dict:
    """Core computation used by both the API and the evaluation script."""
    chans = data.load_channels()
    if chan_id not in chans:
        raise ValueError(f"Unknown channel {chan_id}")
    ch = chans[chan_id]
    ckpt = version_dir / f"{chan_id}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model for {chan_id} in {version_dir.name}. "
                                f"Run: task:telemetry.train?channels={chan_id}")
    train, test = data.load_arrays(chan_id)
    model = _model(str(ckpt), ckpt.stat().st_mtime)
    y_pred = predict(model, test)
    y_true = test[WINDOW:, 0]
    det = detect(y_true, y_pred)
    base = zscore_baseline(train[:, 0], test[:, 0])
    return {
        "channel": ch, "train": train, "test": test, "y_pred": y_pred, "det": det, "base": base,
        "model_score": score_events(det["sequences"], ch.sequences),
        "baseline_score": score_events(base["sequences"], ch.sequences),
    }


def run(chan_id: str, version_dir: Path) -> dict:
    """API result: everything the timeline view needs."""
    a = analyse_channel(chan_id, version_dir)
    ch, test = a["channel"], a["test"]
    n = len(test)
    predicted = np.full(n, np.nan)
    predicted[WINDOW:] = a["y_pred"]
    err = np.full(n, np.nan)
    err[WINDOW:] = a["det"]["smoothed_error"]
    flagged = a["det"]["sequences"]
    return {
        "chan_id": chan_id,
        "spacecraft": ch.spacecraft,
        "n_steps": n,
        "window": WINDOW,
        "actual": clean_floats(test[:, 0], 4),
        "predicted": clean_floats(predicted, 4),
        "smoothed_error": clean_floats(err, 5),
        "threshold": a["det"]["threshold"],
        "threshold_z": a["det"]["z"],
        "quiet_channel": a["det"]["quiet_channel"],
        "pruned_sequences": a["det"]["pruned"],
        "flagged": [{"start": s, "end": e, "peak_error": float(np.nanmax(err[s:e + 1]))} for s, e in flagged],
        "baseline": {
            "name": f"z-score > {a['base']['k']:g} (train mean/std)",
            "flagged": [{"start": s, "end": e} for s, e in a["base"]["sequences"]],
            "upper": a["base"]["mean"] + a["base"]["k"] * a["base"]["std"],
            "lower": a["base"]["mean"] - a["base"]["k"] * a["base"]["std"],
        },
        "labeled": [{"start": s, "end": e, "class": c, "found_by_model": f, "found_by_baseline": fb}
                    for (s, e), c, f, fb in zip(ch.sequences, ch.classes, a["model_score"]["found"],
                                                a["baseline_score"]["found"])],
        "score": {k: a["model_score"][k] for k in ("tp", "fp", "fn")},
        "baseline_score": {k: a["baseline_score"][k] for k in ("tp", "fp", "fn")},
        "n_commands": int(test.shape[1] - 1),
    }
