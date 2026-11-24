"""Code-path tests. Synthetic arrays here only exercise the pipelines; the app itself
only ever serves real NASA data. Run from backend/:  python -m pytest -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- telemetry
def test_detection_finds_injected_error_burst():
    from spaceml.telemetry.detection import detect, score_events

    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 60, 4000)).astype(np.float32)
    pred = y + rng.normal(0, 0.02, len(y)).astype(np.float32)
    pred[2000:2060] += 0.8  # the model is badly wrong for 60 steps
    det = detect(y, pred, offset=250)
    assert det["sequences"], "burst should be flagged"
    s = score_events(det["sequences"], [[2250, 2310]])
    assert s["tp"] == 1 and s["fp"] == 0


def test_event_scoring_counts_overlaps():
    from spaceml.telemetry.detection import prf, score_events

    s = score_events([[10, 20], [100, 110]], [[15, 30], [200, 210]])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
    assert prf(1, 1, 1)["precision"] == 0.5


def test_windows_align_with_targets():
    from spaceml.telemetry.model import make_windows

    x = np.arange(300 * 2, dtype=np.float32).reshape(300, 2)
    X, y = make_windows(x, window=250)
    assert X.shape == (50, 250, 2)
    assert float(y[0]) == x[250, 0] and float(X[0, -1, 0]) == x[249, 0]


# ----------------------------------------------------------------- exoplanet
def _transit_lc(period=3.5, t0=1.2, dur_hr=3.0, depth=0.01, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 120, n)
    f = 1 + 0.002 * np.sin(t / 7) + rng.normal(0, 4e-4, n)
    phase = (t - t0 + period / 2) % period - period / 2
    f[np.abs(phase) < dur_hr / 48] -= depth
    return t, f.astype(np.float32)


def test_views_are_centered_and_normalised():
    from spaceml.exoplanet.data import GLOBAL_BINS, LOCAL_BINS, make_views

    t, f = _transit_lc()
    v = make_views(t, f, 3.5, 1.2, 3.0)
    assert v["global"].shape == (GLOBAL_BINS,) and v["local"].shape == (LOCAL_BINS,)
    assert abs(v["local"].min() + 1) < 1e-6
    assert abs(int(np.argmin(v["local"])) - LOCAL_BINS // 2) < 15  # dip at phase 0
    assert 7000 < v["measured_depth_ppm"] < 13000  # flattening kept the dip (transit was masked)


def _env(tmp: Path) -> dict:
    return {**os.environ, "SPACEML_DATA_DIR": str(tmp / "data"), "SPACEML_MODELS_DIR": str(tmp / "models"),
            "SPACEML_STORAGE_DIR": str(tmp / "storage"), "PYTHONPATH": str(BACKEND)}


@pytest.mark.slow
def test_exoplanet_train_and_infer_end_to_end(tmp_path):
    """Builds a small cache of views, trains CNN + baseline, then scores one TCE."""
    from spaceml.exoplanet import data as exo

    data_dir = tmp_path / "data" / "exoplanet"
    (data_dir / "views").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(1)
    for i in range(120):
        kepid, planet = 1000 + i, i % 3 == 0
        period, dur = float(rng.uniform(2, 20)), float(rng.uniform(2, 6))
        t, f = _transit_lc(period, 1.0, dur, 0.008 if planet else 0.02, n=8000, seed=i)
        if not planet:  # eclipsing-binary-like secondary eclipse at phase 0.5
            ph = (t - 1.0) % period
            f[np.abs(ph - period / 2) < dur / 48] -= 0.01
        v = exo.make_views(t, f, period, 1.0, dur)
        np.savez_compressed(data_dir / "views" / f"{kepid}_1.npz", **v)
        rows.append({"kepid": kepid, "tce_plnt_num": 1, "tce_period": period, "tce_time0bk": 1.0,
                     "tce_duration": dur, "tce_depth": 8000 if planet else 30000, "tce_model_snr": 20,
                     "tce_prad": 2.0, "av_training_set": "PC" if planet else "AFP"})
    pd.DataFrame(rows).to_csv(data_dir / "q1_q17_dr24_tce.csv", index=False)

    r = subprocess.run([sys.executable, "-m", "spaceml.exoplanet.train", "--epochs", "3", "--version", "vtest"],
                       cwd=BACKEND, env=_env(tmp_path), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    vdir = tmp_path / "models" / "exoplanet" / "vtest"
    metrics = json.loads((vdir / "metrics.json").read_text())
    assert metrics["rows"][0]["metrics"][0]["name"] == "ROC-AUC"
    code = ("from pathlib import Path; from spaceml.exoplanet import infer; "
            f"r = infer.run(1000, 1, Path(r'{vdir}')); print(r['candidate_score'], r['split'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    score, split = r.stdout.split()[-2:]
    assert 0 <= float(score) <= 1 and split in ("train", "val", "test")


# ----------------------------------------------------------------- mars
@pytest.mark.slow
def test_mars_inference_writes_mask(tmp_path):
    import torch
    from PIL import Image

    from spaceml.mars import infer
    from spaceml.mars.model import build_model

    vdir = tmp_path / "v"
    vdir.mkdir()
    torch.save({"state_dict": build_model(encoder_weights=None).state_dict(), "img_size": 256}, vdir / "model.pt")
    img = Image.fromarray((np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8))
    img.save(tmp_path / "in.png")
    label = np.full((300, 400), 255, np.uint8)
    label[200:, :] = 0
    label[200:, :100] = 2
    Image.fromarray(label).save(tmp_path / "label.png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    res = infer.run(tmp_path / "in.png", vdir, run_dir, tmp_path / "label.png")
    mask = np.array(Image.open(run_dir / "mask.png"))
    assert mask.shape == (300, 400) and mask.max() <= 3
    assert abs(sum(res["class_shares"].values()) - 1) < 1e-6
    assert res["evaluation"]["baseline"]["iou"]["soil"] is not None
