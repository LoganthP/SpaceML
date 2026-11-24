"""Mars data path: mirror-shaped rows (incl. unlabeled ones) → validated local pairs → preflight →
train (quick) → evaluate → samples, all without network. Rows are generated to the
hassanjbara/AI4MARS schema (image, label_mask, rover_mask, range_mask, has_labels, file_name)."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from PIL import Image


def _rows(n, labeled_every=2, size=96, seed=0, prefix="NLB"):
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = Image.fromarray((rng.random((size, size)) * 255).astype(np.uint8))
        labeled = i % labeled_every == 0
        lab = np.zeros((size, size), np.uint8)
        lab[size // 2:] = 1
        lab[:, : size // 4] = 2
        lab[:4] = 255
        rover = np.zeros((size, size), np.uint8)
        rover[-6:] = 1
        yield {"image": img, "label_mask": lab if labeled else None, "rover_mask": rover,
               "range_mask": np.zeros((size, size), np.uint8), "has_labels": labeled, "file_name": f"{prefix}_{i:05d}EDR.JPG"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in ("DATA", "MODELS", "STORAGE"):
        monkeypatch.setenv(f"SPACEML_{k}_DIR", str(tmp_path / k.lower()))
    for m in [m for m in sys.modules if m.startswith(("spaceml", "app"))]:
        del sys.modules[m]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda name, split, streaming=True, revision=None: _rows(40 if split == "train" else 12,
                                                                                 2 if split == "train" else 1,
                                                                                 seed=1 if split == "train" else 2,
                                                                                 prefix="NLA" if split == "train" else "NLB"))
    return tmp_path


def test_unlabeled_rows_are_skipped_not_fatal(env):
    local = importlib.import_module("spaceml.mars.local")
    idx = local.build_from_hf({"test_min3": None, "train": None}, size=64)
    assert idx["splits"]["train"]["pairs"] == 20
    assert idx["splits"]["train"]["skipped"] == {"unlabeled": 20}
    assert idx["splits"]["test_min3"]["pairs"] == 12
    pf = local.preflight("train")
    assert pf["pairs"] == 20
    img, lab = local.load_pair(local.pairs("train")[0])
    assert lab.shape == (img.height, img.width) and set(np.unique(lab)) <= {0, 1, 2, 255}
    assert (lab[-3:] == 255).all()          # rover mask became "no label"


def test_unknown_schema_fails_with_row_keys(env):
    local = importlib.import_module("spaceml.mars.local")
    with pytest.raises(local.SchemaError, match="row keys"):
        local.row_to_pair({"image": Image.new("L", (8, 8)), "something_else": 1})


def test_quick_train_evaluate_and_samples_offline(env, monkeypatch):
    pytest.importorskip("segmentation_models_pytorch")
    local = importlib.import_module("spaceml.mars.local")
    local.build_from_hf({"test_min3": None, "train": None}, size=64)
    train = importlib.import_module("spaceml.mars.train")
    monkeypatch.setattr(sys, "argv", ["train", "--quick", "--epochs", "1", "--img-size", "64", "--batch", "4",
                                      "--encoder", "resnet18", "--no-pretrained", "--version", "vtest"])
    train.main()
    from spaceml.config import MODELS_DIR
    vd = MODELS_DIR / "mars" / "vtest"
    assert (vd / "model.pt").exists() and not (vd / "last.pt").exists()
    ev = importlib.import_module("spaceml.mars.evaluate")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--version", "vtest"])
    ev.main()
    assert (vd / "metrics.json").exists()
    data = importlib.import_module("spaceml.mars.data")
    assert data.prepare_samples_local(5) and len(data.list_samples()) == 5


def test_resume_continues_from_last_epoch(env, monkeypatch):
    pytest.importorskip("segmentation_models_pytorch")
    import torch
    local = importlib.import_module("spaceml.mars.local")
    local.build_from_hf({"train": None}, size=64)
    train = importlib.import_module("spaceml.mars.train")
    from spaceml.config import MODELS_DIR
    real_save = torch.save
    calls = {"n": 0}

    def crash_after_first_epoch(obj, path, *a, **k):
        real_save(obj, path, *a, **k)
        if str(path).endswith("last.pt.tmp"):
            calls["n"] += 1
            if calls["n"] == 1:
                (MODELS_DIR / "mars" / ".staging-vres" / "last.pt.tmp").replace(MODELS_DIR / "mars" / ".staging-vres" / "last.pt")
                raise KeyboardInterrupt("simulated restart")
    monkeypatch.setattr(torch, "save", crash_after_first_epoch)
    monkeypatch.setattr(sys, "argv", ["train", "--quick", "--epochs", "2", "--img-size", "64", "--batch", "4",
                                      "--encoder", "resnet18", "--no-pretrained", "--version", "vres"])
    with pytest.raises(KeyboardInterrupt):
        train.main()
    assert (MODELS_DIR / "mars" / ".staging-vres" / "last.pt").exists()
    monkeypatch.setattr(torch, "save", real_save)
    monkeypatch.setattr(sys, "argv", ["train", "--resume"])
    train.main()
    import json
    m = json.loads((MODELS_DIR / "mars" / "vres" / "manifest.json").read_text())
    assert [h["epoch"] for h in m["history"]] == [1, 2]


def _no_id(rows):
    """The real mirror rows the user downloaded carry no file_name/name/id, so keys are row numbers."""
    for r in rows:
        r = dict(r)
        r.pop("file_name", None)
        yield r


def test_row_numbers_are_not_image_ids(env, monkeypatch):
    """Regression: train/row000005 and test_min3/row000005 are DIFFERENT images - training must run."""
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split, streaming=True, revision=None:
                        _no_id(_rows(24, 1, seed=1 if split == "train" else 2)))
    local = importlib.import_module("spaceml.mars.local")
    local.build_from_hf({"test_min3": None, "train": None}, size=64)
    keys = lambda sp: {p["key"] for p in local.pairs(sp)}  # noqa: E731
    assert keys("train") & keys("test_min3")                  # the row-number collision exists ...
    v = local.validate()
    assert v["splits"]["train"] == {"pairs": 24, "usable": 24, "held_out": 0, "problems": {}}   # ... and is harmless
    import subprocess
    r = subprocess.run([sys.executable, "-m", "spaceml.mars.train", "--quick", "--epochs", "1", "--img-size", "64"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "24 train images used" in r.stdout and "0 train duplicates" in r.stdout


def test_same_image_in_both_splits_is_held_out(env, monkeypatch):
    """The same pixels in train and expert test (even with another mask) are detected and kept out of training."""
    import datasets
    test_rows = list(_no_id(_rows(12, 1, seed=2)))

    def train_rows():
        for i, r in enumerate(_no_id(_rows(24, 1, seed=1))):
            if i in (3, 7, 11):                  # these three train images ARE expert test images
                r = dict(r, image=test_rows[i]["image"].copy(), label_mask=np.ones_like(np.asarray(r["label_mask"])))
            yield r
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split, streaming=True, revision=None:
                        iter(test_rows) if split == "test_min3" else train_rows())
    local = importlib.import_module("spaceml.mars.local")
    local.build_from_hf({"test_min3": None, "train": None}, size=64)
    v = local.validate()
    assert v["splits"]["train"]["held_out"] == 3 and v["splits"]["train"]["usable"] == 21
    used = {p["img_sha"] for p in local.usable_pairs("train")}
    assert not used & {p["img_sha"] for p in local.usable_pairs("test_min3")}


def test_manifest_from_previous_version_is_migrated_in_place(env, monkeypatch):
    """An index.json written before pixel hashes existed is validated and upgraded - no rebuild, files kept."""
    import datasets
    import json
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split, streaming=True, revision=None:
                        _no_id(_rows(20, 1, seed=1 if split == "train" else 2)))
    local = importlib.import_module("spaceml.mars.local")
    local.build_from_hf({"test_min3": None, "train": None}, size=64)
    idx = json.loads(local.INDEX.read_text())
    for p in idx["pairs"]:
        for k in ("img_sha", "usable", "held_out", "problem", "_sig"):
            p.pop(k, None)
    idx.pop("validation", None)
    local.INDEX.write_text(json.dumps(idx))
    files = sorted(x.name for x in (local.ROOT / "train" / "img").iterdir())
    # break one pair so validation has something to report
    (local.ROOT / "train" / "label" / f"{idx['pairs'][-1]['key']}.png").unlink()
    v = local.validate()
    assert v["splits"]["train"]["pairs"] == 20 and v["splits"]["train"]["usable"] == 19
    assert any("missing file mask" in e for e in v["problem_examples"])
    assert all(p.get("img_sha") for p in json.loads(local.INDEX.read_text())["pairs"] if p.get("usable"))
    assert sorted(x.name for x in (local.ROOT / "train" / "img").iterdir()) == files
    # missing manifest: rebuilt from the PNG pairs on disk
    local.INDEX.unlink()
    v = local.validate()
    assert v["splits"]["train"]["usable"] == 19 and v["splits"]["test_min3"]["usable"] == 20
