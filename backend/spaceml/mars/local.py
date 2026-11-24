"""AI4Mars as local, validated image/mask pairs - what training and evaluation read.

Built once (release preparation, or Data & Training → Mars → Get dataset), then used offline:

    python -m spaceml.mars.local build --train 3000 --test all          # from Hugging Face
    python -m spaceml.mars.local build --from-nasa /path/ai4mars-dataset-merged-0.1
    python -m spaceml.mars.local check                                  # preflight

Layout (data/mars/ai4mars/):
    index.json                      dataset id, revision, schema, per-split counts, every pair
    <split>/img/<key>.png           8-bit grayscale NAVCAM image, long side <= --size
    <split>/label/<key>.png         class ids 0 soil 1 bedrock 2 sand 3 big rock, 255 = no label
                                    (rover body and >30 m range already set to 255)

Why: streaming from Hugging Face during training tied every run to the network and to the
mirror's row schema. The schema is checked here once, per row, with explicit counts of what
was skipped (unlabeled rows are normal in the train split and are skipped, not fatal).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.progress import fail, report, step
from ..common.utils import now_iso, read_json, write_json
from ..config import MARS_DIR
from .data import CLASSES, HF_DATASET, IGNORE, _arr, _first

ROOT = MARS_DIR / "ai4mars"
INDEX = ROOT / "index.json"
VALID = {0, 1, 2, 3, IGNORE}
LABEL_KEYS = ("label_mask", "label", "labels", "mask", "annotation")
IMAGE_KEYS = ("image", "img", "jpg", "png")


class SchemaError(ValueError):
    pass


# ------------------------------------------------------------------ row → validated pair
def row_to_pair(row: dict) -> tuple[Image.Image, np.ndarray] | str:
    """Validated (image, label) or the reason the row is skipped. Raises SchemaError only when
    the row layout itself is unknown (wrong dataset/split), never for an ordinary unlabeled row."""
    image = _first(row, *IMAGE_KEYS)
    if image is None:
        raise SchemaError(f"no image column; row keys: {sorted(row)}")
    if row.get("has_labels") is False:
        return "unlabeled"
    if not any(k in row for k in LABEL_KEYS):
        raise SchemaError(f"no label column (expected one of {LABEL_KEYS}); row keys: {sorted(row)}")
    lab = _first(row, *LABEL_KEYS)
    if lab is None:
        return "unlabeled"
    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image))
    label = _arr(lab)
    if label is None or label.ndim != 2:
        return "bad_mask_shape"
    label = label.astype(np.int64)
    if label.shape != (image.height, image.width):
        return "size_mismatch"
    for key in ("rover_mask", "range_mask"):
        m = _arr(row.get(key))
        if m is not None and m.shape == label.shape:
            label[m > 0] = IGNORE
    bad = set(np.unique(label).tolist()) - VALID
    if bad:
        if bad - set(range(4, 255)):
            return "invalid_class_ids"
        label[(label > 3) & (label != IGNORE)] = IGNORE
    if (label != IGNORE).mean() < 0.05:
        return "almost_no_labels"
    return image, label


def _save(split: str, key: str, image: Image.Image, label: np.ndarray, size: int) -> dict:
    g = image.convert("L")
    w, h = g.size
    f = min(1.0, size / max(w, h))
    if f < 1:
        nw, nh = max(8, round(w * f)), max(8, round(h * f))
        g = g.resize((nw, nh), Image.BILINEAR)
        lab = Image.fromarray(label.astype(np.uint8)).resize((nw, nh), Image.NEAREST)
    else:
        lab = Image.fromarray(label.astype(np.uint8))
    (ROOT / split / "img").mkdir(parents=True, exist_ok=True)
    (ROOT / split / "label").mkdir(parents=True, exist_ok=True)
    g.save(ROOT / split / "img" / f"{key}.png", optimize=True)
    lab.save(ROOT / split / "label" / f"{key}.png", optimize=True)
    a = np.asarray(lab)
    shares = np.bincount(a[a != IGNORE].ravel(), minlength=4)[:4] / max(int((a != IGNORE).sum()), 1)
    return {"key": key, "split": split, "w": g.width, "h": g.height, "img_sha": pixel_sha(g),
            "labeled": round(float((a != IGNORE).mean()), 4), "shares": [round(float(x), 4) for x in shares]}


def pixel_sha(img: Image.Image) -> str:
    """Identity of an image = hash of its decoded pixels (+ size/mode). Row numbers are NOT identities:
    without a file_name, every split numbers its rows from 0, so train/row000042 and test_min3/row000042 are
    different images. The same image can have different masks in different splits, so only pixels count."""
    h = hashlib.sha256(f"{img.mode}|{img.width}x{img.height}|".encode())
    h.update(np.asarray(img).tobytes())
    return h.hexdigest()[:32]


def _key(name: str, i: int) -> str:
    base = "".join(c for c in Path(str(name)).stem if c.isalnum() or c in "-_")[:60]
    return base or f"row{i:06d}"


# ------------------------------------------------------------------ builders
def build_from_hf(splits: dict[str, int | None], size: int = 512, dataset: str = HF_DATASET) -> dict:
    from datasets import load_dataset
    revision = None
    try:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(dataset).sha
    except Exception as e:  # noqa: BLE001
        print(f"(could not read the dataset revision: {e})")
    idx = _load_or_new(dataset, revision, size)
    for split, want in splits.items():
        step(f"AI4Mars {split}: downloading and validating image/mask pairs")
        try:
            ds = load_dataset(dataset, split=split, streaming=True, revision=revision)
            it = iter(ds)
            first = next(it)
        except Exception as e:  # noqa: BLE001
            fail(f"source_unavailable: Could not download AI4Mars from Hugging Face (huggingface.co). Check the internet "
                 f"connection or proxy, then press Retry. Details: {dataset} [{split}]: {type(e).__name__}: {str(e)[:200]}")
        print(f"  row keys: {sorted(first)}", flush=True)
        import itertools
        have = {p["key"] for p in idx["pairs"] if p["split"] == split}
        stats = {"rows_read": 0, "kept": len(have), "skipped": {}}
        for i, row in enumerate(itertools.chain([first], it)):
            if want and stats["kept"] >= want:
                break
            stats["rows_read"] += 1
            key = _key(_first(row, "file_name", "name", "id") or "", i)
            if key in have:
                continue
            try:
                res = row_to_pair(row)
            except SchemaError as e:
                fail(f"AI4Mars schema problem in {dataset} [{split}] (revision {revision or 'unknown'}): {e}")
            if isinstance(res, str):
                stats["skipped"][res] = stats["skipped"].get(res, 0) + 1
                continue
            idx["pairs"].append(_save(split, key, *res, size))
            have.add(key)
            stats["kept"] += 1
            if stats["kept"] % 25 == 0:
                report(stats["kept"], want or max(stats["kept"] + 1, 1000),
                       f"{split}: {stats['kept']} pairs kept, {sum(stats['skipped'].values())} rows skipped")
                _write(idx)
        idx["splits"][split] = {**stats, "requested": want, "pairs": stats["kept"]}
        print(f"  {split}: kept {stats['kept']} pairs; skipped {stats['skipped'] or 'none'}", flush=True)
        if stats["kept"] == 0:
            fail(f"AI4Mars [{split}] has no usable image/mask pairs ({stats['skipped']}). Check the dataset id/split.")
        _write(idx)
    return idx


def build_from_nasa(root: str, size: int = 512, max_train: int | None = None) -> dict:
    """The official data.nasa.gov release (ai4mars-dataset-merged-0.1/msl)."""
    r = Path(root)
    msl = r / "msl" if (r / "msl").exists() else r
    idx = _load_or_new("data.nasa.gov AI4Mars ai4mars-dataset-merged-0.1", None, size)
    jobs = [("train", sorted((msl / "labels" / "train").glob("*.png"))[: max_train or None])]
    jobs += [(f"test_{s}", sorted((msl / "labels" / "test" / f"masked-gold-{s}-100agree").glob("*.png")))
             for s in ("min3", "min2", "min1")]
    for split, files in jobs:
        stats = {"rows_read": 0, "kept": 0, "skipped": {}}
        for f in files:
            stats["rows_read"] += 1
            stem = f.stem.replace("_merged", "")
            img = msl / "images" / "edr" / f"{stem}.JPG"
            if not img.exists():
                stats["skipped"]["image_missing"] = stats["skipped"].get("image_missing", 0) + 1
                continue
            row = {"image": Image.open(img), "label_mask": np.array(Image.open(f))}
            for name, sub in (("rover_mask", "rov"), ("range_mask", "range30m")):
                mp = msl / "images" / "mxy" / f"{stem.replace('EDR', 'MXY')}.png" if sub == "rov" else \
                    msl / "images" / "rng-30m" / f"{stem.replace('EDR', 'RNG')}.png"
                if mp.exists():
                    row[name] = np.array(Image.open(mp))
            res = row_to_pair(row)
            if isinstance(res, str):
                stats["skipped"][res] = stats["skipped"].get(res, 0) + 1
                continue
            idx["pairs"].append(_save(split, _key(stem, stats["rows_read"]), *res, size))
            stats["kept"] += 1
            report(stats["kept"], len(files), f"{split}: {stats['kept']}/{len(files)}")
        if files:
            idx["splits"][split] = {**stats, "pairs": stats["kept"]}
        _write(idx)
    return idx


def _load_or_new(dataset: str, revision: str | None, size: int) -> dict:
    idx = read_json(INDEX, None)
    if idx and idx.get("dataset") == dataset and idx.get("size") == size:
        return idx
    return {"dataset": dataset, "revision": revision, "size": size, "created_at": now_iso(),
            "classes": CLASSES, "ignore": IGNORE, "splits": {}, "pairs": []}


def _write(idx: dict) -> None:
    idx["updated_at"] = now_iso()
    write_json(INDEX, idx)


# ------------------------------------------------------------------ reading
def index() -> dict | None:
    return read_json(INDEX, None)


def pairs(split: str) -> list[dict]:
    return [p for p in (index() or {}).get("pairs", []) if p["split"] == split]


def splits() -> dict:
    out: dict[str, int] = {}
    for p in (index() or {}).get("pairs", []):
        out[p["split"]] = out.get(p["split"], 0) + 1
    return out


def load_pair(p: dict) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(ROOT / p["split"] / "img" / f"{p['key']}.png").convert("RGB")
    lab = np.asarray(Image.open(ROOT / p["split"] / "label" / f"{p['key']}.png"), np.int64)
    return img, lab


def img_path(p: dict) -> Path:
    return ROOT / p["split"] / "img" / f"{p['key']}.png"


def label_path(p: dict) -> Path:
    return ROOT / p["split"] / "label" / f"{p['key']}.png"


def reindex_from_files() -> dict | None:
    """Rebuild index.json from the PNG pairs on disk (no download) when the manifest is missing."""
    if not ROOT.exists():
        return None
    found = []
    for sd in sorted(d for d in ROOT.iterdir() if d.is_dir() and (d / "img").is_dir()):
        for ip in sorted((sd / "img").glob("*.png")):
            if (sd / "label" / ip.name).exists():
                found.append({"key": ip.stem, "split": sd.name})
    if not found:
        return None
    idx = {"dataset": "local files (index rebuilt)", "revision": None, "size": None, "classes": CLASSES,
           "ignore": IGNORE, "splits": {}, "pairs": found, "created_at": now_iso(), "reindexed": True}
    write_json(INDEX, idx)
    return idx


def validate(force: bool = False, progress: bool = False) -> dict:
    """Check every pair in the manifest and store the result in it (the dataset page, preflight, trainer and
    evaluation all read this one manifest).

    Per pair: both files exist and decode, image and mask dimensions match, mask values are class ids
    {0,1,2,3} or the ignore label 255, and the mask has labeled pixels. Every image gets a pixel hash; train
    images whose pixels also appear in an expert test split are held out of training (reported, not fatal).
    Results are cached in index.json and recomputed only when files change (or force=True).
    """
    idx = index() or reindex_from_files()
    if not idx:
        return {"ok": False, "root": str(ROOT), "index": str(INDEX), "error": f"no AI4Mars manifest at {INDEX}",
                "splits": {}}
    ps = idx.get("pairs", [])
    changed = False
    for n, p in enumerate(ps):
        ip, lp = img_path(p), label_path(p)
        sig = f"{ip.stat().st_size}:{lp.stat().st_size}" if ip.exists() and lp.exists() else "missing"
        if not force and p.get("_sig") == sig and "usable" in p and p.get("img_sha"):
            continue
        changed = True
        p["_sig"], p["problem"] = sig, None
        try:
            if sig == "missing":
                raise ValueError(f"missing file {'image' if not ip.exists() else 'mask'}")
            img = Image.open(ip)
            img.load()
            lab = np.asarray(Image.open(lp))
            if lab.ndim != 2 or lab.shape != (img.height, img.width):
                raise ValueError(f"image {img.width}x{img.height} vs mask {lab.shape[1] if lab.ndim > 1 else '?'}x{lab.shape[0]}")
            bad = set(np.unique(lab).tolist()) - VALID
            if bad:
                raise ValueError(f"invalid class ids {sorted(bad)[:6]}")
            if not (lab != IGNORE).any():
                raise ValueError("mask has no labeled pixels")
            p["img_sha"] = p.get("img_sha") if (p.get("img_sha") and not force) else pixel_sha(img)
            p["w"], p["h"] = img.width, img.height
            p["usable"] = True
        except Exception as e:  # noqa: BLE001
            p["usable"], p["problem"] = False, f"{type(e).__name__}: {e}"[:200]
        if progress and n % 50 == 0:
            report(n, len(ps), f"validating pair {n}/{len(ps)}")
    test_sha = {p["img_sha"] for p in ps if p["split"].startswith("test") and p.get("usable") and p.get("img_sha")}
    for p in ps:
        held = p["split"] == "train" and p.get("img_sha") in test_sha
        if bool(p.get("held_out")) != held:
            changed = True
        p["held_out"] = "same image as an expert test pair" if held else None
    summary: dict = {}
    for p in ps:
        s_ = summary.setdefault(p["split"], {"pairs": 0, "usable": 0, "held_out": 0, "problems": {}})
        s_["pairs"] += 1
        if p.get("usable") and not p.get("held_out"):
            s_["usable"] += 1
        if p.get("held_out"):
            s_["held_out"] += 1
        if p.get("problem"):
            k = p["problem"].split(":")[0] if ":" in p["problem"] else p["problem"]
            s_["problems"][k] = s_["problems"].get(k, 0) + 1
    examples = [f"{p['split']}/{p['key']}: {p['problem']}" for p in ps if p.get("problem")][:5]
    out = {"ok": summary.get("train", {}).get("usable", 0) >= 16, "root": str(ROOT), "index": str(INDEX),
           "splits": summary, "problem_examples": examples, "validated_at": now_iso() if changed else
           (idx.get("validation") or {}).get("validated_at")}
    if changed or idx.get("validation") != out:
        idx["validation"] = out
        write_json(INDEX, idx)
    return out


def usable_pairs(split: str) -> list[dict]:
    """Pairs of a split that passed validate() and are not held out (what training/evaluation read)."""
    v = (index() or {}).get("validation")
    if not v:
        validate()
    return [p for p in pairs(split) if p.get("usable") and not p.get("held_out")]


def test_split() -> str | None:
    s = splits()
    for name in ("test_min3", "test_min2", "test_min1", "test"):
        if s.get(name):
            return name
    return None


def expert_splits() -> list[str]:
    return [k for k in splits() if k.startswith("test")]


def preflight(split: str = "train", n: int = 8) -> dict:
    """Read n pairs and check dimensions and class ids. Raises SchemaError with the reason."""
    ps = usable_pairs(split)
    if not ps:
        raise SchemaError(f"no usable '{split}' pairs in {INDEX}")
    rng = np.random.default_rng(0)
    for p in [ps[i] for i in rng.choice(len(ps), min(n, len(ps)), replace=False)]:
        img, lab = load_pair(p)
        if lab.shape != (img.height, img.width):
            raise SchemaError(f"{p['key']}: image {img.size} vs mask {lab.shape[::-1]}")
        bad = set(np.unique(lab).tolist()) - VALID
        if bad:
            raise SchemaError(f"{p['key']}: invalid class ids {sorted(bad)}")
    return {"split": split, "pairs": len(ps), "checked": min(n, len(ps))}


def fingerprint() -> str:
    """Stable id of the local dataset (recorded in model manifests)."""
    idx = index() or {}
    h = hashlib.sha256(f"{idx.get('dataset')}|{idx.get('revision')}|{idx.get('size')}".encode())
    for p in sorted(idx.get("pairs", []), key=lambda x: (x["split"], x["key"])):
        h.update(f"{p['split']}/{p['key']}".encode())
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description="Local AI4Mars image/mask pairs")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train", default="3000", help="number of train pairs, or 'all'")
    b.add_argument("--test", default="all", help="number of expert test pairs (test_min3), or 'all'")
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--dataset", default=HF_DATASET)
    b.add_argument("--from-nasa", default=None, help="official release folder instead of Hugging Face")
    sub.add_parser("check")
    a = p.parse_args()
    if a.cmd == "check":
        import json, os
        v = validate(force=True)
        print(json.dumps({"cwd": os.getcwd(), **v}, indent=2))
        if not v["ok"]:
            fail(f"AI4Mars local data is not trainable: {v.get('error') or v['splits'].get('train')}")
        return
    n = lambda v: None if str(v).lower() == "all" else int(v)
    if a.from_nasa:
        build_from_nasa(a.from_nasa, a.size, n(a.train))
    else:
        build_from_hf({"test_min3": n(a.test), "train": n(a.train)}, a.size, a.dataset)
    v = validate()
    print("AI4Mars local dataset:", {k: f"{c['usable']}/{c['pairs']} usable" for k, c in v["splits"].items()})


if __name__ == "__main__":
    main()
